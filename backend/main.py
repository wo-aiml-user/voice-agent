import uvicorn
import os
import asyncio
import base64
import json
import logging
import time
import websockets
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging to file
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"voice_agent_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Assistant API - Streaming", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPGRAM_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("API keys for Deepgram and DeepSeek must be set in .env file")

# Initialize Deepgram client with keepalive for streaming
dg_config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, dg_config)

# Configure DeepSeek client (OpenAI-compatible API) - ASYNC VERSION
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# TTS Configuration - OPTIMIZED FOR LOW LATENCY
TTS_MODEL = "aura-2-thalia-en"
TTS_SAMPLE_RATE = 24000
TTS_ENCODING = "linear16"
SEND_EVERY_CHARS = 15  # Reduced from 50 for faster first audio

# System prompt - OPTIMIZED for shorter responses
SYSTEM_PROMPT = (
    "You are a succinct, helpful voice assistant. "
    "Respond in 2-3 sentences. Be direct and friendly."
)


class StreamingSession:
    """Manages a streaming voice session with STT, LLM, and TTS pipelines."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.dg_connection = None
        self.tts_websocket = None
        self.is_active = True
        self.final_transcript = ""
        self.is_processing = False
        self.is_speaking = False
        
        # Latency tracking
        self.rtt_start_ts: Optional[float] = None
        self.llm_start_ts: Optional[float] = None
        self.tts_start_ts: Optional[float] = None
        self.first_llm_token_logged = False
        self.first_audio_logged = False
        
        # TTS token queue for micro-batching
        self.tts_token_queue: asyncio.Queue[str] = asyncio.Queue()
        self.tts_sender_task: Optional[asyncio.Task] = None
        self.tts_receiver_task: Optional[asyncio.Task] = None
        
        # Persistent TTS WebSocket state
        self.tts_connected = False
        
        # Audio buffer for handling speech during playback
        self.pending_audio_buffer: list[bytes] = []
        
    async def send_message(self, msg_type: str, data: dict = None):
        """Send a JSON message to the frontend."""
        if not self.is_active:
            return
        try:
            message = {"type": msg_type, **(data or {})}
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[{self.session_id}] Error sending message: {e}")
    
    async def start_stt_stream(self):
        """Initialize Deepgram live transcription with VAD and interim results."""
        try:
            self.dg_connection = deepgram.listen.asyncwebsocket.v("1")
            
            # Event handlers
            async def on_message(self_conn, result, **kwargs):
                try:
                    transcript = result.channel.alternatives[0].transcript
                    if not transcript:
                        return
                    
                    is_final = result.is_final
                    speech_final = result.speech_final
                    
                    if is_final:
                        self.final_transcript += " " + transcript
                        self.final_transcript = self.final_transcript.strip()
                        await self.send_message("interim_transcript", {
                            "text": self.final_transcript,
                            "is_final": True
                        })
                        logger.info(f"[{self.session_id}] STT | Final segment: '{transcript}'")
                        
                        # If speech_final (endpoint detected), trigger processing
                        if speech_final and self.final_transcript and not self.is_processing:
                            # Start RTT timer
                            self.rtt_start_ts = time.perf_counter()
                            self.is_processing = True
                            asyncio.create_task(self.process_and_respond())
                    else:
                        # Interim result - show partial transcript
                        partial = self.final_transcript + " " + transcript if self.final_transcript else transcript
                        await self.send_message("interim_transcript", {
                            "text": partial.strip(),
                            "is_final": False
                        })
                except Exception as e:
                    logger.error(f"[{self.session_id}] STT | Error in on_message: {e}")
            
            async def on_speech_started(self_conn, speech_started, **kwargs):
                logger.info(f"[{self.session_id}] STT | Speech started")
                await self.send_message("speech_started")
                
                # If user starts speaking while assistant is speaking, stop playback
                if self.is_speaking:
                    logger.info(f"[{self.session_id}] STT | User barged in - stopping playback")
                    await self.handle_barge_in()
            
            async def on_utterance_end(self_conn, utterance_end, **kwargs):
                logger.info(f"[{self.session_id}] STT | Utterance end detected")
                if self.final_transcript and not self.is_processing:
                    self.rtt_start_ts = time.perf_counter()
                    self.is_processing = True
                    asyncio.create_task(self.process_and_respond())
            
            async def on_error(self_conn, error, **kwargs):
                logger.error(f"[{self.session_id}] STT | Error: {error}")
                await self.send_message("error", {"message": f"STT error: {str(error)}"})
            
            async def on_close(self_conn, close, **kwargs):
                logger.info(f"[{self.session_id}] STT | Connection closed")
            
            # Register event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
            
            # Configure live transcription options with VAD - OPTIMIZED TIMING
            options = LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                punctuate=True,
                interim_results=True,
                utterance_end_ms=2000,  # Reduced from 2000 for faster response
                vad_events=True,
                endpointing=200,  # Reduced from 300 for faster endpoint detection
            )
            
            # Start the connection
            if await self.dg_connection.start(options):
                logger.info(f"[{self.session_id}] STT | Stream started successfully")
                await self.send_message("stt_ready")
                return True
            else:
                logger.error(f"[{self.session_id}] STT | Failed to start stream")
                return False
                
        except Exception as e:
            import traceback
            logger.error(f"[{self.session_id}] STT | Error starting: {type(e).__name__}: {e}")
            logger.error(f"[{self.session_id}] STT | Traceback: {traceback.format_exc()}")
            await self.send_message("error", {"message": f"Failed to start STT: {str(e)}"})
            return False
    
    async def handle_barge_in(self):
        """Handle user interruption (barge-in) during assistant playback."""
        self.is_speaking = False
        self.first_audio_logged = False
        
        # Clear the TTS queue
        while not self.tts_token_queue.empty():
            try:
                self.tts_token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Notify frontend to stop playback
        await self.send_message("stop_playback")
        logger.info(f"[{self.session_id}] BARGE-IN | Playback interrupted by user")
    
    async def send_audio_to_stt(self, audio_data: bytes):
        """Forward audio data to Deepgram STT stream."""
        # Always accept audio - don't drop during playback
        # Let barge-in detection handle interruptions
        if self.dg_connection:
            try:
                await self.dg_connection.send(audio_data)
            except Exception as e:
                logger.error(f"[{self.session_id}] STT | Error sending audio: {e}")
    
    async def start_tts_websocket(self):
        """Initialize PERSISTENT WebSocket connection to Deepgram TTS (Aura-2)."""
        if self.tts_connected and self.tts_websocket:
            logger.info(f"[{self.session_id}] TTS | Reusing existing WebSocket")
            return True
            
        try:
            tts_url = (
                f"wss://api.deepgram.com/v1/speak?"
                f"model={TTS_MODEL}&encoding={TTS_ENCODING}&sample_rate={TTS_SAMPLE_RATE}"
            )
            
            self.tts_websocket = await websockets.connect(
                tts_url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10
            )
            
            self.tts_connected = True
            logger.info(f"[{self.session_id}] TTS | WebSocket connected (persistent)")
            
            # Start receiver task (runs for entire session)
            self.tts_receiver_task = asyncio.create_task(self.tts_receiver())
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Failed to connect WebSocket: {e}")
            self.tts_connected = False
            return False
    
    async def tts_sender_for_response(self):
        """Send tokens for a single response with minimal batching for low latency."""
        buffer = []
        chars_in_buffer = 0
        
        try:
            while self.is_active and self.tts_websocket:
                try:
                    # Wait for token with short timeout for responsiveness
                    tok = await asyncio.wait_for(self.tts_token_queue.get(), timeout=0.05)
                    
                    if tok == "[[FLUSH]]":
                        # Send remaining buffer immediately
                        if buffer:
                            text = "".join(buffer)
                            await self.tts_websocket.send(json.dumps({
                                "type": "Speak",
                                "text": text
                            }))
                            logger.info(f"[{self.session_id}] TTS | Sent final batch: '{text[:50]}...'")
                            buffer.clear()
                            chars_in_buffer = 0
                        
                        # Send Flush command
                        await self.tts_websocket.send(json.dumps({"type": "Flush"}))
                        logger.info(f"[{self.session_id}] TTS | Sent Flush command")
                        return  # Exit sender for this response
                    else:
                        buffer.append(tok)
                        chars_in_buffer += len(tok)
                        
                        # Send batch when threshold reached - smaller for faster first audio
                        if chars_in_buffer >= SEND_EVERY_CHARS:
                            text = "".join(buffer)
                            await self.tts_websocket.send(json.dumps({
                                "type": "Speak",
                                "text": text
                            }))
                            logger.info(f"[{self.session_id}] TTS | Sent batch ({chars_in_buffer} chars): '{text[:30]}...'")
                            buffer.clear()
                            chars_in_buffer = 0
                            
                except asyncio.TimeoutError:
                    # Send any accumulated buffer on timeout to reduce latency
                    if buffer and chars_in_buffer >= 5:  # Minimum viable chunk
                        text = "".join(buffer)
                        await self.tts_websocket.send(json.dumps({
                            "type": "Speak",
                            "text": text
                        }))
                        logger.info(f"[{self.session_id}] TTS | Sent timeout batch: '{text[:30]}...'")
                        buffer.clear()
                        chars_in_buffer = 0
                    continue
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Sender error: {e}")
    
    async def tts_receiver(self):
        """Receive audio chunks from TTS WebSocket - RUNS FOR ENTIRE SESSION."""
        last_audio_ts = time.perf_counter()
        queue_empty_wait = 0.20  # Reduced from 250ms for faster detection
        
        try:
            while self.is_active and self.tts_websocket:
                try:
                    msg = await asyncio.wait_for(self.tts_websocket.recv(), timeout=0.05)
                    
                    # Handle control frames (JSON)
                    if isinstance(msg, str):
                        try:
                            evt = json.loads(msg)
                            if evt.get("type") == "Flushed":
                                logger.info(f"[{self.session_id}] TTS | Received Flushed event")
                            elif evt.get("type") == "Warning":
                                logger.warning(f"[{self.session_id}] TTS | Warning: {evt}")
                            elif evt.get("type") == "Error":
                                logger.error(f"[{self.session_id}] TTS | Error: {evt}")
                        except json.JSONDecodeError:
                            pass
                        continue
                    
                    # Handle audio frames (bytes)
                    if isinstance(msg, bytes):
                        # Log first audio (TTFB)
                        if not self.first_audio_logged and self.tts_start_ts:
                            ttfb = int((time.perf_counter() - self.tts_start_ts) * 1000)
                            logger.info(f"[{self.session_id}] TTS | ⚡ First audio (TTFB): {ttfb}ms")
                            self.first_audio_logged = True
                            
                            # Set speaking state and notify frontend
                            self.is_speaking = True
                            await self.send_message("playback_started")
                        
                        last_audio_ts = time.perf_counter()
                        
                        # Send audio chunk to frontend immediately
                        audio_base64 = base64.b64encode(msg).decode('utf-8')
                        await self.send_message("audio_chunk", {
                            "audio": audio_base64,
                            "encoding": TTS_ENCODING,
                            "sample_rate": TTS_SAMPLE_RATE
                        })
                        
                except asyncio.TimeoutError:
                    # Check if playback is finished (no audio for 200ms after first audio)
                    if self.first_audio_logged and self.is_speaking:
                        silence_duration = time.perf_counter() - last_audio_ts
                        if silence_duration > queue_empty_wait:
                            await self.finish_playback()
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.session_id}] TTS | WebSocket closed")
                    self.tts_connected = False
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Receiver error: {e}")
            self.tts_connected = False
    
    async def finish_playback(self):
        """Handle playback completion."""
        self.is_speaking = False
        self.first_audio_logged = False  # Reset for next response
        await self.send_message("playback_finished")
        
        # Log RTT
        if self.rtt_start_ts:
            rtt = int((time.perf_counter() - self.rtt_start_ts) * 1000)
            logger.info(f"[{self.session_id}] PIPELINE | ⏱ End-to-end RTT: {rtt}ms")
            self.rtt_start_ts = None
        
        logger.info(f"[{self.session_id}] TTS | 🎧 Playback finished")
    
    async def close_tts_websocket(self):
        """Close the TTS WebSocket connection."""
        if self.tts_receiver_task:
            self.tts_receiver_task.cancel()
            try:
                await self.tts_receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except Exception:
                pass
            self.tts_websocket = None
            self.tts_connected = False
    
    async def process_and_respond(self):
        """Process final transcript through LLM and TTS streaming pipeline."""
        try:
            transcript = self.final_transcript.strip()
            if not transcript:
                self.is_processing = False
                return
            
            logger.info(f"[{self.session_id}] PIPELINE | Processing transcript: '{transcript}'")
            
            # Send final transcript to frontend
            await self.send_message("final_transcript", {"text": transcript})
            
            # Ensure TTS WebSocket is connected (persistent connection)
            if not self.tts_connected:
                if not await self.start_tts_websocket():
                    await self.send_message("error", {"message": "Failed to start TTS"})
                    self.is_processing = False
                    return
            
            # Reset latency tracking for this response
            self.first_llm_token_logged = False
            self.first_audio_logged = False
            self.tts_start_ts = time.perf_counter()
            
            # Clear any stale tokens from previous responses
            while not self.tts_token_queue.empty():
                try:
                    self.tts_token_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Start sender task for this response
            sender_task = asyncio.create_task(self.tts_sender_for_response())
            
            # Stream LLM response and pipe to TTS (fully async)
            await self.stream_llm_to_tts_async(transcript)
            
            # Wait for sender to complete
            await sender_task
            
        except Exception as e:
            logger.error(f"[{self.session_id}] PIPELINE | Error in process_and_respond: {e}")
            await self.send_message("error", {"message": str(e)})
        finally:
            self.is_processing = False
            self.final_transcript = ""
    
    async def stream_llm_to_tts_async(self, user_input: str):
        """Stream LLM response tokens directly to TTS queue - FULLY ASYNC."""
        try:
            full_response = ""
            self.llm_start_ts = time.perf_counter()
            
            logger.info(f"[{self.session_id}] LLM | Starting async stream...")
            
            # Use DeepSeek via AsyncOpenAI with async streaming
            response_stream = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                stream=True,
                temperature=0.4  # Lower temperature for faster, more consistent responses
            )
            
            # Async iteration over streaming response
            async for chunk in response_stream:
                if not self.is_active:
                    break
                
                # Check for barge-in and stop processing if user interrupted
                if not self.is_processing:
                    logger.info(f"[{self.session_id}] LLM | Stopping due to barge-in")
                    break
                
                # Extract text from OpenAI-style streaming response
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    
                    # Log first token (TTFT)
                    if not self.first_llm_token_logged and self.llm_start_ts:
                        ttft = int((time.perf_counter() - self.llm_start_ts) * 1000)
                        logger.info(f"[{self.session_id}] LLM | ⚡ First token (TTFT): {ttft}ms")
                        self.first_llm_token_logged = True
                    
                    # Send token to frontend for live display
                    await self.send_message("llm_token", {"text": text})
                    
                    # Push token to TTS queue immediately
                    await self.tts_token_queue.put(text)
            
            # Signal TTS to flush remaining buffer
            await self.tts_token_queue.put("[[FLUSH]]")
            
            logger.info(f"[{self.session_id}] LLM | Response complete: '{full_response[:100]}...'")
            
            # Send response complete signal
            await self.send_message("llm_complete", {"text": full_response})
            
        except Exception as e:
            logger.error(f"[{self.session_id}] LLM | Error in async stream: {e}")
            await self.send_message("error", {"message": f"LLM error: {str(e)}"})
            # Still need to flush TTS
            await self.tts_token_queue.put("[[FLUSH]]")
    
    async def stop_stt_stream(self):
        """Stop the Deepgram STT stream."""
        if self.dg_connection:
            try:
                await self.dg_connection.finish()
                logger.info(f"[{self.session_id}] STT | Stream stopped")
            except Exception as e:
                logger.error(f"[{self.session_id}] STT | Error stopping: {e}")
    
    async def cleanup(self):
        """Clean up session resources."""
        self.is_active = False
        await self.stop_stt_stream()
        await self.close_tts_websocket()
        logger.info(f"[{self.session_id}] SESSION | Cleaned up")


# Store active sessions
active_sessions: dict[str, StreamingSession] = {}


@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """Handle streaming audio WebSocket connection."""
    await websocket.accept()
    logger.info(f"[{session_id}] SESSION | Client connected")
    
    session = StreamingSession(session_id, websocket)
    active_sessions[session_id] = session
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "start_recording":
                logger.info(f"[{session_id}] SESSION | Starting recording...")
                session.final_transcript = ""
                session.is_processing = False
                
                # Start STT stream
                success = await session.start_stt_stream()
                if success:
                    # Pre-connect TTS WebSocket for faster first response
                    await session.start_tts_websocket()
                    await session.send_message("recording_started", {"session_id": session_id})
                else:
                    await session.send_message("error", {"message": "Failed to start speech recognition"})
            
            elif msg_type == "audio_chunk":
                if "audio_data" in data:
                    try:
                        audio_bytes = base64.b64decode(data["audio_data"])
                        await session.send_audio_to_stt(audio_bytes)
                    except Exception as e:
                        logger.error(f"[{session_id}] SESSION | Error processing audio chunk: {e}")
            
            elif msg_type == "stop_recording":
                logger.info(f"[{session_id}] SESSION | Stopping recording...")
                await session.stop_stt_stream()
                
                # If there's accumulated transcript that wasn't processed, process it now
                if session.final_transcript and not session.is_processing:
                    session.rtt_start_ts = time.perf_counter()
                    session.is_processing = True
                    await session.process_and_respond()
                
                await session.send_message("recording_stopped")
            
            elif msg_type == "force_process":
                # Force process any pending transcript
                if session.final_transcript and not session.is_processing:
                    session.rtt_start_ts = time.perf_counter()
                    session.is_processing = True
                    await session.process_and_respond()
    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] SESSION | WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] SESSION | WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await session.cleanup()
        if session_id in active_sessions:
            del active_sessions[session_id]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)