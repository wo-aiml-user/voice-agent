"""
Voice Assistant API - Streaming
Main module with FastAPI app and WebSocket endpoint.
"""

import uvicorn
import asyncio
import base64
import json
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from client import logger, TTS_ENCODING, TTS_SAMPLE_RATE
from stt import STTManager
from tts import TTSManager
from llm import get_llm_response_with_queue


app = FastAPI(title="Voice Assistant API - Streaming", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamingSession:
    """Manages a streaming voice session with STT, LLM, and TTS pipelines."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.is_active = True
        self.is_processing = False
        
        # Initialize managers
        self.stt_manager = STTManager(session_id)
        self.tts_manager = TTSManager(session_id)
        
        # Latency tracking
        self.rtt_start_ts: Optional[float] = None
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup callbacks for STT and TTS managers."""
        # STT callbacks
        self.stt_manager.on_interim_transcript = self._on_interim_transcript
        self.stt_manager.on_speech_started = self._on_speech_started
        self.stt_manager.on_utterance_end = self._on_utterance_end
        self.stt_manager.on_error = self._on_stt_error
        
        # TTS callbacks
        self.tts_manager.on_audio_chunk = self._on_audio_chunk
        self.tts_manager.on_playback_started = self._on_playback_started
        self.tts_manager.on_playback_finished = self._on_playback_finished
        self.tts_manager.on_error = self._on_tts_error
    
    async def send_message(self, msg_type: str, data: dict = None):
        """Send a JSON message to the frontend."""
        if not self.is_active:
            return
        try:
            message = {"type": msg_type, **(data or {})}
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[{self.session_id}] Error sending message: {e}")

    
    async def _on_interim_transcript(self, text: str, is_final: bool):
        """Handle interim transcript from STT."""
        await self.send_message("interim_transcript", {
            "text": text,
            "is_final": is_final
        })
    
    async def _on_speech_started(self):
        """Handle speech started event."""
        await self.send_message("speech_started")
        
        # If user starts speaking while assistant is speaking, stop playback
        if self.tts_manager.is_speaking:
            logger.info(f"[{self.session_id}] User barged in - stopping playback")
            await self.handle_barge_in()
    
    async def _on_utterance_end(self, transcript: str):
        """Handle utterance end - trigger processing."""
        if transcript and not self.is_processing:
            self.rtt_start_ts = time.perf_counter()
            self.is_processing = True
            asyncio.create_task(self.process_and_respond())
    
    async def _on_stt_error(self, error: str):
        """Handle STT error."""
        await self.send_message("error", {"message": f"STT error: {error}"})

    
    async def _on_audio_chunk(self, audio_base64: str):
        """Handle audio chunk from TTS."""
        await self.send_message("audio_chunk", {
            "audio": audio_base64,
            "encoding": TTS_ENCODING,
            "sample_rate": TTS_SAMPLE_RATE
        })
    
    async def _on_playback_started(self):
        """Handle playback started event."""
        await self.send_message("playback_started")
    
    async def _on_playback_finished(self):
        """Handle playback finished event."""
        await self.send_message("playback_finished")
        
        # Log RTT
        if self.rtt_start_ts:
            rtt = int((time.perf_counter() - self.rtt_start_ts) * 1000)
            logger.info(f"[{self.session_id}] PIPELINE | ⏱ End-to-end RTT: {rtt}ms")
            self.rtt_start_ts = None
    
    async def _on_tts_error(self, error: str):
        """Handle TTS error."""
        await self.send_message("error", {"message": f"TTS error: {error}"})
    
    async def start_session(self) -> bool:
        """Start STT and TTS streams."""
        # Start STT stream
        success = await self.stt_manager.start_stream()
        if success:
            await self.send_message("stt_ready")
            # Pre-connect TTS WebSocket for faster first response
            await self.tts_manager.connect()
            return True
        return False
    
    async def handle_barge_in(self):
        """Handle user interruption during assistant playback."""
        self.is_processing = False
        await self.tts_manager.stop_playback()
        await self.send_message("stop_playback")
        logger.info(f"[{self.session_id}] BARGE-IN | Playback interrupted by user")
    
    async def process_and_respond(self):
        """Process final transcript through LLM and TTS streaming pipeline."""
        try:
            transcript = self.stt_manager.get_transcript().strip()
            if not transcript:
                self.is_processing = False
                return
            
            logger.info(f"[{self.session_id}] PIPELINE | Processing transcript: '{transcript}'")
            
            # Send final transcript to frontend
            await self.send_message("final_transcript", {"text": transcript})
            
            # Ensure TTS is connected
            if not self.tts_manager.is_connected:
                if not await self.tts_manager.connect():
                    await self.send_message("error", {"message": "Failed to start TTS"})
                    self.is_processing = False
                    return
            
            # Start TTS response processing
            await self.tts_manager.start_response()
            
            # Stream LLM response and pipe to TTS
            await get_llm_response_with_queue(
                session_id=self.session_id,
                user_input=transcript,
                token_queue=self.tts_manager.token_queue,
                send_message_callback=self.send_message,
                is_active_check=lambda: self.is_active,
                is_processing_check=lambda: self.is_processing,
            )
            
            # Wait for TTS sender to complete
            if self.tts_manager.sender_task:
                await self.tts_manager.sender_task
            
        except Exception as e:
            logger.error(f"[{self.session_id}] PIPELINE | Error in process_and_respond: {e}")
            await self.send_message("error", {"message": str(e)})
        finally:
            self.is_processing = False
            self.stt_manager.reset_transcript()
    
    async def cleanup(self):
        """Clean up session resources."""
        self.is_active = False
        await self.stt_manager.stop_stream()
        await self.tts_manager.close()
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
                session.stt_manager.reset_transcript()
                session.is_processing = False
                
                # Start STT and TTS streams
                success = await session.start_session()
                if success:
                    await session.send_message("recording_started", {"session_id": session_id})
                else:
                    await session.send_message("error", {"message": "Failed to start speech recognition"})
            
            elif msg_type == "audio_chunk":
                if "audio_data" in data:
                    try:
                        audio_bytes = base64.b64decode(data["audio_data"])
                        await session.stt_manager.send_audio(audio_bytes)
                    except Exception as e:
                        logger.error(f"[{session_id}] SESSION | Error processing audio chunk: {e}")
            
            elif msg_type == "stop_recording":
                logger.info(f"[{session_id}] SESSION | Stopping recording...")
                await session.stt_manager.stop_stream()
                
                # If there's accumulated transcript that wasn't processed, process it now
                if session.stt_manager.get_transcript() and not session.is_processing:
                    session.rtt_start_ts = time.perf_counter()
                    session.is_processing = True
                    await session.process_and_respond()
                
                await session.send_message("recording_stopped")
            
            elif msg_type == "force_process":
                # Force process any pending transcript
                if session.stt_manager.get_transcript() and not session.is_processing:
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