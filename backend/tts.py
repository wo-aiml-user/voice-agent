"""
Text-to-Speech module for Voice Agent.
Handles Deepgram Aura-2 TTS streaming.
"""

import asyncio
import json
import time
import base64
from typing import Callable, Optional
import websockets

from client import (
    DEEPGRAM_API_KEY,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    TTS_ENCODING,
    SEND_EVERY_CHARS,
    logger,
)


class TTSManager:
    """Manages Deepgram TTS WebSocket connection."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.websocket = None
        self.is_connected = False
        self.is_active = True
        
        # TTS token queue for micro-batching
        self.token_queue: asyncio.Queue[str] = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None
        self.receiver_task: Optional[asyncio.Task] = None
        
        # Latency tracking
        self.tts_start_ts: Optional[float] = None
        self.first_audio_logged = False
        
        # State tracking
        self.is_speaking = False
        
        # Callbacks
        self.on_audio_chunk: Optional[Callable] = None
        self.on_playback_started: Optional[Callable] = None
        self.on_playback_finished: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def connect(self) -> bool:
        """Initialize persistent WebSocket connection to Deepgram TTS."""
        if self.is_connected and self.websocket:
            logger.info(f"[{self.session_id}] TTS | Reusing existing WebSocket")
            return True
            
        try:
            tts_url = (
                f"wss://api.deepgram.com/v1/speak?"
                f"model={TTS_MODEL}&encoding={TTS_ENCODING}&sample_rate={TTS_SAMPLE_RATE}"
            )
            
            self.websocket = await websockets.connect(
                tts_url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            logger.info(f"[{self.session_id}] TTS | WebSocket connected (persistent)")
            
            # Start receiver task (runs for entire session)
            self.receiver_task = asyncio.create_task(self._receiver())
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Failed to connect WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def start_response(self):
        """Start processing a new response."""
        # Reset state for new response
        self.first_audio_logged = False
        self.tts_start_ts = time.perf_counter()
        
        # Clear any stale tokens from previous responses
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Start sender task for this response
        self.sender_task = asyncio.create_task(self._sender())
    
    async def send_token(self, token: str):
        """Queue a token for TTS."""
        await self.token_queue.put(token)
    
    async def flush(self):
        """Signal end of response and wait for sender to complete."""
        await self.token_queue.put("[[FLUSH]]")
        if self.sender_task:
            await self.sender_task
    
    async def _sender(self):
        """Send tokens for a single response with minimal batching for low latency."""
        buffer = []
        chars_in_buffer = 0
        
        try:
            while self.is_active and self.websocket:
                try:
                    # Wait for token with short timeout for responsiveness
                    tok = await asyncio.wait_for(self.token_queue.get(), timeout=0.05)
                    
                    if tok == "[[FLUSH]]":
                        # Send remaining buffer immediately
                        if buffer:
                            text = "".join(buffer)
                            await self.websocket.send(json.dumps({
                                "type": "Speak",
                                "text": text
                            }))
                            logger.info(f"[{self.session_id}] TTS | Sent final batch: '{text}'")
                            buffer.clear()
                            chars_in_buffer = 0
                        
                        # Send Flush command
                        await self.websocket.send(json.dumps({"type": "Flush"}))
                        logger.info(f"[{self.session_id}] TTS | Sent Flush command")
                        return  # Exit sender for this response
                    else:
                        buffer.append(tok)
                        chars_in_buffer += len(tok)
                        
                        # Send batch when threshold reached
                        if chars_in_buffer >= SEND_EVERY_CHARS:
                            text = "".join(buffer)
                            await self.websocket.send(json.dumps({
                                "type": "Speak",
                                "text": text
                            }))
                            logger.info(f"[{self.session_id}] TTS | Sent batch ({chars_in_buffer} chars): '{text}'")
                            buffer.clear()
                            chars_in_buffer = 0
                            
                except asyncio.TimeoutError:
                    # Send any accumulated buffer on timeout to reduce latency
                    if buffer and chars_in_buffer >= 5:  # Minimum viable chunk
                        text = "".join(buffer)
                        await self.websocket.send(json.dumps({
                            "type": "Speak",
                            "text": text
                        }))
                        logger.info(f"[{self.session_id}] TTS | Sent timeout batch: '{text}'")
                        buffer.clear()
                        chars_in_buffer = 0
                    continue
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Sender error: {e}")
    
    async def _receiver(self):
        """Receive audio chunks from TTS WebSocket - runs for entire session."""
        last_audio_ts = time.perf_counter()
        queue_empty_wait = 0.20  # 200ms silence detection
        
        try:
            while self.is_active and self.websocket:
                try:
                    msg = await asyncio.wait_for(self.websocket.recv(), timeout=0.05)
                    
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
                                if self.on_error:
                                    await self.on_error(str(evt))
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
                            
                            # Set speaking state and notify
                            self.is_speaking = True
                            if self.on_playback_started:
                                await self.on_playback_started()
                        
                        last_audio_ts = time.perf_counter()
                        
                        # Send audio chunk via callback
                        if self.on_audio_chunk:
                            audio_base64 = base64.b64encode(msg).decode('utf-8')
                            await self.on_audio_chunk(audio_base64)
                        
                except asyncio.TimeoutError:
                    # Check if playback is finished (no audio for 200ms after first audio)
                    if self.first_audio_logged and self.is_speaking:
                        silence_duration = time.perf_counter() - last_audio_ts
                        if silence_duration > queue_empty_wait:
                            await self._finish_playback()
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.session_id}] TTS | WebSocket closed")
                    self.is_connected = False
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS | Receiver error: {e}")
            self.is_connected = False
    
    async def _finish_playback(self):
        """Handle playback completion."""
        self.is_speaking = False
        self.first_audio_logged = False  # Reset for next response
        
        if self.on_playback_finished:
            await self.on_playback_finished()
        
        logger.info(f"[{self.session_id}] TTS | 🎧 Playback finished")
    
    async def stop_playback(self):
        """Stop current playback (for barge-in handling)."""
        self.is_speaking = False
        self.first_audio_logged = False
        
        # Clear the token queue
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"[{self.session_id}] TTS | Playback stopped (barge-in)")
    
    async def close(self):
        """Close the TTS WebSocket connection."""
        self.is_active = False
        
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.sender_task:
            self.sender_task.cancel()
            try:
                await self.sender_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
            self.is_connected = False
        
        logger.info(f"[{self.session_id}] TTS | Connection closed")
