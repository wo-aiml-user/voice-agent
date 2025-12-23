"""
Speech-to-Text module for Voice Agent.
Handles Deepgram live transcription with VAD and interim results.
"""

import asyncio
from typing import Callable, Optional
from deepgram import LiveTranscriptionEvents, LiveOptions

from client import deepgram, logger


class STTManager:
    """Manages Deepgram STT streaming connection."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dg_connection = None
        self.is_active = True
        self.final_transcript = ""
        
        # Callbacks
        self.on_interim_transcript: Optional[Callable] = None
        self.on_final_transcript: Optional[Callable] = None
        self.on_speech_started: Optional[Callable] = None
        self.on_utterance_end: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def start_stream(self) -> bool:
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
                        
                        if self.on_interim_transcript:
                            await self.on_interim_transcript(self.final_transcript, True)
                        
                        logger.info(f"[{self.session_id}] STT | Final segment: '{transcript}'")
                        
                        # If speech_final (endpoint detected), trigger processing
                        if speech_final and self.final_transcript:
                            if self.on_utterance_end:
                                await self.on_utterance_end(self.final_transcript)
                    else:
                        # Interim result - show partial transcript
                        partial = self.final_transcript + " " + transcript if self.final_transcript else transcript
                        if self.on_interim_transcript:
                            await self.on_interim_transcript(partial.strip(), False)
                            
                except Exception as e:
                    logger.error(f"[{self.session_id}] STT | Error in on_message: {e}")
            
            async def on_speech_started_event(self_conn, speech_started, **kwargs):
                logger.info(f"[{self.session_id}] STT | Speech started")
                if self.on_speech_started:
                    await self.on_speech_started()
            
            async def on_utterance_end_event(self_conn, utterance_end, **kwargs):
                logger.info(f"[{self.session_id}] STT | Utterance end detected")
                if self.final_transcript and self.on_utterance_end:
                    await self.on_utterance_end(self.final_transcript)
            
            async def on_error_event(self_conn, error, **kwargs):
                logger.error(f"[{self.session_id}] STT | Error: {error}")
                if self.on_error:
                    await self.on_error(str(error))
            
            async def on_close_event(self_conn, close, **kwargs):
                logger.info(f"[{self.session_id}] STT | Connection closed")
            
            # Register event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started_event)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end_event)
            self.dg_connection.on(LiveTranscriptionEvents.Error, on_error_event)
            self.dg_connection.on(LiveTranscriptionEvents.Close, on_close_event)
            
            # Configure live transcription options with VAD - optimized timing
            options = LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                punctuate=True,
                interim_results=True,
                utterance_end_ms=2000,
                vad_events=True,
                endpointing=200,  # Fast endpoint detection
            )
            
            # Start the connection
            if await self.dg_connection.start(options):
                logger.info(f"[{self.session_id}] STT | Stream started successfully")
                return True
            else:
                logger.error(f"[{self.session_id}] STT | Failed to start stream")
                return False
                
        except Exception as e:
            import traceback
            logger.error(f"[{self.session_id}] STT | Error starting: {type(e).__name__}: {e}")
            logger.error(f"[{self.session_id}] STT | Traceback: {traceback.format_exc()}")
            return False
    
    async def send_audio(self, audio_data: bytes):
        """Forward audio data to Deepgram STT stream."""
        if self.dg_connection and self.is_active:
            try:
                await self.dg_connection.send(audio_data)
            except Exception as e:
                logger.error(f"[{self.session_id}] STT | Error sending audio: {e}")
    
    async def stop_stream(self):
        """Stop the Deepgram STT stream."""
        self.is_active = False
        if self.dg_connection:
            try:
                await self.dg_connection.finish()
                logger.info(f"[{self.session_id}] STT | Stream stopped")
            except Exception as e:
                logger.error(f"[{self.session_id}] STT | Error stopping: {e}")
    
    def reset_transcript(self):
        """Reset the accumulated transcript."""
        self.final_transcript = ""
    
    def get_transcript(self) -> str:
        """Get the current accumulated transcript."""
        return self.final_transcript
