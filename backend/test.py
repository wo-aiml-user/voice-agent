"""
Low-Latency Voice Agent using Deepgram Voice Agent API V1.
Uses DeepSeek as custom LLM via OpenAI-compatible endpoint.

Deepgram Voice Agent handles: STT (Nova-3) → LLM → TTS (Aura-2)
All in a single optimized WebSocket connection for minimal latency.
"""

import os
import json
import base64
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Logging setup
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

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPGRAM_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY and DEEPSEEK_API_KEY must be set in .env")

# Deepgram Voice Agent V1 endpoint
VOICE_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# FastAPI app
app = FastAPI(title="Voice Agent API - Deepgram Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_voice_agent_settings() -> dict:
    """
    Configure Deepgram Voice Agent with DeepSeek as custom LLM.
    """
    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": 16000
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none"
            }
        },
        "agent": {
            "language": "en",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3"
                }
            },
            "think": {
                "provider": {
                    "type": "open_ai",
                    "model": "deepseek-chat",
                    "temperature": 0.4
                },
                "endpoint": {
                    "url": "https://api.deepseek.com/v1/chat/completions",
                    "headers": {
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                    }
                },
                "prompt": (
                    "You are a succinct, helpful voice assistant. "
                    "Respond in 2-3 sentences. Be direct and friendly."
                )
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": "aura-2-thalia-en"
                }
            },
            "greeting": "Hello! How can I help you today?"
        }
    }


class VoiceAgentSession:
    """Manages a session with Deepgram Voice Agent API."""
    
    def __init__(self, session_id: str, client_ws: WebSocket):
        self.session_id = session_id
        self.client_ws = client_ws
        self.agent_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_active = True
        self.start_time: Optional[float] = None
        self.audio_chunk_count = 0
        self.playback_started_sent = False
    
    async def connect_to_agent(self) -> bool:
        """Connect to Deepgram Voice Agent API."""
        try:
            self.agent_ws = await websockets.connect(
                VOICE_AGENT_URL,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10
            )
            logger.info(f"[{self.session_id}] Connected to Deepgram Voice Agent")
            
            # Send Settings message to configure the agent
            settings = get_voice_agent_settings()
            await self.agent_ws.send(json.dumps(settings))
            logger.info(f"[{self.session_id}] Sent Settings to Voice Agent")
            
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to connect to Voice Agent: {e}")
            return False
    
    async def forward_audio_to_agent(self, audio_data: bytes):
        """Forward audio from client to Deepgram Voice Agent."""
        if self.agent_ws:
            try:
                await self.agent_ws.send(audio_data)
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending audio to agent: {e}")
    
    async def receive_from_agent(self):
        """Receive messages/audio from Deepgram Voice Agent and forward to client."""
        try:
            while self.is_active and self.agent_ws:
                try:
                    msg = await asyncio.wait_for(self.agent_ws.recv(), timeout=0.1)
                    
                    if isinstance(msg, bytes):
                        # Audio data from TTS - forward to client
                        self.audio_chunk_count += 1
                        
                        # Send playback_started on first audio chunk
                        if not self.playback_started_sent:
                            self.playback_started_sent = True
                            if self.start_time:
                                latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                                logger.info(f"[{self.session_id}] Agent | ⚡ First audio (latency: {latency_ms}ms)")
                            await self.client_ws.send_text(json.dumps({
                                "type": "playback_started"
                            }))
                        
                        # Log every 10th chunk
                        if self.audio_chunk_count % 10 == 1:
                            logger.info(f"[{self.session_id}] Agent | Audio chunk #{self.audio_chunk_count} ({len(msg)} bytes)")
                        
                        audio_base64 = base64.b64encode(msg).decode('utf-8')
                        await self.client_ws.send_text(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio_base64,
                            "encoding": "linear16",
                            "sample_rate": 24000
                        }))
                        
                    elif isinstance(msg, str):
                        # JSON message from agent
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "Welcome":
                            logger.info(f"[{self.session_id}] Agent | Welcome received")
                            await self.client_ws.send_text(json.dumps({
                                "type": "agent_ready"
                            }))
                            
                        elif msg_type == "SettingsApplied":
                            logger.info(f"[{self.session_id}] Agent | Settings applied")
                            await self.client_ws.send_text(json.dumps({
                                "type": "settings_applied"
                            }))
                            
                        elif msg_type == "UserStartedSpeaking":
                            self.start_time = time.perf_counter()
                            logger.info(f"[{self.session_id}] Agent | User started speaking")
                            await self.client_ws.send_text(json.dumps({
                                "type": "speech_started"
                            }))
                            
                        elif msg_type == "AgentThinking":
                            logger.info(f"[{self.session_id}] Agent | Thinking...")
                            await self.client_ws.send_text(json.dumps({
                                "type": "thinking"
                            }))
                            
                        elif msg_type == "AgentStartedSpeaking":
                            if self.start_time:
                                latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                                logger.info(f"[{self.session_id}] Agent | ⚡ Started speaking (latency: {latency_ms}ms)")
                            await self.client_ws.send_text(json.dumps({
                                "type": "playback_started"
                            }))
                            
                        elif msg_type == "AgentAudioDone":
                            logger.info(f"[{self.session_id}] Agent | Audio done (total chunks: {self.audio_chunk_count})")
                            # Reset for next response
                            self.audio_chunk_count = 0
                            self.playback_started_sent = False
                            await self.client_ws.send_text(json.dumps({
                                "type": "playback_finished"
                            }))
                            
                        elif msg_type == "ConversationText":
                            # Transcript or response text
                            role = data.get("role")
                            content = data.get("content", "")
                            
                            if role == "user":
                                logger.info(f"[{self.session_id}] Agent | User: {content}")
                                await self.client_ws.send_text(json.dumps({
                                    "type": "transcript",
                                    "text": content
                                }))
                            elif role == "assistant":
                                logger.info(f"[{self.session_id}] Agent | Assistant: {content}")
                                await self.client_ws.send_text(json.dumps({
                                    "type": "response",
                                    "text": content
                                }))
                                
                        elif msg_type == "Error":
                            error_msg = data.get("message", "Unknown error")
                            logger.error(f"[{self.session_id}] Agent | Error: {error_msg}")
                            await self.client_ws.send_text(json.dumps({
                                "type": "error",
                                "message": error_msg
                            }))
                            
                        else:
                            logger.debug(f"[{self.session_id}] Agent | {msg_type}: {data}")
                            
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.session_id}] Agent connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Error receiving from agent: {e}")
    
    async def close(self):
        """Close the Voice Agent connection."""
        self.is_active = False
        if self.agent_ws:
            try:
                await self.agent_ws.close()
            except Exception:
                pass
        logger.info(f"[{self.session_id}] Session closed")


# Store active sessions
active_sessions: dict[str, VoiceAgentSession] = {}


@app.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connection for voice agent."""
    await websocket.accept()
    logger.info(f"[{session_id}] Client connected")
    
    session = VoiceAgentSession(session_id, websocket)
    active_sessions[session_id] = session
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
                
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "start_session":
                    logger.info(f"[{session_id}] Starting voice agent session...")
                    success = await session.connect_to_agent()
                    
                    if success:
                        # Start receiving from agent in background
                        asyncio.create_task(session.receive_from_agent())
                        await session.client_ws.send_text(json.dumps({
                            "type": "session_started",
                            "session_id": session_id
                        }))
                    else:
                        await session.client_ws.send_text(json.dumps({
                            "type": "error",
                            "message": "Failed to connect to voice agent"
                        }))
                
                elif msg_type == "audio_chunk":
                    # Decode and forward audio to Deepgram Voice Agent
                    if "audio_data" in data:
                        audio_bytes = base64.b64decode(data["audio_data"])
                        await session.forward_audio_to_agent(audio_bytes)
                
                elif msg_type == "end_session":
                    logger.info(f"[{session_id}] Ending session...")
                    break
            
            elif "bytes" in message:
                # Raw binary audio - forward directly
                await session.forward_audio_to_agent(message["bytes"])
    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await session.close()
        if session_id in active_sessions:
            del active_sessions[session_id]


if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)
