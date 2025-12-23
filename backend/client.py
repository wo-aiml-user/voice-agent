"""
Client initialization module for Voice Agent.
Contains API clients, configuration, and logging setup.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from deepgram import DeepgramClient, DeepgramClientOptions
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

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

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPGRAM_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("API keys for Deepgram and DeepSeek must be set in .env file")

dg_config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, dg_config)

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# TTS Configuration - Optimized for low latency
TTS_MODEL = "aura-2-thalia-en"
TTS_SAMPLE_RATE = 24000
TTS_ENCODING = "linear16"
SEND_EVERY_CHARS = 15  # Micro-batch size for faster first audio

# LLM Configuration
SYSTEM_PROMPT = (
    "You are a succinct, helpful voice assistant. "
    "Respond in 2-3 sentences. Be direct and friendly."
)

logger.info("Client module initialized successfully")
