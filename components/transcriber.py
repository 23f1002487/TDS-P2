"""Audio Transcription Component
Provides async transcription for audio quiz tasks with graceful fallbacks.
"""
from loguru import logger
from typing import Optional, Union
import asyncio
import io
import base64
from .fallback_strategies import TranscriptionFallbackStrategy, TranscriptionEngine

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "opus", "webm"}

class AudioTranscriber:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini-transcribe"):
        self.model = model
        self.client = None
        self.availability = TranscriptionFallbackStrategy.check_availability(api_key)
        self.selected_engine = TranscriptionFallbackStrategy.select_engine(
            api_key=api_key,
            availability=self.availability
        )
        self.available = (self.selected_engine == TranscriptionEngine.OPENAI_WHISPER)
        
        if self.available and AsyncOpenAI and api_key:
            try:
                self.client = AsyncOpenAI(api_key=api_key)
                logger.info(f"AudioTranscriber initialized with model {model}")
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
                self.available = False

    @staticmethod
    def is_audio_url(url: str) -> bool:
        lower = url.lower().split('?')[0]
        return any(lower.endswith(f".{ext}") for ext in AUDIO_EXTENSIONS)

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav", registry=None) -> str:
        """Transcribe given audio bytes. Falls back to placeholder if model unavailable.
        
        Args:
            audio_bytes: Audio file bytes
            filename: Filename for the audio
            registry: Optional CapabilityRegistry to record engine used
        """
        if not self.available:
            logger.warning("Transcriber unavailable - returning fallback message.")
            if registry:
                registry.record("transcription_engine", "unavailable")
            return "TRANSCRIPTION_UNAVAILABLE"
        try:
            model = self.model
            if "transcribe" not in model:
                model = "whisper-1"
            file_obj = io.BytesIO(audio_bytes)
            file_obj.name = filename
            response = await self.client.audio.transcriptions.create(
                model=model,
                file=file_obj
            )
            text = getattr(response, "text", "") or response.get("text", "")
            logger.success("Audio transcription completed")
            if registry:
                registry.record("transcription_engine", f"openai-{model}")
            return text.strip() or ""
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if registry:
                registry.record("transcription_engine", "failed")
            return "TRANSCRIPTION_FAILED"

    async def transcribe_from_url(self, url: str, http_client, registry=None) -> str:
        """Download audio from URL and transcribe.
        
        Args:
            url: Audio file URL
            http_client: HTTP client to download audio
            registry: Optional CapabilityRegistry
        """
        try:
            resp = await http_client.get(url)
            resp.raise_for_status()
            return await self.transcribe_bytes(resp.content, filename=url.split('/')[-1], registry=registry)
        except Exception as e:
            logger.error(f"Failed to download audio for transcription: {e}")
            if registry:
                registry.record("transcription_engine", "download_failed")
            return "TRANSCRIPTION_DOWNLOAD_FAILED"
