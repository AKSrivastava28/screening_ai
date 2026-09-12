"""Groq Whisper STT module for candidate spoken answers."""

from __future__ import annotations

import io
import logging
from typing import Optional
from groq import AsyncGroq, Groq

from app.audio_utils import pcm16_to_wav_bytes
from app.config import settings

logger = logging.getLogger(__name__)

_ASYNC_GROQ_CLIENT: Optional[AsyncGroq] = None


def get_async_groq_client() -> Optional[AsyncGroq]:
    """Get or create singleton AsyncGroq client."""
    global _ASYNC_GROQ_CLIENT
    if _ASYNC_GROQ_CLIENT is None and settings.GROQ_API_KEY.strip():
        _ASYNC_GROQ_CLIENT = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    return _ASYNC_GROQ_CLIENT


async def transcribe_answer(
    pcm_bytes: bytes,
    sample_rate: int = 8000,
    mock_text: Optional[str] = None,
) -> str:
    """Transcribe buffered candidate 8kHz PCM audio using Groq Whisper.

    Args:
        pcm_bytes: Raw 16-bit linear PCM audio.
        sample_rate: Sample rate (default 8000 Hz).
        mock_text: Optional mock transcript for testing.

    Returns:
        Transcribed string.
    """
    if mock_text is not None:
        return mock_text

    # Check minimum audio duration (at 8000Hz 16-bit: 1 sec = 16,000 bytes)
    duration_seconds = len(pcm_bytes) / (sample_rate * 2)
    if duration_seconds < 0.4:
        logger.info("Audio too short (%.2fs), skipping transcription.", duration_seconds)
        return "[No speech detected]"

    client = get_async_groq_client()
    if client is None:
        logger.warning("GROQ_API_KEY not set. Returning mock transcription.")
        return f"[Simulated answer: Candidate spoke for {duration_seconds:.1f} seconds]"

    wav_bytes = pcm16_to_wav_bytes(pcm_bytes, sample_rate=sample_rate, channels=1)

    try:
        # Pass in-memory file tuple (filename, bytes, content_type)
        file_tuple = ("answer.wav", io.BytesIO(wav_bytes), "audio/wav")

        transcription = await client.audio.transcriptions.create(
            file=file_tuple,
            model=settings.GROQ_STT_MODEL,
            response_format="json",
            language="en",
        )

        text = transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()
        logger.info("Transcription completed: %s", text[:80] + ("..." if len(text) > 80 else ""))
        return text if text else "[Unintelligible / silence]"
    except Exception as e:
        logger.error("Error transcribing audio with Groq Whisper: %s", e)
        return f"[Transcription error: {str(e)}]"
