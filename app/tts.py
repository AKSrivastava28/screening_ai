"""TTS synthesis module for live dynamic voice generation."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional

from app.audio_utils import resample_audio_to_8khz
from app.config import settings

logger = logging.getLogger(__name__)


async def synthesize_followup_speech(
    text: str,
    output_path: Path,
    voice: Optional[str] = None,
    timeout_seconds: float = 3.5,
) -> bool:
    """Synthesize speech using Edge-TTS into an 8kHz 16-bit mono WAV.

    Args:
        text: The sentence to speak.
        output_path: Destination path for the 8kHz WAV file.
        voice: TTS voice name (defaults to settings.FOLLOWUP_VOICE).
        timeout_seconds: Maximum time allowed before giving up.

    Returns:
        bool: True if generation and resampling succeeded, False otherwise.
    """
    clean_text = text.strip()
    if not clean_text:
        return False

    chosen_voice = voice or settings.FOLLOWUP_VOICE or "en-IN-NeerjaNeural"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _do_synth() -> bool:
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            communicate = edge_tts.Communicate(clean_text, chosen_voice)
            await communicate.save(tmp_path)
            # Run ffmpeg resampling in thread pool to prevent blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, resample_audio_to_8khz, tmp_path, output_path)
            return output_path.exists() and output_path.stat().st_size > 44
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    try:
        return await asyncio.wait_for(_do_synth(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "TTS synthesis timed out after %.1fs for text: '%s'",
            timeout_seconds,
            clean_text[:40],
        )
        return False
    except Exception as e:
        logger.error("TTS synthesis error for text '%s': %s", clean_text[:40], e)
        return False
