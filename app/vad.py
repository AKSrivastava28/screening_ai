"""Silero VAD turn detection module for Exotel 8kHz audio streams."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional
import numpy as np
import torch

from app.audio_utils import normalize_audio_pcm
from app.config import settings

logger = logging.getLogger(__name__)

_SILERO_MODEL: Optional[torch.nn.Module] = None


def get_silero_model() -> torch.nn.Module:
    """Load and cache the Silero VAD model from local file or torch.hub fallback."""
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        local_model_path = Path(__file__).resolve().parent.parent / "models" / "silero_vad.jit"
        if local_model_path.exists():
            logger.info("Loading Silero VAD model from local bundle: %s", local_model_path)
            try:
                model = torch.jit.load(str(local_model_path), map_location="cpu")
                model.eval()
                _SILERO_MODEL = model
                logger.info("Silero VAD model loaded successfully from local file.")
                return _SILERO_MODEL
            except Exception as e:
                logger.warning("Failed to load local Silero model (%s), trying torch.hub...", e)

        logger.info("Loading Silero VAD model via torch.hub...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        model.eval()
        _SILERO_MODEL = model
        logger.info("Silero VAD model loaded successfully.")
    return _SILERO_MODEL


class TurnDetector:
    """State machine for detecting end-of-turn for candidate spoken answers using Silero VAD."""

    def __init__(
        self,
        sample_rate: int = 8000,
        max_silence_seconds: Optional[float] = None,
        max_answer_seconds: Optional[float] = None,
        min_answer_seconds: float = 0.0,
        initial_silence_timeout: float = 7.0,
        speech_threshold: float = 0.20,
        silence_threshold: float = 0.10,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_silence_seconds = (
            max_silence_seconds if max_silence_seconds is not None else settings.MAX_SILENCE_SECONDS
        )
        self.max_answer_seconds = (
            max_answer_seconds if max_answer_seconds is not None else settings.MAX_ANSWER_SECONDS
        )
        self.min_answer_seconds = min_answer_seconds
        self.initial_silence_timeout = initial_silence_timeout
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold

        # 256 samples @ 8000Hz = 0.032s (32ms). 256 * 2 bytes = 512 bytes for 16-bit PCM.
        self.window_size_samples = 256
        self.window_size_bytes = self.window_size_samples * 2
        self.chunk_duration_seconds = self.window_size_samples / self.sample_rate

        self.model: Optional[torch.nn.Module] = None
        self.reset()

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            self.model = get_silero_model()
        return self.model

    def reset(self, min_answer_seconds: Optional[float] = None) -> None:
        """Reset state for a new question/turn."""
        self.has_started_speaking = False
        self.is_speaking_now = False
        self.accumulated_silence_seconds = 0.0
        self.total_answer_seconds = 0.0
        self.is_turn_complete = False
        self.turn_complete_reason = ""
        self.buffered_pcm = bytearray()
        self.leftover_pcm = bytearray()
        self.start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        if min_answer_seconds is not None:
            self.min_answer_seconds = min_answer_seconds

        if self.model is not None and hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def check_timeouts(self) -> bool:
        """Check wall-clock timeouts for silence and answer limits.

        Crucial for telephony environments where silence suppression (Comfort Noise
        Generation) pauses incoming audio frames when the candidate stops speaking.
        """
        if self.is_turn_complete:
            return True
        if self.start_time is None:
            return False

        now = time.monotonic()
        wall_elapsed = now - self.start_time

        # 1. Candidate spoke and silence exceeded max_silence_seconds (with min answer constraint)
        if self.has_started_speaking and self.last_speech_time is not None:
            silence_elapsed = now - self.last_speech_time
            if wall_elapsed >= self.min_answer_seconds and silence_elapsed >= self.max_silence_seconds:
                self.is_turn_complete = True
                self.turn_complete_reason = "silence_timeout"
                logger.info(
                    "VAD: Silence timeout reached (%.1fs silence after speech, total=%.1fs, min=%.1fs)",
                    silence_elapsed,
                    wall_elapsed,
                    self.min_answer_seconds,
                )
                return True

        # 2. Hard answer cap exceeded
        if wall_elapsed >= self.max_answer_seconds:
            self.is_turn_complete = True
            self.turn_complete_reason = "max_answer_timeout"
            logger.info("VAD: Max answer duration reached (%.1fs wallclock)", wall_elapsed)
            return True

        # 3. Initial silence timeout (candidate never started speaking)
        if not self.has_started_speaking and wall_elapsed >= self.initial_silence_timeout:
            self.is_turn_complete = True
            self.turn_complete_reason = "initial_silence_timeout"
            logger.info(
                "VAD: Initial silence timeout reached (%.1fs wallclock with no speech)",
                wall_elapsed,
            )
            return True

        return False

    def feed_audio(self, pcm_bytes: bytes) -> bool:
        """Process incoming raw 16-bit PCM audio bytes from Exotel.

        Returns:
            True if candidate's turn is complete, False otherwise.
        """
        if self.is_turn_complete:
            return True

        if not pcm_bytes:
            return self.check_timeouts()

        if self.start_time is None:
            self.start_time = time.monotonic()

        # Apply software Automatic Gain Control (AGC) to boost faint phone audio
        boosted_chunk = normalize_audio_pcm(pcm_bytes, target_rms=1400.0, max_gain=50.0)

        # Buffer boosted audio for later Whisper transcription
        self.buffered_pcm.extend(boosted_chunk)

        # Prepend any leftovers from previous frame
        combined = self.leftover_pcm + boosted_chunk
        total_len = len(combined)

        model = self._ensure_model()

        idx = 0
        while idx + self.window_size_bytes <= total_len:
            window_bytes = combined[idx : idx + self.window_size_bytes]
            idx += self.window_size_bytes

            # Convert 16-bit signed PCM to float32 tensor normalized in [-1.0, 1.0]
            audio_int16 = np.frombuffer(window_bytes, dtype=np.int16)
            audio_float32 = torch.from_numpy(audio_int16).float() / 32768.0

            with torch.no_grad():
                speech_prob = model(audio_float32, self.sample_rate).item()

            rms = float(np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2)))
            is_speech = (speech_prob >= self.speech_threshold) or (rms >= 300.0)

            self.total_answer_seconds += self.chunk_duration_seconds

            # State transitions
            if is_speech:
                if not self.has_started_speaking:
                    logger.info("VAD: Candidate started speaking (prob=%.2f, rms=%.1f)", speech_prob, rms)
                    self.has_started_speaking = True
                self.is_speaking_now = True
                self.accumulated_silence_seconds = 0.0
                self.last_speech_time = time.monotonic()
            elif speech_prob < self.silence_threshold and rms < 200.0:
                self.is_speaking_now = False
                if self.has_started_speaking:
                    self.accumulated_silence_seconds += self.chunk_duration_seconds

            # Check termination conditions:
            # 1. Candidate finished speaking and silence exceeded MAX_SILENCE_SECONDS
            if (
                self.has_started_speaking
                and self.total_answer_seconds >= self.min_answer_seconds
                and self.accumulated_silence_seconds >= self.max_silence_seconds
            ):
                self.is_turn_complete = True
                self.turn_complete_reason = "silence_timeout"
                logger.info(
                    "VAD: End of turn detected (silence >= %.1fs, total=%.1fs, min=%.1fs)",
                    self.max_silence_seconds,
                    self.total_answer_seconds,
                    self.min_answer_seconds,
                )
                break

            # 2. Hard cap per answer hit
            if self.total_answer_seconds >= self.max_answer_seconds:
                self.is_turn_complete = True
                self.turn_complete_reason = "max_answer_timeout"
                logger.info(
                    "VAD: Answer hard cap reached (%.1fs)",
                    self.max_answer_seconds,
                )
                break

            # 3. Initial silence timeout (candidate never spoke)
            if not self.has_started_speaking and self.total_answer_seconds >= self.initial_silence_timeout:
                self.is_turn_complete = True
                self.turn_complete_reason = "initial_silence_timeout"
                logger.info(
                    "VAD: Initial silence timeout reached (%.1fs with no speech)",
                    self.initial_silence_timeout,
                )
                break

        # Save remaining bytes that couldn't form a full 256-sample window
        self.leftover_pcm = combined[idx:]

        # Also verify wall-clock timeout
        if not self.is_turn_complete:
            self.check_timeouts()

        return self.is_turn_complete

    def get_audio_bytes(self) -> bytes:
        """Return raw PCM bytes buffered for this answer."""
        return bytes(self.buffered_pcm)

    @property
    def speech_detected(self) -> bool:
        """Whether candidate spoke at all during this turn."""
        return self.has_started_speaking
