"""Unit tests for audio manipulation and chunking utilities."""

from __future__ import annotations

import io
from pathlib import Path
import struct
import wave
import pytest

from app.audio_utils import (
    chunk_pcm_for_exotel,
    generate_mock_speech_wav,
    pcm16_to_wav_bytes,
    wav_to_pcm16_bytes,
)


def test_pcm16_to_wav_bytes() -> None:
    # 8000 samples = 1 second of 8kHz 16-bit mono = 16000 bytes
    dummy_pcm = b"\x00\x00" * 8000
    wav_bytes = pcm16_to_wav_bytes(dummy_pcm, sample_rate=8000, channels=1)

    assert len(wav_bytes) > len(dummy_pcm)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 8000


def test_wav_to_pcm16_bytes(tmp_path: Path) -> None:
    test_wav = tmp_path / "test.wav"
    generate_mock_speech_wav(test_wav, duration_seconds=1.0, sample_rate=8000)

    pcm_bytes, sr, ch = wav_to_pcm16_bytes(test_wav)
    assert sr == 8000
    assert ch == 1
    # 1 second @ 8000Hz 16-bit = 8000 samples * 2 bytes = 16,000 bytes
    assert len(pcm_bytes) == 16000


def test_chunk_pcm_for_exotel_sizing() -> None:
    # 10,000 bytes of PCM
    pcm = b"\x01\x02" * 5000
    chunks = chunk_pcm_for_exotel(pcm, chunk_size=3200)

    assert len(chunks) > 0
    for chunk in chunks:
        # Exotel requirement 1: chunk size >= 3200
        assert len(chunk) >= 3200
        # Exotel requirement 2: chunk size is multiple of 320
        assert len(chunk) % 320 == 0


def test_chunk_pcm_for_exotel_small_audio() -> None:
    # Small audio (e.g. 500 bytes)
    small_pcm = b"\x00" * 500
    chunks = chunk_pcm_for_exotel(small_pcm, chunk_size=3200)

    assert len(chunks) == 1
    assert len(chunks[0]) == 3200
    assert len(chunks[0]) % 320 == 0


def test_chunk_pcm_for_exotel_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        # < 3200
        chunk_pcm_for_exotel(b"\x00" * 100, chunk_size=1600)

    with pytest.raises(ValueError):
        # not multiple of 320
        chunk_pcm_for_exotel(b"\x00" * 100, chunk_size=3300)
