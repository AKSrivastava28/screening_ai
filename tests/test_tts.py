"""Tests for app.tts module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from app.tts import synthesize_followup_speech


def test_synthesize_followup_speech_empty() -> None:
    assert asyncio.run(synthesize_followup_speech("", Path("dummy.wav"), timeout_seconds=1.0)) is False


def test_synthesize_followup_speech_valid(tmp_path: Path) -> None:
    wav_path = tmp_path / "test_followup.wav"
    ok = asyncio.run(synthesize_followup_speech("Could you share a key project you built?", wav_path, timeout_seconds=5.0))
    if ok:
        assert wav_path.exists()
        assert wav_path.stat().st_size > 44
