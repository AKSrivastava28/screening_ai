"""Unit tests for Silero VAD turn detection and state machine."""

from __future__ import annotations

import torch
import pytest

from app.vad import TurnDetector, get_silero_model


def generate_pcm_silence(duration_sec: float, sample_rate: int = 8000) -> bytes:
    """Generate pure PCM silence."""
    num_samples = int(duration_sec * sample_rate)
    return b"\x00\x00" * num_samples


def test_real_silero_model_inference() -> None:
    """Verify that real Silero VAD model loads and evaluates 8kHz audio tensors."""
    model = get_silero_model()
    # 256 samples @ 8kHz
    dummy_input = torch.zeros(256, dtype=torch.float32)
    with torch.no_grad():
        prob = model(dummy_input, 8000).item()
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_turn_detector_init() -> None:
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=2.0,
        max_answer_seconds=10.0,
    )
    assert detector.sample_rate == 8000
    assert detector.max_silence_seconds == 2.0
    assert detector.max_answer_seconds == 10.0
    assert not detector.has_started_speaking
    assert not detector.is_turn_complete


def test_turn_detector_initial_silence_timeout() -> None:
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=2.0,
        max_answer_seconds=20.0,
        initial_silence_timeout=1.0,  # 1 second for fast test
    )

    # Feed 0.5s of silence
    silence_05s = generate_pcm_silence(0.5, 8000)
    complete = detector.feed_audio(silence_05s)
    assert not complete
    assert not detector.has_started_speaking

    # Feed another 0.6s of silence (total 1.1s > initial_silence_timeout)
    silence_06s = generate_pcm_silence(0.6, 8000)
    complete = detector.feed_audio(silence_06s)
    assert complete
    assert detector.turn_complete_reason == "initial_silence_timeout"


class MockVADModel:
    """Mock VAD model allowing deterministic testing of speech and silence transitions."""

    def __init__(self, prob: float = 0.0) -> None:
        self.prob = prob

    def __call__(self, tensor: torch.Tensor, sr: int) -> torch.Tensor:
        return torch.tensor(self.prob)

    def reset_states(self) -> None:
        pass


def test_turn_detector_speech_then_silence() -> None:
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=1.0,
        max_answer_seconds=20.0,
        speech_threshold=0.5,
        silence_threshold=0.35,
    )
    mock_model = MockVADModel(prob=0.85)  # Simulate speech
    detector.model = mock_model

    # Feed 1.0 second with speech
    audio_1s = b"\x01\x00" * 8000
    detector.feed_audio(audio_1s)
    assert detector.has_started_speaking
    assert detector.is_speaking_now
    assert not detector.is_turn_complete

    # Switch mock to silence
    mock_model.prob = 0.05
    # Feed 0.5s silence - not yet complete (1.0s threshold)
    audio_05s = b"\x00\x00" * 4000
    complete = detector.feed_audio(audio_05s)
    assert not complete
    assert not detector.is_speaking_now

    # Feed another 0.6s silence (total silence = 1.1s > 1.0s)
    audio_06s = b"\x00\x00" * 4800
    complete = detector.feed_audio(audio_06s)
    assert complete
    assert detector.turn_complete_reason == "silence_timeout"


def test_turn_detector_max_answer_timeout() -> None:
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=5.0,
        max_answer_seconds=1.0,  # 1.0s hard cap
    )
    detector.model = MockVADModel(prob=0.8)

    # Feed 1.5s of speech (exceeds max_answer_seconds)
    audio_15s = b"\x01\x00" * 12000
    complete = detector.feed_audio(audio_15s)
    assert complete
    assert detector.turn_complete_reason == "max_answer_timeout"


def test_turn_detector_reset() -> None:
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=1.0,
        max_answer_seconds=5.0,
        initial_silence_timeout=1.0,
    )
    detector.feed_audio(generate_pcm_silence(1.5, 8000))
    assert detector.is_turn_complete

    detector.reset()
    assert not detector.is_turn_complete
    assert not detector.has_started_speaking
    assert detector.turn_complete_reason == ""
    assert len(detector.get_audio_bytes()) == 0


def test_turn_detector_min_answer_seconds() -> None:
    """Verify that turn does not terminate on silence until min_answer_seconds has elapsed."""
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=1.0,
        max_answer_seconds=20.0,
        min_answer_seconds=3.0,
        speech_threshold=0.5,
        silence_threshold=0.35,
    )
    mock_model = MockVADModel(prob=0.85)
    detector.model = mock_model

    # Feed 1.0s speech
    detector.feed_audio(b"\x01\x00" * 8000)
    assert detector.has_started_speaking

    # Switch to silence
    mock_model.prob = 0.05
    # Feed 1.2s silence (> max_silence_seconds 1.0s, but total audio 2.2s < min_answer_seconds 3.0s)
    complete = detector.feed_audio(b"\x00\x00" * 9600)
    assert not complete, "Turn should not complete before min_answer_seconds"

    # Feed another 1.0s silence (total audio 3.2s > 3.0s, silence continues)
    complete = detector.feed_audio(b"\x00\x00" * 8000)
    assert complete, "Turn should complete once both min_answer_seconds and max_silence_seconds are satisfied"
    assert detector.turn_complete_reason == "silence_timeout"


def test_turn_detector_rms_gating_ignores_low_energy_noise() -> None:
    """Verify that faint noise (< min_speech_rms) does not trigger speech detection even if model probability is high."""
    detector = TurnDetector(
        sample_rate=8000,
        max_silence_seconds=1.0,
        max_answer_seconds=20.0,
        speech_threshold=0.60,
        min_speech_rms=50.0,
        speech_debounce_frames=3,
    )
    mock_model = MockVADModel(prob=0.85)
    detector.model = mock_model

    # Faint noise (sample amplitude 5 -> RMS ~5.0 < 50.0)
    faint_audio = b"\x05\x00" * 4000  # 0.5s
    complete = detector.feed_audio(faint_audio)
    assert not complete
    assert not detector.has_started_speaking, "Faint audio below min_speech_rms must not trigger speech"

    # Loud speech (sample amplitude 1000 -> RMS ~1000.0 > 50.0)
    loud_audio = b"\xe8\x03" * 4000  # 0.5s
    complete = detector.feed_audio(loud_audio)
    assert detector.has_started_speaking, "Loud audio above min_speech_rms must trigger speech"


