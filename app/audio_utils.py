"""Audio utility functions for PCM/WAV manipulation, chunking, and resampling.

Exotel AgentStream requires:
- Format: Raw PCM (linear16, little-endian, mono, 8000 Hz)
- Outgoing chunk size: 3,200 - 100,000 bytes, must be a multiple of 320 bytes.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
from typing import List, Tuple, Union
import wave


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = 8000,
    channels: int = 1,
) -> bytes:
    """Convert raw 16-bit linear PCM bytes to in-memory WAV file bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_out:
        wav_out.setnchannels(channels)
        wav_out.setsampwidth(2)  # 16-bit = 2 bytes per sample
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm_bytes)
    return buffer.getvalue()


def wav_to_pcm16_bytes(
    wav_source: Union[bytes, str, Path],
) -> Tuple[bytes, int, int]:
    """Extract raw PCM bytes, sample rate, and channels from a WAV file or bytes.

    Returns:
        Tuple of (pcm_bytes, sample_rate, channels)
    """
    if isinstance(wav_source, (str, Path)):
        with wave.open(str(wav_source), "rb") as wav_in:
            channels = wav_in.getnchannels()
            sample_rate = wav_in.getframerate()
            pcm_bytes = wav_in.readframes(wav_in.getnframes())
            return pcm_bytes, sample_rate, channels

    with wave.open(io.BytesIO(wav_source), "rb") as wav_in:
        channels = wav_in.getnchannels()
        sample_rate = wav_in.getframerate()
        pcm_bytes = wav_in.readframes(wav_in.getnframes())
        return pcm_bytes, sample_rate, channels


def chunk_pcm_for_exotel(
    pcm_bytes: bytes,
    chunk_size: int = 3200,
) -> List[bytes]:
    """Slice raw PCM audio into Exotel-compliant chunks.

    Exotel constraints:
    - Chunk size: between 3,200 and 100,000 bytes
    - Must be a multiple of 320 bytes (20ms at 8000Hz 16-bit mono)
    - Default 3200 bytes = 200ms of audio
    """
    if chunk_size < 3200 or chunk_size % 320 != 0:
        raise ValueError(
            f"chunk_size must be >= 3200 and a multiple of 320, got {chunk_size}"
        )

    chunks: List[bytes] = []
    total_len = len(pcm_bytes)

    if total_len == 0:
        return chunks

    for i in range(0, total_len, chunk_size):
        chunk = pcm_bytes[i : i + chunk_size]
        # Check if chunk is smaller than 3200 or not a multiple of 320
        remainder = len(chunk) % 320
        if remainder != 0:
            pad_needed = 320 - remainder
            chunk += b"\x00" * pad_needed

        if len(chunk) < 3200:
            # Pad up to 3200 bytes with silence (zeros)
            chunk += b"\x00" * (3200 - len(chunk))

        chunks.append(chunk)

    return chunks


def resample_audio_to_8khz(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Resample an audio file to 8kHz, 16-bit, mono WAV format using ffmpeg or torchaudio fallback."""
    input_str = str(input_path)
    output_str = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_str)), exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",  # overwrite output
            "-i",
            input_str,
            "-ar",
            "8000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            output_str,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            return

    # Fallback to torchaudio if ffmpeg failed or not found
    try:
        import torch
        import torchaudio

        waveform, orig_freq = torchaudio.load(input_str)
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Resample to 8000 Hz
        if orig_freq != 8000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=8000
            )
            waveform = resampler(waveform)

        torchaudio.save(
            output_str,
            waveform,
            8000,
            encoding="PCM_S",
            bits_per_sample=16,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to resample audio using both ffmpeg and torchaudio: {e}"
        ) from e


def generate_mock_speech_wav(
    output_file: Union[str, Path],
    duration_seconds: float = 2.0,
    sample_rate: int = 8000,
    freq: float = 440.0,
) -> None:
    """Generate a clean synthetic WAV file for offline testing without calling Groq TTS."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    num_samples = int(duration_seconds * sample_rate)
    pcm_data = bytearray()

    for i in range(num_samples):
        # Sine wave with gentle envelope to avoid clicks
        t = i / sample_rate
        # Amplitude envelope (fade in 50ms, fade out 50ms)
        fade_in = min(1.0, t / 0.05)
        fade_out = min(1.0, (duration_seconds - t) / 0.05)
        env = fade_in * fade_out
        sample = int(10000 * env * math.sin(2 * math.pi * freq * t))
        pcm_data.extend(struct.pack("<h", sample))

    wav_bytes = pcm16_to_wav_bytes(bytes(pcm_data), sample_rate=sample_rate, channels=1)
    with open(output_file, "wb") as f:
        f.write(wav_bytes)
