"""Setup script to pre-generate static 8kHz audio files for fixed screening questions.

Reads questions from questions.json, invokes Groq Orpheus TTS once per question,
resamples output to 8kHz 16-bit mono WAV (Exotel's expected format), and saves each as
audio/{id}.wav.

Usage:
    python generate_question_audio.py
    python generate_question_audio.py --mock   # Generate synthetic audio for offline testing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from app.audio_utils import generate_mock_speech_wav, resample_audio_to_8khz, wav_to_pcm16_bytes
from app.config import settings


def generate_audio_for_questions(
    questions_file: Path,
    output_dir: Path,
    use_mock: bool = False,
    force: bool = False,
) -> None:
    """Generate audio files for all questions in questions.json."""
    if not questions_file.exists():
        print(f"Error: Questions file not found at {questions_file}")
        sys.exit(1)

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(questions)} questions from {questions_file}")
    print(f"Output directory: {output_dir}")

    client = None
    if not use_mock:
        api_key = settings.GROQ_API_KEY.strip()
        if not api_key:
            print("Warning: GROQ_API_KEY is not set.")
            if not use_mock:
                print("Falling back to synthetic audio generation (--mock mode).")
                use_mock = True
        else:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
                print("Falling back to synthetic audio generation.")
                use_mock = True

    for item in questions:
        q_id = item["id"]
        text = item["text"]
        target_path = output_dir / f"{q_id}.wav"

        if target_path.exists() and not force:
            print(f"  [{q_id}] Already exists: {target_path} (use --force to overwrite)")
            continue

        print(f"  [{q_id}] Generating audio for: \"{text}\"")

        if use_mock:
            # Generate synthetic 8kHz WAV for testing
            generate_mock_speech_wav(target_path, duration_seconds=2.5, sample_rate=8000)
            print(f"  [{q_id}] -> Saved synthetic audio to {target_path}")
        else:
            try:
                # Call Groq TTS API
                # Groq Orpheus endpoint
                response = client.audio.speech.create(
                    model=settings.GROQ_TTS_MODEL,
                    voice=settings.GROQ_TTS_VOICE,
                    input=text,
                    response_format="wav",
                )

                # Write to temp file then resample to 8000Hz mono
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    # response is a BinaryAPIResponse
                    if hasattr(response, "content"):
                        tmp.write(response.content)
                    elif hasattr(response, "read"):
                        tmp.write(response.read())
                    else:
                        tmp.write(bytes(response))

                try:
                    resample_audio_to_8khz(tmp_path, target_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                print(f"  [{q_id}] -> Saved 8kHz WAV to {target_path}")
            except Exception as e:
                print(f"  [{q_id}] Error calling Groq TTS: {e}")
                print(f"  [{q_id}] Falling back to synthetic tone for {q_id}")
                generate_mock_speech_wav(target_path, duration_seconds=2.5, sample_rate=8000)

        # Verify output file
        if target_path.exists():
            pcm_bytes, sr, ch = wav_to_pcm16_bytes(target_path)
            duration = len(pcm_bytes) / (sr * 2 * ch)
            print(f"  [{q_id}] Verified: {sr}Hz, {ch}ch, {duration:.2f}s, {len(pcm_bytes)} bytes PCM")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate static 8kHz audio for screening questions."
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=settings.QUESTIONS_FILE,
        help="Path to questions JSON file (default: questions.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.AUDIO_DIR,
        help="Directory to save generated .wav files (default: audio/)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic mock audio without calling Groq API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing audio files",
    )
    args = parser.parse_args()

    generate_audio_for_questions(
        questions_file=args.questions_file,
        output_dir=args.output_dir,
        use_mock=args.mock,
        force=args.force,
    )
    print("\nAudio pre-generation complete!")


if __name__ == "__main__":
    main()
