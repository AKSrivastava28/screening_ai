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


def generate_with_edge_tts(text: str, target_path: Path, voice: str = "en-IN-NeerjaNeural") -> bool:
    """Generate audio using Edge-TTS and resample to 8kHz mono WAV."""
    try:
        import asyncio
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            asyncio.run(edge_tts.Communicate(text, voice).save(tmp_path))
            resample_audio_to_8khz(tmp_path, target_path)
            return True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"    Edge-TTS generation error: {e}")
        return False


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
        else:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")

    for item in questions:
        q_id = item["id"]
        text = item["text"]
        target_path = output_dir / f"{q_id}.wav"

        if target_path.exists() and not force:
            print(f"  [{q_id}] Already exists: {target_path} (use --force to overwrite)")
            continue

        print(f"  [{q_id}] Generating audio for: \"{text}\"")

        if use_mock:
            generate_mock_speech_wav(target_path, duration_seconds=2.5, sample_rate=8000)
            print(f"  [{q_id}] -> Saved synthetic audio to {target_path}")
        else:
            success = False
            # 1. Try Groq TTS if client is available
            if client is not None:
                try:
                    response = client.audio.speech.create(
                        model=settings.GROQ_TTS_MODEL,
                        voice=settings.GROQ_TTS_VOICE,
                        input=text,
                        response_format="wav",
                    )

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                        if hasattr(response, "content"):
                            tmp.write(response.content)
                        elif hasattr(response, "read"):
                            tmp.write(response.read())
                        else:
                            tmp.write(bytes(response))

                    try:
                        resample_audio_to_8khz(tmp_path, target_path)
                        print(f"  [{q_id}] -> Saved 8kHz WAV via Groq TTS to {target_path}")
                        success = True
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                except Exception as e:
                    print(f"  [{q_id}] Groq TTS unavailable ({e}). Trying edge-tts fallback...")

            # 2. Fallback to edge-tts if Groq failed or not configured
            if not success:
                print(f"  [{q_id}] Generating with edge-tts (voice: en-IN-NeerjaNeural)...")
                success = generate_with_edge_tts(text, target_path, voice="en-IN-NeerjaNeural")
                if success:
                    print(f"  [{q_id}] -> Saved 8kHz WAV via edge-tts to {target_path}")

            # 3. Last resort fallback to synthetic tone
            if not success:
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

    # Also generate conversational bridges and conclusion
    extra_clips = [
        {"id": "ack", "text": "Got it, thank you."},
        {
            "id": "conclusion",
            "text": "Thank you for sharing your responses. That concludes our preliminary screening call. We will review your profile. Have a great day!",
        },
    ]
    for item in extra_clips:
        c_id = item["id"]
        c_text = item["text"]
        target_path = args.output_dir / f"{c_id}.wav"
        if target_path.exists() and not args.force:
            print(f"  [{c_id}] Already exists: {target_path}")
            continue
        print(f"  [{c_id}] Generating audio for: \"{c_text}\"")
        if args.mock:
            generate_mock_speech_wav(target_path, duration_seconds=2.0, sample_rate=8000)
        else:
            success = generate_with_edge_tts(c_text, target_path, voice="en-IN-NeerjaNeural")
            if not success:
                generate_mock_speech_wav(target_path, duration_seconds=2.0, sample_rate=8000)

    print("\nAudio pre-generation complete!")


if __name__ == "__main__":
    main()
