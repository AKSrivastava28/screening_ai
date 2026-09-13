"""Unit tests for candidate evaluation report generation and cost estimation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.config import settings
from app.evaluator import (
    calculate_call_cost,
    format_markdown_report,
    generate_evaluation_report,
)


def test_calculate_call_cost() -> None:
    # 3-minute call, 1 minute of candidate audio, 500 prompt tokens, 200 completion tokens
    cost = calculate_call_cost(
        call_duration_seconds=180.0,
        audio_transcribed_seconds=60.0,
        prompt_tokens=500,
        completion_tokens=200,
    )

    expected_telephony = 3.0 * settings.TELEPHONY_RATE_PER_MINUTE
    expected_stt = 1.0 * settings.GROQ_WHISPER_RATE_PER_MINUTE
    expected_llm = (500 / 1_000_000 * settings.GROQ_LLAMA_INPUT_PER_1M) + (
        200 / 1_000_000 * settings.GROQ_LLAMA_OUTPUT_PER_1M
    )

    assert cost["telephony_cost_usd"] == round(expected_telephony, 5)
    assert cost["stt_cost_usd"] == round(expected_stt, 5)
    assert cost["llm_cost_usd"] == round(expected_llm, 5)
    assert cost["total_estimated_cost_usd"] > 0


def test_format_markdown_report() -> None:
    sample_data = {
        "call_sid": "CA12345678",
        "candidate_phone": "+919876543210",
        "call_duration_seconds": 125.4,
        "criteria": {
            "communication_clarity": {"score": 4, "justification": "Clear and polite."},
            "experience_relevance": {"score": 5, "justification": "Strong matching background."},
            "availability_notice_fit": {"score": 5, "justification": "Immediate joiner."},
            "compensation_fit": {"score": 4, "justification": "Within allocated budget."},
        },
        "overall_recommendation": {
            "decision": "proceed",
            "justification": "Highly qualified candidate.",
        },
        "key_observations": ["Candidate spoke with confidence."],
        "cost_estimate": {
            "telephony_cost_usd": 0.0314,
            "stt_cost_usd": 0.0012,
            "llm_cost_usd": 0.0005,
            "total_estimated_cost_usd": 0.0331,
        },
        "transcript": [
            {"question_id": "q1", "question": "Record call ok?", "answer": "Yes, absolutely."}
        ],
    }

    md = format_markdown_report(sample_data)
    assert "# Candidate Screening Report — Call `CA12345678`" in md
    assert "PROCEED" in md
    assert "Communication Clarity" in md
    assert "4/5" in md
    assert "Strong matching background." in md
    assert "$0.0331" in md
    assert "Record call ok?" in md


def test_generate_evaluation_report_file_creation(tmp_path: Path) -> None:
    import asyncio

    transcript = [
        {"question_id": "q1", "question": "Record ok?", "answer": "Yes, completely fine."},
        {"question_id": "q2", "question": "Role description?", "answer": "I am a senior backend engineer working with Python and cloud."},
        {"question_id": "q5", "question": "Notice period?", "answer": "I can join immediately."},
    ]

    report = asyncio.run(
        generate_evaluation_report(
            call_sid="TEST_CALL_SID_001",
            candidate_phone="+919876543210",
            transcript_records=transcript,
            call_duration_seconds=120.0,
            audio_transcribed_seconds=45.0,
            output_dir=tmp_path,
        )
    )

    json_file = tmp_path / "TEST_CALL_SID_001.json"
    md_file = tmp_path / "TEST_CALL_SID_001.md"
    pdf_file = tmp_path / "TEST_CALL_SID_001.pdf"

    assert json_file.exists()
    assert md_file.exists()
    assert pdf_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["call_sid"] == "TEST_CALL_SID_001"
    assert "criteria" in data
    assert "communication_clarity" in data["criteria"]
    assert "experience_relevance" in data["criteria"]
    assert "overall_recommendation" in data
    assert data["overall_recommendation"]["decision"] in ("proceed", "hold", "reject")
    assert "cost_estimate" in data
    assert data["cost_estimate"]["total_estimated_cost_usd"] > 0


def test_generate_followup_question_short_or_empty() -> None:
    import asyncio
    from app.evaluator import generate_followup_question

    # Empty answer should safely return None
    assert asyncio.run(generate_followup_question("What is your role?", "")) is None
    # Short answer (< 3 words) should safely return None
    assert asyncio.run(generate_followup_question("What is your role?", "Yes ok")) is None
    # Inaudible / placeholder answer should safely return None
    assert asyncio.run(generate_followup_question("What is your role?", "[No speech detected]")) is None


def test_generate_followup_question_substantive() -> None:
    import asyncio
    from app.evaluator import generate_followup_question

    res = asyncio.run(
        generate_followup_question(
            "What are your qualifications?",
            "I have a bachelor's degree in chemical engineering from BITS Goa.",
            timeout_seconds=3.0,
        )
    )
    # If API key is present and model reachable, should return a question string
    if res is not None:
        assert isinstance(res, str)
        assert res.endswith("?")
        assert len(res) > 5
