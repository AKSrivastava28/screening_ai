"""Standalone CLI script to generate candidate evaluation report using Groq LLaMA-3.3-70b.

Usage:
    python generate_report.py --demo
    python generate_report.py --transcript-file path/to/transcript.json --call-sid CALL123
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from app.evaluator import generate_evaluation_report


DEMO_TRANSCRIPT = [
    {
        "question_id": "q1",
        "question": "This call will be recorded for evaluation purposes, is that okay with you?",
        "answer": "Yes, that is completely fine with me.",
        "duration_seconds": 3.2,
    },
    {
        "question_id": "q2",
        "question": "Could you briefly describe your current role and responsibilities?",
        "answer": "I am a Senior Backend Engineer at TechCorp. I design and build distributed microservices in Python and FastAPI, handle database migrations with PostgreSQL, and deploy on Kubernetes.",
        "duration_seconds": 12.5,
    },
    {
        "question_id": "q3",
        "question": "How many years of total experience do you have, and how much of that is relevant to this role?",
        "answer": "I have 6 years of total software engineering experience, and all 6 years are directly relevant to backend development, API design, and cloud architecture.",
        "duration_seconds": 8.1,
    },
    {
        "question_id": "q4",
        "question": "What's prompting you to look for a new opportunity right now?",
        "answer": "I'm looking for a more high-impact role with technical leadership opportunities where I can solve deeper scalability challenges in AI and cloud systems.",
        "duration_seconds": 9.4,
    },
    {
        "question_id": "q5",
        "question": "What's your current notice period, and how soon could you join if selected?",
        "answer": "My notice period is officially 30 days, but I have already discussed with my manager and can be relieved in 15 days, so I can join almost immediately.",
        "duration_seconds": 7.8,
    },
    {
        "question_id": "q6",
        "question": "What's your current CTC, and what are you expecting for this role?",
        "answer": "My current CTC is 24 LPA fixed, and I am expecting around 32 to 35 LPA based on market standards and my experience level.",
        "duration_seconds": 6.5,
    },
    {
        "question_id": "q7",
        "question": "Are you open to the work location and mode for this role, and would you consider relocating?",
        "answer": "Yes, I am completely comfortable with hybrid or in-office work in Bangalore or Gurgaon, and I am ready to relocate without any issue.",
        "duration_seconds": 7.0,
    },
    {
        "question_id": "q8",
        "question": "Could you share an example of a challenging technical project you led recently and how you resolved key obstacles?",
        "answer": "I led the migration of our monolith payment service to an event-driven architecture using Kafka. We had strict zero-downtime requirements and handled 10,000 requests per second with idempotency keys and outbox pattern.",
        "duration_seconds": 15.2,
    },
    {
        "question_id": "q9",
        "question": "That's all my questions. Do you have anything you'd like to ask us? Thanks for your time today.",
        "answer": "Thank you! I would love to learn more about the team structure and technical stack in the next round. Thanks for your time!",
        "duration_seconds": 6.0,
    },
]


async def run_report(
    call_sid: str,
    candidate_phone: str,
    transcript_records: list,
    call_duration_seconds: float,
    audio_transcribed_seconds: float,
) -> None:
    print(f"\nGenerating evaluation report for Call SID: {call_sid}...")
    report = await generate_evaluation_report(
        call_sid=call_sid,
        candidate_phone=candidate_phone,
        transcript_records=transcript_records,
        call_duration_seconds=call_duration_seconds,
        audio_transcribed_seconds=audio_transcribed_seconds,
    )

    rec = report.get("overall_recommendation", {})
    cost = report.get("cost_estimate", {})
    crit = report.get("criteria", {})

    print("\n" + "=" * 60)
    print(f"CANDIDATE SCREENING REPORT: {call_sid}")
    print("=" * 60)
    print(f"Decision: {rec.get('decision', '').upper()}")
    print(f"Justification: {rec.get('justification', '')}")
    print("-" * 60)
    print("CRITERIA SCORES:")
    for k, v in crit.items():
        clean_just = str(v.get("justification", "")).encode("ascii", errors="replace").decode("ascii")
        print(f"  - {k.replace('_', ' ').title():<26}: {v.get('score')}/5 - {clean_just}")
    print("-" * 60)
    print(f"TOTAL ESTIMATED COST: ${cost.get('total_estimated_cost_usd'):.4f}")
    print("=" * 60)
    print(f"Saved JSON report: reports/{call_sid}.json")
    print(f"Saved Markdown report: reports/{call_sid}.md\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate screening evaluation report."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run report generation on demo interview transcript",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        help="Path to JSON file containing list of Q&A transcript items",
    )
    parser.add_argument(
        "--call-sid",
        type=str,
        default="DEMO_CALL_001",
        help="Call SID identifier (default: DEMO_CALL_001)",
    )
    parser.add_argument(
        "--candidate-phone",
        type=str,
        default="+919876543210",
        help="Candidate phone number",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=240.0,
        help="Call duration in seconds (default: 240.0)",
    )

    args = parser.parse_args()

    if args.demo:
        transcript = DEMO_TRANSCRIPT
        total_audio = sum(item.get("duration_seconds", 0.0) for item in transcript)
        asyncio.run(
            run_report(
                call_sid=args.call_sid,
                candidate_phone=args.candidate_phone,
                transcript_records=transcript,
                call_duration_seconds=args.duration,
                audio_transcribed_seconds=total_audio,
            )
        )
    elif args.transcript_file:
        if not args.transcript_file.exists():
            print(f"Error: Transcript file not found at {args.transcript_file}")
            sys.exit(1)
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        total_audio = sum(item.get("duration_seconds", 0.0) for item in transcript)
        asyncio.run(
            run_report(
                call_sid=args.call_sid,
                candidate_phone=args.candidate_phone,
                transcript_records=transcript,
                call_duration_seconds=args.duration,
                audio_transcribed_seconds=total_audio,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
