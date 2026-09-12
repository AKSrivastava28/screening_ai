"""Evaluation report generation using Groq LLaMA-3.3-70b and call cost estimation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from groq import AsyncGroq, Groq

from app.config import settings

logger = logging.getLogger(__name__)


def calculate_call_cost(
    call_duration_seconds: float,
    audio_transcribed_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, float]:
    """Calculate estimated cost breakdown for the call session.

    Rates are loaded from settings (.env) - never hardcoded.
    """
    call_minutes = max(0.0, call_duration_seconds / 60.0)
    stt_minutes = max(0.0, audio_transcribed_seconds / 60.0)

    telephony_cost = call_minutes * settings.TELEPHONY_RATE_PER_MINUTE
    stt_cost = stt_minutes * settings.GROQ_WHISPER_RATE_PER_MINUTE
    llm_cost = (prompt_tokens / 1_000_000 * settings.GROQ_LLAMA_INPUT_PER_1M) + (
        completion_tokens / 1_000_000 * settings.GROQ_LLAMA_OUTPUT_PER_1M
    )
    total_cost = telephony_cost + stt_cost + llm_cost

    return {
        "telephony_cost_usd": round(telephony_cost, 5),
        "stt_cost_usd": round(stt_cost, 5),
        "llm_cost_usd": round(llm_cost, 5),
        "total_estimated_cost_usd": round(total_cost, 5),
    }


def format_markdown_report(data: Dict[str, Any]) -> str:
    """Format evaluation data into clean markdown."""
    rec = data.get("overall_recommendation", {})
    decision = rec.get("decision", "hold").upper()
    crit = data.get("criteria", {})
    cost = data.get("cost_estimate", {})

    status_badge = {
        "PROCEED": "🟢 **PROCEED**",
        "HOLD": "🟡 **HOLD**",
        "REJECT": "🔴 **REJECT**",
    }.get(decision, f"⚪ **{decision}**")

    lines = [
        f"# Candidate Screening Report — Call `{data.get('call_sid', 'N/A')}`",
        "",
        f"- **Candidate Phone**: `{data.get('candidate_phone', 'N/A')}`",
        f"- **Call Duration**: {data.get('call_duration_seconds', 0):.1f} seconds",
        f"- **Status / Recommendation**: {status_badge}",
        f"- **Justification**: {rec.get('justification', '')}",
        "",
        "## Evaluation Criteria (1–5 Scale)",
        "",
        "| Criteria | Score | Justification |",
        "| :--- | :---: | :--- |",
    ]

    criteria_labels = {
        "communication_clarity": "Communication Clarity",
        "experience_relevance": "Experience Relevance",
        "availability_notice_fit": "Availability / Notice Fit",
        "compensation_fit": "Compensation Fit",
    }

    for key, label in criteria_labels.items():
        c_item = crit.get(key, {})
        score = c_item.get("score", "N/A")
        just = c_item.get("justification", "")
        lines.append(f"| **{label}** | {score}/5 | {just} |")

    lines.extend([
        "",
        "## Key Observations",
        "",
    ])
    for obs in data.get("key_observations", []):
        lines.append(f"- {obs}")

    lines.extend([
        "",
        "## Cost Estimate Breakdown",
        "",
        f"- **Telephony Cost**: ${cost.get('telephony_cost_usd', 0):.4f}",
        f"- **Whisper STT Cost**: ${cost.get('stt_cost_usd', 0):.4f}",
        f"- **LLaMA 3.3 LLM Cost**: ${cost.get('llm_cost_usd', 0):.4f}",
        f"- **Total Estimated Cost**: **${cost.get('total_estimated_cost_usd', 0):.4f}**",
        "",
        "## Full Screening Transcript",
        "",
    ])

    for item in data.get("transcript", []):
        lines.append(f"**Q ({item.get('question_id', '')}):** {item.get('question', '')}")
        lines.append(f"> **Candidate:** {item.get('answer', '')}")
        lines.append("")

    return "\n".join(lines)


async def generate_evaluation_report(
    call_sid: str,
    candidate_phone: str,
    transcript_records: List[Dict[str, Any]],
    call_duration_seconds: float,
    audio_transcribed_seconds: float,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate structured candidate evaluation report using Groq LLaMA-3.3-70b."""
    reports_path = output_dir or settings.REPORTS_DIR
    reports_path.mkdir(parents=True, exist_ok=True)

    formatted_qa = "\n\n".join(
        f"Question [{rec.get('question_id', f'q{i+1}')}]: {rec.get('question', '')}\nAnswer: {rec.get('answer', '[No response]')}"
        for i, rec in enumerate(transcript_records)
    )

    system_prompt = (
        "You are an expert HR screening analyst. You will be provided with the verbatim transcript of a scripted "
        "first-round phone screening interview. Score the candidate objectively and concisely based solely on "
        "their answers.\n"
        "Evaluate exactly these 5 criteria:\n"
        "1. communication_clarity: 1 to 5 (1 = poor/incoherent, 5 = exceptionally articulate and clear)\n"
        "2. experience_relevance: 1 to 5 (1 = irrelevant background, 5 = directly relevant experience)\n"
        "3. availability_notice_fit: 1 to 5 (1 = incompatible timeline, 5 = immediate or quick joiner)\n"
        "4. compensation_fit: 1 to 5 (1 = unrealistic or huge mismatch, 5 = within standard expectations)\n"
        "5. overall_recommendation: Must be one of 'proceed', 'hold', or 'reject'.\n"
        "Each criterion must have a one-line justification.\n"
        "Also include a list of 2-3 key observations.\n"
        "Return ONLY a valid JSON object matching this schema without markdown fences:\n"
        "{\n"
        '  "criteria": {\n'
        '    "communication_clarity": {"score": 4, "justification": "..."},\n'
        '    "experience_relevance": {"score": 4, "justification": "..."},\n'
        '    "availability_notice_fit": {"score": 4, "justification": "..."},\n'
        '    "compensation_fit": {"score": 4, "justification": "..."}\n'
        "  },\n"
        '  "overall_recommendation": {\n'
        '    "decision": "proceed",\n'
        '    "justification": "..."\n'
        "  },\n"
        '  "key_observations": ["...", "..."]\n'
        "}"
    )

    prompt_tokens = 0
    completion_tokens = 0
    parsed_llm_response: Dict[str, Any] = {}

    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not configured. Generating simulated evaluation report.")
        parsed_llm_response = {
            "criteria": {
                "communication_clarity": {
                    "score": 4,
                    "justification": "Clear and responsive during simulated screening.",
                },
                "experience_relevance": {
                    "score": 4,
                    "justification": "Relevant skills and project background mentioned.",
                },
                "availability_notice_fit": {
                    "score": 4,
                    "justification": "Manageable notice period aligned with hiring timeline.",
                },
                "compensation_fit": {
                    "score": 4,
                    "justification": "Expectations align with standard role budget.",
                },
            },
            "overall_recommendation": {
                "decision": "proceed",
                "justification": "Candidate meets primary screening prerequisites.",
            },
            "key_observations": [
                "Answered all 9 screening questions.",
                "Simulated evaluation generated for offline testing.",
            ],
        }
    else:
        try:
            client = AsyncGroq(api_key=api_key)
            completion = await client.chat.completions.create(
                model=settings.GROQ_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the interview transcript:\n\n{formatted_qa}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            if completion.usage:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens

            content_text = completion.choices[0].message.content or "{}"
            parsed_llm_response = json.loads(content_text)
        except Exception as e:
            logger.error("Error generating LLM evaluation report: %s", e)
            parsed_llm_response = {
                "criteria": {
                    "communication_clarity": {"score": 3, "justification": f"LLM error: {e}"},
                    "experience_relevance": {"score": 3, "justification": f"LLM error: {e}"},
                    "availability_notice_fit": {"score": 3, "justification": f"LLM error: {e}"},
                    "compensation_fit": {"score": 3, "justification": f"LLM error: {e}"},
                },
                "overall_recommendation": {
                    "decision": "hold",
                    "justification": f"Report fallback due to error: {e}",
                },
                "key_observations": ["Report generation encountered error."],
            }

    # Calculate costs
    cost_summary = calculate_call_cost(
        call_duration_seconds=call_duration_seconds,
        audio_transcribed_seconds=audio_transcribed_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    full_report = {
        "call_sid": call_sid,
        "candidate_phone": candidate_phone,
        "call_duration_seconds": call_duration_seconds,
        "audio_transcribed_seconds": audio_transcribed_seconds,
        "criteria": parsed_llm_response.get("criteria", {}),
        "overall_recommendation": parsed_llm_response.get("overall_recommendation", {}),
        "key_observations": parsed_llm_response.get("key_observations", []),
        "cost_estimate": cost_summary,
        "transcript": transcript_records,
    }

    # Save to reports/{call_sid}.json
    json_path = reports_path / f"{call_sid}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Save to reports/{call_sid}.md
    md_path = reports_path / f"{call_sid}.md"
    md_content = format_markdown_report(full_report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Evaluation report saved to %s and %s", json_path, md_path)
    logger.info("Total Call Estimated Cost: $%s", cost_summary["total_estimated_cost_usd"])
    return full_report
