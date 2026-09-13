"""Evaluation report generation using Groq LLaMA-3.3-70b and call cost estimation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from groq import AsyncGroq, Groq

from app.config import settings
from app.pdf_generator import generate_candidate_pdf

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

    cand_name = data.get("candidate_name", "Candidate")
    job_role = data.get("job_role", "Software Developer")

    lines = [
        f"# Candidate Screening Report — Call `{data.get('call_sid', 'N/A')}`",
        "",
        f"- **Candidate Name**: **{cand_name}**",
        f"- **Applied Role**: **{job_role}**",
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
        "qualifications_fit": "Qualifications Fit",
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
        "## Cost Estimate Breakdown (INR / USD)",
        "",
        f"- **Telephony Cost**: ₹{float(cost.get('telephony_cost_usd', 0) or 0) * 85.0:.4f} (${cost.get('telephony_cost_usd', 0):.4f})",
        f"- **Whisper STT Cost**: ₹{float(cost.get('stt_cost_usd', 0) or 0) * 85.0:.4f} (${cost.get('stt_cost_usd', 0):.4f})",
        f"- **LLaMA 3.3 LLM Cost**: ₹{float(cost.get('llm_cost_usd', 0) or 0) * 85.0:.4f} (${cost.get('llm_cost_usd', 0):.4f})",
        f"- **Total Estimated Cost**: **₹{float(cost.get('total_estimated_cost_usd', 0) or 0) * 85.0:.4f}** (${cost.get('total_estimated_cost_usd', 0):.4f})",
        "",
        "## Full Screening Transcript",
        "",
    ])

    for item in data.get("transcript", []):
        lines.append(f"**Q ({item.get('question_id', '')}):** {item.get('question', '')}")
        lines.append(f"> **Candidate:** {item.get('answer', '')}")
        lines.append("")

    return "\n".join(lines)


async def generate_followup_question(
    question_text: str,
    answer_text: str,
    timeout_seconds: float = 2.5,
) -> Optional[str]:
    """Generate a brief, natural follow-up question using Groq LLM based on candidate's answer.

    Args:
        question_text: The screening question that was asked.
        answer_text: The candidate's transcribed answer.
        timeout_seconds: Maximum time to wait before timing out.

    Returns:
        Optional[str]: 1-sentence follow-up question (max 15 words), or None if failed/timed out.
    """
    clean_ans = answer_text.strip()
    # Skip if answer is too short, inaudible, or placeholder
    if not clean_ans or len(clean_ans.split()) < 3 or clean_ans.startswith("["):
        return None

    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        return None

    system_prompt = (
        "You are a polite, professional telephone interviewer conducting a first-round job screening call in India. "
        "The candidate just answered a question. Generate exactly ONE concise, natural follow-up question "
        "(maximum 15 words) that specifically probes deeper into what they just mentioned. "
        "Do NOT include any greetings, praise, preambles, or conversational filler like 'Great!' or 'Understood!'. "
        "Output ONLY the single question."
    )

    user_prompt = f"Question Asked: {question_text}\nCandidate Answer: {clean_ans}\n\nAsk 1 concise follow-up question:"

    async def _call_llm() -> Optional[str]:
        client = AsyncGroq(api_key=api_key)
        candidate_models = [
            settings.GROQ_LLM_MODEL,
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
        ]
        for model_name in candidate_models:
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=40,
                    temperature=0.3,
                )
                raw_followup = completion.choices[0].message.content or ""
                clean_followup = raw_followup.strip().strip('"').strip("'").strip()
                if clean_followup and len(clean_followup) > 5:
                    if not clean_followup.endswith("?"):
                        clean_followup += "?"
                    logger.info("Dynamic follow-up generated (%s): '%s'", model_name, clean_followup)
                    return clean_followup
            except Exception as e:
                logger.warning("Follow-up generation with %s failed: %s", model_name, e)
                continue
        return None

    try:
        return await asyncio.wait_for(_call_llm(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("Follow-up LLM generation timed out after %.1fs", timeout_seconds)
        return None
    except Exception as err:
        logger.error("Error generating follow-up question: %s", err)
        return None


async def generate_evaluation_report(
    call_sid: str,
    candidate_phone: str,
    transcript_records: List[Dict[str, Any]],
    call_duration_seconds: float,
    audio_transcribed_seconds: float,
    output_dir: Optional[Path] = None,
    candidate_name: str = "Candidate",
    job_role: str = "Software Developer",
) -> Dict[str, Any]:
    """Generate structured candidate evaluation report using Groq LLaMA-3.3-70b."""
    reports_path = output_dir or settings.REPORTS_DIR
    reports_path.mkdir(parents=True, exist_ok=True)

    formatted_qa = "\n\n".join(
        f"Question [{rec.get('question_id', f'q{i+1}')}]: {rec.get('question', '')}\nAnswer: {rec.get('answer', '[No response]')}"
        for i, rec in enumerate(transcript_records)
    )

    system_prompt = (
        "You are an expert HR screening analyst. You will be provided with the verbatim transcript of a rapid "
        "first-round phone screening interview focusing on qualifications and experience. Score the candidate objectively "
        "and concisely based solely on their answers.\n"
        "Evaluate these criteria:\n"
        "1. communication_clarity: 1 to 5 (1 = poor/incoherent, 5 = exceptionally articulate and clear)\n"
        "2. qualifications_fit: 1 to 5 (1 = no relevant degree or skills, 5 = strong academic/technical background)\n"
        "3. experience_relevance: 1 to 5 (1 = zero relevant experience, 5 = substantial directly relevant experience)\n"
        "4. overall_recommendation: Must be one of 'proceed', 'hold', or 'reject'.\n"
        "Each criterion must have a one-line justification.\n"
        "Also include a list of 2-3 key observations.\n"
        "Return ONLY a valid JSON object matching this schema without markdown fences:\n"
        "{\n"
        '  "criteria": {\n'
        '    "communication_clarity": {"score": 4, "justification": "..."},\n'
        '    "qualifications_fit": {"score": 4, "justification": "..."},\n'
        '    "experience_relevance": {"score": 4, "justification": "..."}\n'
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
                "qualifications_fit": {
                    "score": 4,
                    "justification": "Academic qualifications and technical skills fit role requirements.",
                },
                "experience_relevance": {
                    "score": 4,
                    "justification": "Demonstrated relevant experience in core technical areas.",
                },
            },
            "overall_recommendation": {
                "decision": "proceed",
                "justification": "Candidate meets primary screening prerequisites.",
            },
            "key_observations": [
                "Completed preliminary screening interview.",
                "Simulated evaluation generated for offline testing.",
            ],
        }
    else:
        client = AsyncGroq(api_key=api_key)
        candidate_models = [settings.GROQ_LLM_MODEL]
        for fallback in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "groq/compound"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_name in candidate_models:
            try:
                logger.info("Generating evaluation report with Groq model: %s", model_name)
                completion = await client.chat.completions.create(
                    model=model_name,
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
                logger.info("Successfully generated evaluation report using model: %s", model_name)
                break
            except Exception as e:
                logger.warning("Model %s failed: %s. Trying next candidate...", model_name, e)
                last_error = e

        if not parsed_llm_response:
            logger.error("All LLM candidates failed. Last error: %s", last_error)
            parsed_llm_response = {
                "criteria": {
                    "communication_clarity": {"score": 3, "justification": f"LLM evaluation notice: {last_error}"},
                    "qualifications_fit": {"score": 3, "justification": f"LLM evaluation notice: {last_error}"},
                    "experience_relevance": {"score": 3, "justification": f"LLM evaluation notice: {last_error}"},
                },
                "overall_recommendation": {
                    "decision": "hold",
                    "justification": f"Report fallback due to: {last_error}",
                },
                "key_observations": ["Report generation fallback invoked."],
            }

    # Calculate costs
    cost_summary = calculate_call_cost(
        call_duration_seconds=call_duration_seconds,
        audio_transcribed_seconds=audio_transcribed_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    raw_criteria = parsed_llm_response.get("criteria", {})
    clean_criteria = {}
    clean_observations = parsed_llm_response.get("key_observations", [])
    if isinstance(raw_criteria, dict):
        for k, v in raw_criteria.items():
            if isinstance(v, dict):
                clean_criteria[k] = v
            elif k == "key_observations" and isinstance(v, list) and not clean_observations:
                clean_observations = v

    for req_key in ("communication_clarity", "qualifications_fit", "experience_relevance"):
        if req_key not in clean_criteria:
            clean_criteria[req_key] = {"score": 3, "justification": "Evaluated during screening."}

    rec = parsed_llm_response.get("overall_recommendation")
    if isinstance(rec, str):
        overall_rec = {"decision": rec.lower(), "justification": "Candidate evaluation recommendation."}
    elif isinstance(rec, dict):
        decision = rec.get("decision") or "proceed"
        overall_rec = {
            "decision": decision.lower() if isinstance(decision, str) else "proceed",
            "justification": rec.get("justification", "Evaluated based on screening criteria.")
        }
    else:
        overall_rec = {"decision": "proceed", "justification": "Candidate evaluation recommendation."}

    full_report = {
        "call_sid": call_sid,
        "candidate_name": candidate_name,
        "job_role": job_role,
        "candidate_phone": candidate_phone,
        "call_duration_seconds": call_duration_seconds,
        "audio_transcribed_seconds": audio_transcribed_seconds,
        "criteria": clean_criteria,
        "overall_recommendation": overall_rec,
        "key_observations": clean_observations,
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

    # Save to reports/{call_sid}.pdf
    pdf_path = reports_path / f"{call_sid}.pdf"
    try:
        generate_candidate_pdf(full_report, pdf_path)
        logger.info("Evaluation report PDF saved to %s", pdf_path)
    except Exception as pdf_err:
        logger.error("Failed to generate candidate PDF: %s", pdf_err)

    logger.info("Evaluation report saved to %s, %s, and %s", json_path, md_path, pdf_path)
    logger.info("Total Call Estimated Cost: $%s", cost_summary["total_estimated_cost_usd"])
    return full_report
