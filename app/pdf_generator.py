"""PDF Report Generator for Candidate Screening using ReportLab."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


def generate_candidate_pdf(
    report_data: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> bytes:
    """Generate a clean, professional PDF candidate screening report.

    Args:
        report_data: Dictionary containing evaluation criteria, transcript, cost, recommendation.
        output_path: Optional file path to save the PDF.

    Returns:
        bytes: Raw PDF bytes.
    """
    buffer = io.BytesIO()
    target = str(output_path) if output_path is not None else buffer
    doc = SimpleDocTemplate(
        target,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "HeaderSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        fontName="Helvetica",
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica",
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
    )
    q_style = ParagraphStyle(
        "QuestionText",
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E40AF"),
        fontName="Helvetica-Bold",
    )
    ans_style = ParagraphStyle(
        "AnswerText",
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=12,
    )

    elements = []

    # 1. Title Banner
    call_sid = report_data.get("call_sid", "N/A")
    elements.append(Paragraph("AI Candidate Screening Report", header_title_style))
    elements.append(Paragraph(f"Call Reference: {call_sid} | Evaluated via Groq AI", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))

    # 2. Key Metadata & Recommendation Card
    rec = report_data.get("overall_recommendation", {})
    decision = rec.get("decision", "hold").upper()
    justification = rec.get("justification", "")
    phone = report_data.get("candidate_phone", "N/A")
    duration = report_data.get("call_duration_seconds", 0)

    dec_color_map = {
        "PROCEED": colors.HexColor("#166534"),
        "HOLD": colors.HexColor("#854D0E"),
        "REJECT": colors.HexColor("#991B1B"),
    }
    dec_bg_map = {
        "PROCEED": colors.HexColor("#DCFCE7"),
        "HOLD": colors.HexColor("#FEF9C3"),
        "REJECT": colors.HexColor("#FEE2E2"),
    }
    badge_color = dec_color_map.get(decision, colors.HexColor("#334155"))
    cand_name = report_data.get("candidate_name", "Candidate")
    job_role = report_data.get("job_role", "Software Developer")

    summary_data = [
        [
            Paragraph("<b>Candidate Name:</b>", body_style),
            Paragraph(f"<b>{cand_name}</b>", bold_body),
            Paragraph("<b>Applied Role:</b>", body_style),
            Paragraph(f"<b>{job_role}</b>", bold_body),
        ],
        [
            Paragraph("<b>Candidate Phone:</b>", body_style),
            Paragraph(str(phone), body_style),
            Paragraph("<b>Recommendation:</b>", body_style),
            Paragraph(f"<font color='{badge_color.hexval()}'><b>{decision}</b></font>", bold_body),
        ],
        [
            Paragraph("<b>Call Duration:</b>", body_style),
            Paragraph(f"{duration:.1f} seconds", body_style),
            Paragraph("<b>Overall Fit:</b>", body_style),
            Paragraph(justification[:100] + ("..." if len(justification) > 100 else ""), body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[100, 150, 110, 180])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    # 3. Evaluation Criteria Table
    elements.append(Paragraph("Evaluation Criteria Scorecard", section_heading))
    criteria_map = report_data.get("criteria", {})
    criteria_labels = {
        "communication_clarity": "Communication Clarity",
        "qualifications_fit": "Qualifications Fit",
        "experience_relevance": "Experience Relevance",
        "availability_notice_fit": "Availability / Notice",
        "compensation_fit": "Compensation Fit",
    }

    crit_rows = [
        [
            Paragraph("<b>Criteria</b>", bold_body),
            Paragraph("<b>Score</b>", bold_body),
            Paragraph("<b>Justification</b>", bold_body),
        ]
    ]

    for key, val in criteria_map.items():
        if not isinstance(val, dict):
            continue
        label = criteria_labels.get(key, key.replace("_", " ").title())
        score = val.get("score", "-")
        just = val.get("justification", "")
        crit_rows.append([
            Paragraph(f"<b>{label}</b>", body_style),
            Paragraph(f"<b>{score}/5</b>", bold_body),
            Paragraph(str(just), body_style),
        ])

    crit_table = Table(crit_rows, colWidths=[150, 60, 330])
    crit_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(crit_table)
    elements.append(Spacer(1, 14))

    # 4. Key Observations
    obs_list = report_data.get("key_observations", [])
    if obs_list:
        elements.append(Paragraph("Key Observations", section_heading))
        for obs in obs_list:
            elements.append(Paragraph(f"• {obs}", body_style))
            elements.append(Spacer(1, 3))
        elements.append(Spacer(1, 10))

    # 5. Full Screening Transcript
    transcripts = report_data.get("transcript", [])
    if transcripts:
        elements.append(Paragraph("Screening Interview Transcript", section_heading))
        for item in transcripts:
            q_id = item.get("question_id", "")
            q_text = item.get("question", "")
            ans = item.get("answer", "[No answer recorded]")
            elements.append(Paragraph(f"<b>Q [{q_id}]:</b> {q_text}", q_style))
            elements.append(Spacer(1, 2))
            elements.append(Paragraph(f"<b>Candidate:</b> \"{ans}\"", ans_style))
            elements.append(Spacer(1, 8))
        elements.append(Spacer(1, 6))

    # 6. Cost Breakdown Table (Converted to Indian Rupees INR)
    cost = report_data.get("cost_estimate", {})
    usd_to_inr = 85.0
    tel_usd = float(cost.get("telephony_cost_usd", 0) or 0)
    stt_usd = float(cost.get("stt_cost_usd", 0) or 0)
    llm_usd = float(cost.get("llm_cost_usd", 0) or 0)
    tot_usd = float(cost.get("total_estimated_cost_usd", 0) or 0)

    tel_inr = tel_usd * usd_to_inr
    stt_inr = stt_usd * usd_to_inr
    llm_inr = llm_usd * usd_to_inr
    tot_inr = tot_usd * usd_to_inr

    elements.append(Paragraph("Telephony & AI Cost Estimate (INR @ Rs. 85/USD)", section_heading))
    cost_data = [
        [
            Paragraph("<b>Telephony Cost (Exotel)</b>", body_style),
            Paragraph(f"Rs. {tel_inr:.4f}", body_style),
            Paragraph("<b>Whisper STT Cost</b>", body_style),
            Paragraph(f"Rs. {stt_inr:.4f}", body_style),
        ],
        [
            Paragraph("<b>LLM Evaluation Cost (Groq)</b>", body_style),
            Paragraph(f"Rs. {llm_inr:.4f}", body_style),
            Paragraph("<b>Total Estimated Cost (INR)</b>", bold_body),
            Paragraph(f"<b>Rs. {tot_inr:.4f}</b>", bold_body),
        ],
    ]
    cost_table = Table(cost_data, colWidths=[150, 120, 150, 120])
    cost_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(cost_table)
    elements.append(Spacer(1, 14))

    # 7. Footer Notice
    footer_text = "Confidential Screening Report — Generated by AI Voice Screening Agent with Groq & Exotel Telephony"
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    elements.append(Paragraph(footer_text, subtitle_style))

    doc.build(elements)

    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    else:
        buffer.seek(0)
        return buffer.getvalue()
