"""FastAPI Webhook and WebSocket server for AI Voice Screening Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from app.audio_utils import chunk_pcm_for_exotel, wav_to_pcm16_bytes
from app.config import settings
from app.evaluator import generate_evaluation_report, generate_followup_question
from app.stt import transcribe_answer
from app.telephony import exotel_client
from app.tts import synthesize_followup_speech
from app.vad import TurnDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Voice Screening Agent",
    description="Backend service for Exotel + Groq scripted voice screening interviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NormalizePathMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            raw_path = scope.get("path", "")
            if "//" in raw_path:
                scope["path"] = re.sub(r"/+", "/", raw_path)
        await self.app(scope, receive, send)


app.add_middleware(NormalizePathMiddleware)

# In-memory registry for active & recent call sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def on_startup() -> None:
    """Ensure audio files are present or generate them with Groq TTS on startup."""
    logger.info("Server starting up. Checking audio questions...")
    api_key = settings.GROQ_API_KEY.strip()
    if api_key:
        logger.info("GROQ_API_KEY found. Generating/verifying question audio with Groq TTS...")
        from generate_question_audio import generate_audio_for_questions
        try:
            generate_audio_for_questions(
                questions_file=settings.QUESTIONS_FILE,
                output_dir=settings.AUDIO_DIR,
                use_mock=False,
                force=False,
            )
            logger.info("Question audio verified successfully.")
        except Exception as e:
            logger.warning("Error generating audio on startup: %s", e)
    else:
        logger.info("GROQ_API_KEY not configured. Using pre-existing audio files.")

    # Pre-load Silero VAD model into memory for zero latency
    try:
        from app.vad import get_silero_model
        get_silero_model()
        logger.info("Silero VAD pre-loaded successfully on startup.")
    except Exception as e:
        logger.warning("Could not pre-load Silero VAD model on startup: %s", e)


class TriggerCallRequest(BaseModel):
    to_number: Optional[str] = None
    caller_id: Optional[str] = None
    stream_url: Optional[str] = None
    app_id: Optional[str] = None
    time_limit: Optional[int] = None


def load_questions() -> List[Dict[str, str]]:
    """Load screening questions from JSON config."""
    q_path = settings.QUESTIONS_FILE
    if not q_path.exists():
        logger.error("Questions file not found at %s", q_path)
        return []
    with open(q_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system status and audio readiness."""
    questions = load_questions()
    audio_status = {}
    for q in questions:
        q_id = q.get("id", "")
        file_path = settings.AUDIO_DIR / f"{q_id}.wav"
        audio_status[q_id] = {
            "exists": file_path.exists(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }

    for clip in ("ack", "inaudible", "conclusion"):
        clip_path = settings.AUDIO_DIR / f"{clip}.wav"
        audio_status[clip] = {
            "exists": clip_path.exists(),
            "size_bytes": clip_path.stat().st_size if clip_path.exists() else 0,
        }

    return {
        "status": "healthy",
        "service": "AI Voice Screening Agent",
        "questions_count": len(questions),
        "audio_files": audio_status,
        "configured_models": {
            "llm": settings.GROQ_LLM_MODEL,
            "stt": settings.GROQ_STT_MODEL,
            "tts": settings.GROQ_TTS_MODEL,
        },
        "guardrails": {
            "max_silence_seconds": settings.MAX_SILENCE_SECONDS,
            "max_answer_seconds": settings.MAX_ANSWER_SECONDS,
            "total_call_timeout_seconds": settings.TOTAL_CALL_TIMEOUT_SECONDS,
        },
    }


@app.post("/call/trigger")
async def trigger_call(req: TriggerCallRequest = TriggerCallRequest()) -> Dict[str, Any]:
    """Internal manual trigger endpoint to start an outbound screening call via Exotel."""
    try:
        result = await exotel_client.trigger_screening_call(
            to_number=req.to_number,
            caller_id=req.caller_id,
            stream_url=req.stream_url,
            app_id=req.app_id,
            time_limit=req.time_limit,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Failed to trigger call: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error triggering call: {str(e)}")


async def send_audio_file(
    websocket: WebSocket,
    stream_sid: str,
    wav_path: Path,
    question_id: str,
) -> None:
    """Stream a pre-generated 8kHz WAV file to Exotel in 3200-byte PCM chunks."""
    if not wav_path.exists():
        logger.error("Audio file for %s missing at %s", question_id, wav_path)
        return

    pcm_bytes, sr, ch = wav_to_pcm16_bytes(wav_path)
    chunks = chunk_pcm_for_exotel(pcm_bytes, chunk_size=3200)

    logger.info(
        "Streaming %s (%d bytes PCM, %d chunks) to stream %s",
        wav_path.name,
        len(pcm_bytes),
        len(chunks),
        stream_sid,
    )

    for chunk in chunks:
        payload = base64.b64encode(chunk).decode("utf-8")
        media_msg = {
            "event": "media",
            "stream_sid": stream_sid,
            "media": {"payload": payload},
        }
        await websocket.send_text(json.dumps(media_msg))
        # Pacing prevents buffer underflow on telephony gateway (tunable via settings)
        if settings.STREAM_CHUNK_INTERVAL_SECONDS > 0:
            await asyncio.sleep(settings.STREAM_CHUNK_INTERVAL_SECONDS)

    # Send mark event signaling playback completion
    clean_mark = question_id if question_id.endswith("_end") else f"{question_id}_end"
    mark_msg = {
        "event": "mark",
        "stream_sid": stream_sid,
        "mark": {"name": clean_mark},
    }
    await websocket.send_text(json.dumps(mark_msg))
    logger.info("Finished streaming %s to stream %s (mark: %s)", wav_path.name, stream_sid, clean_mark)


@app.websocket("/media")
async def websocket_media_endpoint(websocket: WebSocket) -> None:
    """Exotel Voicebot Applet Bidirectional WebSocket Endpoint.

    Protocol:
    1. Exotel connects -> sends 'connected' and 'start' events.
    2. Server plays pre-recorded question audio (q1.wav).
    3. Server receives candidate PCM audio frames -> Silero VAD detects end of turn.
    4. Server transcribes answer via Groq Whisper.
    5. Server plays next question, repeating until questions exhausted.
    6. Call ends -> Server generates Groq LLaMA 3.3 70B candidate evaluation report.
    """
    await websocket.accept()
    logger.info("WebSocket /media connection accepted.")

    questions = load_questions()
    if not questions:
        logger.error("No screening questions found! Closing socket.")
        await websocket.close()
        return

    stream_sid = ""
    call_sid = ""
    candidate_phone = ""
    current_q_idx = 0
    is_streaming_bot_audio = False

    turn_detector = TurnDetector(min_answer_seconds=4.0, initial_silence_timeout=12.0)
    transcripts: List[Dict[str, Any]] = []
    call_start_time = time.monotonic()
    total_candidate_audio_sec = 0.0
    report_done = False

    async def finalize_session(reason: str = "normal") -> None:
        nonlocal report_done, total_candidate_audio_sec
        if report_done:
            return
        report_done = True
        duration = time.monotonic() - call_start_time
        logger.info(
            "Finalizing screening session for call %s (Reason: %s, Duration: %.1fs)",
            call_sid or "UNKNOWN",
            reason,
            duration,
        )
        sid_key = call_sid or stream_sid or f"call_{int(time.time())}"
        session_data = active_sessions.get(sid_key, {})
        session_data.update({
            "call_sid": sid_key,
            "candidate_phone": candidate_phone,
            "duration": duration,
            "transcripts": transcripts,
            "status": "completed",
        })
        active_sessions[sid_key] = session_data

        try:
            await generate_evaluation_report(
                call_sid=sid_key,
                candidate_phone=candidate_phone,
                transcript_records=transcripts,
                call_duration_seconds=duration,
                audio_transcribed_seconds=total_candidate_audio_sec,
            )
        except Exception as err:
            logger.error("Failed to generate evaluation report: %s", err)

    mark_events: Dict[str, asyncio.Event] = {}
    turn_completed_event: asyncio.Event = asyncio.Event()
    is_bot_speaking: bool = True
    orchestrator_task: Optional[asyncio.Task] = None

    async def play_audio_and_wait(mark_name: str, wav_file: Path, max_wait: float = 35.0) -> None:
        nonlocal is_bot_speaking
        is_bot_speaking = True
        ev = asyncio.Event()
        mark_events[mark_name] = ev
        await send_audio_file(websocket, stream_sid, wav_file, mark_name)
        wait_limit = 0.5 if settings.STREAM_CHUNK_INTERVAL_SECONDS <= 0 else max_wait
        try:
            logger.info("Waiting for Exotel mark [%s] confirmation (max %.1fs)...", mark_name, wait_limit)
            await asyncio.wait_for(ev.wait(), timeout=wait_limit)
            logger.info("Exotel playback mark [%s] confirmed complete.", mark_name)
        except asyncio.TimeoutError:
            logger.info("Playback mark [%s] wait ended (waited %.1fs). Proceeding.", mark_name, wait_limit)
        finally:
            mark_events.pop(mark_name, None)
            await asyncio.sleep(0.3)
            is_bot_speaking = False

    async def run_interview_flow() -> None:
        nonlocal current_q_idx, total_candidate_audio_sec
        try:
            # Brief pause for telephony audio path stabilization
            await asyncio.sleep(0.5)

            for idx, q in enumerate(questions):
                current_q_idx = idx
                q_id = q["id"]
                wav_file = settings.AUDIO_DIR / f"{q_id}.wav"

                logger.info("Orchestrator: Playing question [%s]: %s", q_id, q["text"])
                await play_audio_and_wait(f"{q_id}_end", wav_file, max_wait=35.0 if idx == 0 else 15.0)

                # Reset VAD and listen
                turn_detector.reset(min_answer_seconds=3.0)
                turn_completed_event.clear()
                logger.info("Listening for candidate response to [%s]...", q_id)

                # Wait until candidate finishes answering
                await turn_completed_event.wait()

                # Process and transcribe
                buffered_pcm = turn_detector.get_audio_bytes()
                ans_duration = len(buffered_pcm) / (8000 * 2)
                total_candidate_audio_sec += ans_duration
                logger.info(
                    "Candidate answer for [%s] finished (Reason: %s, Audio: %.2fs)",
                    q_id,
                    turn_detector.turn_complete_reason,
                    ans_duration,
                )

                transcript = await transcribe_answer(buffered_pcm)
                clean_text = transcript.strip().rstrip(".").lower()
                is_inaudible = False
                if len(buffered_pcm) < 6400 or clean_text in ("", "thank you", "thanks", "you", "[unintelligible / silence]", "[no speech detected]"):
                    transcript = "[No speech detected]" if ans_duration < 1.0 else "[No clear response recorded]"
                    is_inaudible = True
                elif transcript.startswith("[No ") or transcript.startswith("[Transcription error"):
                    is_inaudible = True

                logger.info("Transcribed [%s]: '%s' (is_inaudible=%s)", q_id, transcript, is_inaudible)
                transcripts.append({
                    "question_id": q_id,
                    "question": q["text"],
                    "answer": transcript,
                    "duration_seconds": round(ans_duration, 2),
                })

                # --- DYNAMIC FOLLOW-UP FEATURE ---
                # When candidate gives a substantive answer to Q1, generate an intelligent follow-up using LLM + Neural TTS
                did_followup = False
                if (
                    settings.ENABLE_DYNAMIC_FOLLOWUP
                    and not is_inaudible
                    and idx == 0
                    and len(transcript.split()) >= 3
                ):
                    logger.info("Initiating dynamic follow-up pipeline for [%s]...", q_id)
                    followup_file = settings.AUDIO_DIR / f"followup_{q_id}_{int(time.time())}.wav"

                    async def _create_followup_audio() -> Optional[tuple[str, Path]]:
                        f_text = await generate_followup_question(
                            q["text"],
                            transcript,
                            timeout_seconds=settings.FOLLOWUP_MAX_TIMEOUT_SECONDS,
                        )
                        if not f_text:
                            return None
                        tts_ok = await synthesize_followup_speech(
                            f_text,
                            followup_file,
                            timeout_seconds=settings.FOLLOWUP_MAX_TIMEOUT_SECONDS,
                        )
                        if tts_ok and followup_file.exists():
                            return f_text, followup_file
                        return None

                    followup_task = asyncio.create_task(_create_followup_audio())

                    # Concurrently play conversational acknowledgment ('ack.wav') to keep conversation natural
                    ack_file = settings.AUDIO_DIR / "ack.wav"
                    if ack_file.exists():
                        logger.info("Streaming conversational bridge 'ack.wav' while preparing follow-up...")
                        await play_audio_and_wait("ack_end", ack_file, max_wait=10.0)

                    followup_result = None
                    try:
                        followup_result = await asyncio.wait_for(followup_task, timeout=2.5)
                    except (asyncio.TimeoutError, Exception) as f_err:
                        logger.warning("Follow-up audio not ready in time (%s). Proceeding cleanly.", f_err)

                    if followup_result:
                        f_question_text, f_wav_path = followup_result
                        logger.info("Asking dynamic follow-up: '%s'", f_question_text)
                        await play_audio_and_wait("followup_q1_end", f_wav_path, max_wait=15.0)

                        turn_detector.reset(min_answer_seconds=2.5)
                        turn_completed_event.clear()
                        logger.info("Listening for candidate response to follow-up...")
                        await turn_completed_event.wait()

                        f_pcm = turn_detector.get_audio_bytes()
                        f_duration = len(f_pcm) / (8000 * 2)
                        total_candidate_audio_sec += f_duration
                        f_transcript = await transcribe_answer(f_pcm)
                        logger.info("Transcribed follow-up answer: '%s' (%.2fs)", f_transcript, f_duration)

                        transcripts.append({
                            "question_id": f"{q_id}_followup",
                            "question": f_question_text,
                            "answer": f_transcript,
                            "duration_seconds": round(f_duration, 2),
                        })
                        did_followup = True

                        # Cleanup temporary WAV
                        try:
                            if f_wav_path.exists():
                                f_wav_path.unlink()
                        except Exception:
                            pass

                # If no dynamic follow-up was executed and there are more questions, play standard bridge
                if not did_followup and (idx + 1 < len(questions)):
                    bridge_clip = "inaudible" if is_inaudible else "ack"
                    bridge_file = settings.AUDIO_DIR / f"{bridge_clip}.wav"
                    if bridge_file.exists():
                        logger.info("Streaming conversational bridge '%s.wav'...", bridge_clip)
                        await play_audio_and_wait(f"{bridge_clip}_end", bridge_file, max_wait=10.0)

            # All questions finished
            logger.info("All screening questions completed.")
            conclusion_file = settings.AUDIO_DIR / "conclusion.wav"
            if conclusion_file.exists():
                logger.info("Streaming closing statement 'conclusion.wav'...")
                await play_audio_and_wait("conclusion_end", conclusion_file, max_wait=15.0)

            await finalize_session(reason="all_questions_completed")
        except asyncio.CancelledError:
            logger.info("Interview orchestrator task cancelled.")
        except Exception as flow_err:
            logger.error("Error in interview flow: %s", flow_err, exc_info=True)
            await finalize_session(reason=f"error_{str(flow_err)}")

    try:
        while not report_done:
            # Check hard call timeout safety net
            elapsed = time.monotonic() - call_start_time
            if elapsed >= settings.TOTAL_CALL_TIMEOUT_SECONDS:
                logger.warning(
                    "TOTAL_CALL_TIMEOUT_SECONDS (%ds) exceeded! Terminating call.",
                    settings.TOTAL_CALL_TIMEOUT_SECONDS,
                )
                await finalize_session(reason="total_timeout_exceeded")
                break

            try:
                raw_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                # Check silence/timeouts when candidate is expected to speak
                if not is_bot_speaking and turn_detector.check_timeouts():
                    turn_completed_event.set()
                continue

            event = json.loads(raw_data)
            ev_type = str(event.get("event") or event.get("Event") or "").lower()

            if ev_type == "connected":
                logger.info("Exotel event: connected")
                continue

            elif ev_type == "start":
                start_data = event.get("start") or event.get("Start") or {}
                stream_sid = (
                    event.get("stream_sid")
                    or event.get("StreamSid")
                    or start_data.get("stream_sid")
                    or start_data.get("StreamSid", "")
                )
                call_sid = start_data.get("call_sid") or start_data.get("CallSid", "")
                candidate_phone = (
                    start_data.get("from")
                    or start_data.get("From")
                    or settings.EXOTEL_CALLER_NUMBER
                )

                logger.info(
                    "Exotel event: start (StreamSID: %s, CallSID: %s, From: %s)",
                    stream_sid,
                    call_sid,
                    candidate_phone,
                )

                active_sessions[call_sid or stream_sid] = {
                    "stream_sid": stream_sid,
                    "call_sid": call_sid,
                    "candidate_phone": candidate_phone,
                    "start_time": call_start_time,
                }

                # Start the interview flow as an independent orchestrator task
                if orchestrator_task is None or orchestrator_task.done():
                    orchestrator_task = asyncio.create_task(run_interview_flow())

            elif ev_type == "mark":
                mark_data = event.get("mark") or event.get("Mark") or {}
                mark_name = mark_data.get("name") or mark_data.get("Name", "")
                logger.info("Exotel mark received: %s", mark_name)
                for k, ev in list(mark_events.items()):
                    if k == mark_name or mark_name.startswith(k) or k.startswith(mark_name):
                        ev.set()

            elif ev_type == "media":
                # Ignore audio while bot itself is speaking / prompt is playing
                if is_bot_speaking:
                    continue

                media_data = event.get("media") or event.get("Media") or {}
                payload_b64 = (
                    media_data.get("payload")
                    or media_data.get("Payload")
                    or event.get("payload")
                    or event.get("Payload")
                    or ""
                )
                if not payload_b64:
                    continue

                pcm_chunk = base64.b64decode(payload_b64)
                turn_finished = turn_detector.feed_audio(pcm_chunk)
                if turn_finished:
                    turn_completed_event.set()

            elif ev_type in ("stop", "closed"):
                logger.info("Exotel stop event received.")
                if orchestrator_task and not orchestrator_task.done():
                    orchestrator_task.cancel()
                await finalize_session(reason="exotel_stop_received")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by Exotel/client.")
        if orchestrator_task and not orchestrator_task.done():
            orchestrator_task.cancel()
        await finalize_session(reason="websocket_disconnect")
    except Exception as e:
        logger.error("Error during WebSocket streaming session: %s", e, exc_info=True)
        if orchestrator_task and not orchestrator_task.done():
            orchestrator_task.cancel()
        await finalize_session(reason=f"error_{str(e)}")
    finally:
        if orchestrator_task and not orchestrator_task.done():
            orchestrator_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@app.api_route("/call/status", methods=["GET", "POST"])
async def call_status_webhook(request: Request) -> JSONResponse:
    """Exotel StatusCallback & Passthru Applet webhook.

    Triggered when the call terminates. Captures call metadata, final duration,
    and ensures evaluation report is generated.
    """
    params: Dict[str, Any] = {}
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                params = await request.json()
            except Exception:
                pass
        else:
            try:
                form = await request.form()
                params = dict(form)
            except Exception as e:
                logger.warning("request.form() failed, falling back to raw body parse: %s", e)
                try:
                    body_bytes = await request.body()
                    from urllib.parse import parse_qs
                    params = {
                        k: v[0] if len(v) == 1 else v
                        for k, v in parse_qs(body_bytes.decode("utf-8", errors="ignore")).items()
                    }
                except Exception:
                    params = {}
    else:
        params = dict(request.query_params)

    logger.info("Exotel /call/status webhook received: %s", params)

    call_sid = (
        params.get("CallSid")
        or params.get("callsid")
        or params.get("Sid")
        or params.get("sid")
        or ""
    )
    stream_sid = params.get("StreamSid") or params.get("streamsid") or ""
    status = params.get("Status") or params.get("status") or "completed"
    duration_str = (
        params.get("CallDuration")
        or params.get("duration")
        or params.get("Duration")
        or "0"
    )

    try:
        duration_sec = float(duration_str)
    except ValueError:
        duration_sec = 0.0

    key = call_sid or stream_sid
    session = active_sessions.get(key, {})

    if session:
        session["status"] = status
        if duration_sec > 0:
            session["duration"] = duration_sec

        # Trigger report if not already generated
        report_file = settings.REPORTS_DIR / f"{key}.json"
        if not report_file.exists() and session.get("transcripts"):
            asyncio.create_task(
                generate_evaluation_report(
                    call_sid=key,
                    candidate_phone=session.get("candidate_phone", ""),
                    transcript_records=session.get("transcripts", []),
                    call_duration_seconds=duration_sec or session.get("duration", 0.0),
                    audio_transcribed_seconds=0.0,
                )
            )

    return JSONResponse(
        status_code=200,
        content={"success": True, "call_sid": key, "status": status},
    )


@app.get("/reports")
async def list_reports() -> JSONResponse:
    """List all candidate evaluation reports generated on this instance."""
    reports_dir = settings.REPORTS_DIR
    if not reports_dir.exists():
        return JSONResponse(content={"total": 0, "reports": []})

    import os
    files = sorted(reports_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    summary_list = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            base = settings.PUBLIC_BASE_URL.rstrip('/')
            summary_list.append({
                "call_sid": data.get("call_sid", f.stem),
                "candidate_phone": data.get("candidate_phone", "N/A"),
                "recommendation": data.get("overall_recommendation", {}).get("decision", "N/A"),
                "duration_seconds": data.get("call_duration_seconds", 0),
                "total_cost_usd": data.get("cost_estimate", {}).get("total_estimated_cost_usd", 0),
                "view_url": f"{base}/reports/{f.stem}/view",
                "pdf_url": f"{base}/reports/{f.stem}/pdf",
                "json_url": f"{base}/reports/{f.stem}",
                "markdown_url": f"{base}/reports/{f.stem}/markdown",
            })
        except Exception:
            continue
    return JSONResponse(content={"total": len(summary_list), "reports": summary_list})


@app.get("/reports/{call_sid}")
async def get_report(call_sid: str) -> JSONResponse:
    """Get the raw JSON evaluation report for a specific call SID."""
    json_path = settings.REPORTS_DIR / f"{call_sid}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for call {call_sid}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/reports/{call_sid}/markdown", response_class=PlainTextResponse)
async def get_report_markdown(call_sid: str) -> PlainTextResponse:
    """Get the Markdown-formatted evaluation report."""
    md_path = settings.REPORTS_DIR / f"{call_sid}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for call {call_sid}")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content=content)


@app.get("/reports/{call_sid}/pdf")
async def get_report_pdf(call_sid: str) -> Response:
    """Download candidate screening report as PDF."""
    pdf_path = settings.REPORTS_DIR / f"{call_sid}.pdf"
    if not pdf_path.exists():
        json_path = settings.REPORTS_DIR / f"{call_sid}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Report not found for call {call_sid}")
        with open(json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        try:
            from app.pdf_generator import generate_candidate_pdf
            generate_candidate_pdf(report_data, pdf_path)
        except Exception as err:
            logger.error("Error generating PDF on the fly: %s", err)
            raise HTTPException(status_code=500, detail="Failed to generate PDF")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="screening_report_{call_sid}.pdf"'
        },
    )


@app.get("/reports/{call_sid}/view", response_class=HTMLResponse)
async def view_report_html(call_sid: str) -> HTMLResponse:
    """View candidate screening report nicely formatted in HTML in your browser."""
    json_path = settings.REPORTS_DIR / f"{call_sid}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for call {call_sid}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    decision = str(data.get("overall_recommendation", {}).get("decision", "HOLD")).upper()
    badge_color = "#10b981" if decision == "PROCEED" else ("#ef4444" if decision == "REJECT" else "#f59e0b")
    rec_just = data.get("overall_recommendation", {}).get("justification", "")
    crit = data.get("criteria", {})
    cost = data.get("cost_estimate", {})
    usd_to_inr = 85.0
    tel_inr = float(cost.get("telephony_cost_usd", 0) or 0) * usd_to_inr
    stt_inr = float(cost.get("stt_cost_usd", 0) or 0) * usd_to_inr
    llm_inr = float(cost.get("llm_cost_usd", 0) or 0) * usd_to_inr
    tot_inr = float(cost.get("total_estimated_cost_usd", 0) or 0) * usd_to_inr
    transcript = data.get("transcript", [])
    observations = data.get("key_observations", [])

    rows_html = "".join(
        f"<tr><td style='padding:10px;border-bottom:1px solid #e5e7eb;font-weight:600;'>{k.replace('_', ' ').title()}</td>"
        f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:center;'><strong>{v.get('score', 'N/A')} / 5</strong></td>"
        f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;color:#4b5563;'>{v.get('justification', '')}</td></tr>"
        for k, v in crit.items()
    )

    obs_html = "".join(f"<li style='margin-bottom:6px;'>{o}</li>" for o in observations)

    qa_html = "".join(
        f"<div style='margin-bottom:16px;padding:12px;background:#f9fafb;border-radius:8px;border-left:4px solid #3b82f6;'>"
        f"<p style='margin:0 0 6px 0;font-weight:600;color:#1e3a8a;'>Q ({item.get('question_id', '')}): {item.get('question', '')}</p>"
        f"<p style='margin:0;color:#1f2937;'><strong>Candidate:</strong> {item.get('answer', '[No response]')}</p>"
        f"</div>"
        for item in transcript
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Candidate Screening Report - {call_sid}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f3f4f6; margin: 0; padding: 24px; color: #111827; }}
    .container {{ max-width: 860px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 32px; }}
    .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; color: white; font-weight: 700; font-size: 14px; background: {badge_color}; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th {{ background: #f9fafb; padding: 10px; text-align: left; font-size: 13px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb; }}
    .cost-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }}
    .cost-card {{ background: #f9fafb; padding: 12px 16px; border-radius: 8px; border: 1px solid #e5e7eb; }}
    .cost-card div:first-child {{ font-size: 12px; color: #6b7280; text-transform: uppercase; font-weight: 600; }}
    .cost-card div:last-child {{ font-size: 20px; color: #111827; font-weight: 700; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e5e7eb; padding-bottom:16px;">
      <div>
        <h1 style="margin:0 0 6px 0; font-size:24px;">Candidate Screening Report</h1>
        <p style="margin:0; color:#6b7280; font-size:14px;">Call SID: <code>{call_sid}</code> | Phone: <strong>{data.get('candidate_phone', 'N/A')}</strong></p>
      </div>
      <div style="display:flex; gap:12px; align-items:center;">
        <a href="/reports/{call_sid}/pdf" style="display:inline-block; padding:8px 16px; background:#2563eb; color:white; font-weight:600; font-size:13px; text-decoration:none; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">📥 Download PDF</a>
        <span class="badge">{decision}</span>
      </div>
    </div>

    <div style="margin-top:20px; padding:16px; background:#eff6ff; border-radius:8px;">
      <strong>Recommendation Justification:</strong> {rec_just}
    </div>

    <h3 style="margin-top:28px; margin-bottom:8px;">Evaluation Criteria (1-5 Scale)</h3>
    <table>
      <thead><tr><th>Criterion</th><th style="text-align:center;">Score</th><th>Justification</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>

    <h3 style="margin-top:28px; margin-bottom:8px;">Key Observations</h3>
    <ul style="color:#374151; padding-left:20px;">{obs_html}</ul>

    <h3 style="margin-top:28px; margin-bottom:8px;">Estimated Call Cost Breakdown</h3>
    <div class="cost-grid">
      <div class="cost-card"><div>Telephony</div><div>₹{tel_inr:.4f}</div></div>
      <div class="cost-card"><div>Whisper STT</div><div>₹{stt_inr:.4f}</div></div>
      <div class="cost-card"><div>LLaMA 3.3 LLM</div><div>₹{llm_inr:.4f}</div></div>
      <div class="cost-card" style="border-color:#10b981; background:#ecfdf5;"><div>Total Cost</div><div style="color:#059669;">₹{tot_inr:.4f}</div></div>
    </div>

    <h3 style="margin-top:28px; margin-bottom:12px;">Full Screening Transcript</h3>
    {qa_html}
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


def start() -> None:
    """Start uvicorn server."""
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=settings.PORT, reload=False)


if __name__ == "__main__":
    start()
