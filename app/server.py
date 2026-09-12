"""FastAPI Webhook and WebSocket server for AI Voice Screening Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.audio_utils import chunk_pcm_for_exotel, wav_to_pcm16_bytes
from app.config import settings
from app.evaluator import generate_evaluation_report
from app.stt import transcribe_answer
from app.telephony import exotel_client
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
    mark_msg = {
        "event": "mark",
        "stream_sid": stream_sid,
        "mark": {"name": f"{question_id}_end"},
    }
    await websocket.send_text(json.dumps(mark_msg))
    logger.info("Finished streaming %s to stream %s", wav_path.name, stream_sid)


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

    turn_detector = TurnDetector()
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

    try:
        while True:
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
                raw_data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            event = json.loads(raw_data)
            ev_type = event.get("event", "")

            if ev_type == "connected":
                logger.info("Exotel event: connected")
                continue

            elif ev_type == "start":
                start_data = event.get("start", {})
                stream_sid = event.get("stream_sid") or start_data.get("stream_sid", "")
                call_sid = start_data.get("call_sid", "")
                candidate_phone = start_data.get("from", settings.EXOTEL_CALLER_NUMBER)

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

                # Start first question after a brief 0.8s pause for telephony audio path stabilization
                await asyncio.sleep(0.8)
                current_q = questions[current_q_idx]
                q_id = current_q["id"]
                wav_file = settings.AUDIO_DIR / f"{q_id}.wav"
                is_streaming_bot_audio = True
                await send_audio_file(websocket, stream_sid, wav_file, q_id)
                is_streaming_bot_audio = False
                turn_detector.reset()

            elif ev_type == "media":
                # Ignore candidate audio while bot itself is streaming question audio
                if is_streaming_bot_audio:
                    continue

                media_data = event.get("media", {})
                payload_b64 = media_data.get("payload", "")
                if not payload_b64:
                    continue

                pcm_chunk = base64.b64decode(payload_b64)
                turn_finished = turn_detector.feed_audio(pcm_chunk)

                if turn_finished:
                    # Candidate finished speaking current answer
                    buffered_pcm = turn_detector.get_audio_bytes()
                    ans_duration = len(buffered_pcm) / (8000 * 2)
                    total_candidate_audio_sec += ans_duration

                    current_q = questions[current_q_idx]
                    logger.info(
                        "Answer completed for [%s]. Audio duration: %.2fs. Reason: %s",
                        current_q["id"],
                        ans_duration,
                        turn_detector.turn_complete_reason,
                    )

                    # Transcribe answer via Groq Whisper STT
                    transcript = await transcribe_answer(buffered_pcm)
                    transcripts.append({
                        "question_id": current_q["id"],
                        "question": current_q["text"],
                        "answer": transcript,
                        "duration_seconds": round(ans_duration, 2),
                    })

                    current_q_idx += 1
                    if current_q_idx < len(questions):
                        # Play next question
                        next_q = questions[current_q_idx]
                        q_id = next_q["id"]
                        wav_file = settings.AUDIO_DIR / f"{q_id}.wav"
                        logger.info("Advancing to question [%s]: %s", q_id, next_q["text"])

                        is_streaming_bot_audio = True
                        await send_audio_file(websocket, stream_sid, wav_file, q_id)
                        is_streaming_bot_audio = False
                        turn_detector.reset()
                    else:
                        # All questions answered!
                        logger.info("All screening questions completed. Ending call.")
                        # Brief sleep to allow last audio/acknowledgment to settle
                        await asyncio.sleep(1.0)
                        await finalize_session(reason="all_questions_completed")
                        break

            elif ev_type == "mark":
                logger.debug("Playback mark received: %s", event.get("mark", {}).get("name"))

            elif ev_type == "stop":
                logger.info("Exotel event: stop received.")
                await finalize_session(reason="exotel_stop_received")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by Exotel/client.")
        await finalize_session(reason="websocket_disconnect")
    except Exception as e:
        logger.error("Error during WebSocket streaming session: %s", e, exc_info=True)
        await finalize_session(reason=f"error_{str(e)}")
    finally:
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


def start() -> None:
    """Start uvicorn server."""
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=settings.PORT, reload=False)


if __name__ == "__main__":
    start()
