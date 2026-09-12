"""Integration tests for FastAPI endpoints and WebSocket /media call flow."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.server import app


@pytest.fixture
def client() -> TestClient:
    settings.STREAM_CHUNK_INTERVAL_SECONDS = 0.0
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["questions_count"] == 9
    assert "audio_files" in data
    assert "guardrails" in data
    assert data["guardrails"]["max_silence_seconds"] == 3.0


def test_call_trigger_endpoint(client: TestClient) -> None:
    response = client.post(
        "/call/trigger",
        json={"to_number": "+919876543210", "caller_id": "08047491899"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "call" in data or "data" in data or not data["success"]


def test_call_status_webhook_get_and_post(client: TestClient) -> None:
    # Test GET status webhook (Exotel Passthru style)
    get_resp = client.get(
        "/call/status?CallSid=CAtest123&Status=completed&CallDuration=45"
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["call_sid"] == "CAtest123"

    # Test POST status webhook
    post_resp = client.post(
        "/call/status",
        data={"CallSid": "CAtest456", "Status": "completed", "CallDuration": "60"},
    )
    assert post_resp.status_code == 200
    assert post_resp.json()["call_sid"] == "CAtest456"


def test_websocket_media_session(client: TestClient, tmp_path: Path) -> None:
    # Ensure audio files exist for q1
    q1_file = settings.AUDIO_DIR / "q1.wav"
    assert q1_file.exists(), "Audio files must be generated before running test"

    with client.websocket_connect("/media") as ws:
        # 1. Send Exotel 'connected' event
        ws.send_text(json.dumps({"event": "connected"}))

        # 2. Send Exotel 'start' event
        start_payload = {
            "event": "start",
            "stream_sid": "MZteststream123",
            "start": {
                "stream_sid": "MZteststream123",
                "call_sid": "CAtestcall999",
                "from": "+919876543210",
                "to": "08047491899",
                "media_format": {
                    "encoding": "audio/x-raw",
                    "sample_rate": "8000",
                    "bit_rate": "16",
                },
            },
        }
        ws.send_text(json.dumps(start_payload))

        # 3. Read outgoing audio chunks streamed by the bot
        received_media_chunk = False
        received_mark = False

        for _ in range(50):
            try:
                msg_text = ws.receive_text()
                msg = json.loads(msg_text)
                if msg.get("event") == "media":
                    received_media_chunk = True
                    payload = msg.get("media", {}).get("payload", "")
                    assert len(payload) > 0
                elif msg.get("event") == "mark":
                    received_mark = True
                    break
            except Exception:
                break

        assert received_media_chunk, "Server should stream question audio chunks"
        assert received_mark, "Server should send mark event at end of question"

        # 4. Simulate candidate audio (silence frames to complete turn)
        silence_bytes = b"\x00\x00" * 4000  # 0.5s of 8kHz silence
        silence_b64 = base64.b64encode(silence_bytes).decode("utf-8")

        # Send multiple silence chunks to trigger initial_silence_timeout or turn advance
        for _ in range(25):
            ws.send_text(
                json.dumps({
                    "event": "media",
                    "stream_sid": "MZteststream123",
                    "media": {"payload": silence_b64},
                })
            )

        # 5. Send stop event from Exotel
        ws.send_text(
            json.dumps({
                "event": "stop",
                "stream_sid": "MZteststream123",
                "stop": {"call_sid": "CAtestcall999", "reason": "callended"},
            })
        )


def test_reports_endpoints(client: TestClient) -> None:
    # 1. Test /reports listing
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert "reports" in resp.json()

    # Create dummy report for testing
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dummy_sid = "CALL_TEST_REPORT_123"
    dummy_report = {
        "call_sid": dummy_sid,
        "candidate_phone": "+919876543210",
        "call_duration_seconds": 120.0,
        "overall_recommendation": {"decision": "proceed", "justification": "Strong fit"},
        "criteria": {
            "communication_clarity": {"score": 4, "justification": "Clear voice"},
        },
        "cost_estimate": {"total_estimated_cost_usd": 0.0035},
        "transcript": [{"question_id": "q1", "question": "Q?", "answer": "A!"}],
    }
    with open(settings.REPORTS_DIR / f"{dummy_sid}.json", "w", encoding="utf-8") as f:
        json.dump(dummy_report, f)
    with open(settings.REPORTS_DIR / f"{dummy_sid}.md", "w", encoding="utf-8") as f:
        f.write("# Dummy Report")

    try:
        # Test GET /reports/{call_sid}
        json_resp = client.get(f"/reports/{dummy_sid}")
        assert json_resp.status_code == 200
        assert json_resp.json()["call_sid"] == dummy_sid

        # Test GET /reports/{call_sid}/markdown
        md_resp = client.get(f"/reports/{dummy_sid}/markdown")
        assert md_resp.status_code == 200
        assert "# Dummy Report" in md_resp.text

        # Test GET /reports/{call_sid}/view (HTML)
        view_resp = client.get(f"/reports/{dummy_sid}/view")
        assert view_resp.status_code == 200
        assert "Candidate Screening Report" in view_resp.text
        assert "PROCEED" in view_resp.text
    finally:
        # Clean up dummy report
        for ext in (".json", ".md"):
            p = settings.REPORTS_DIR / f"{dummy_sid}{ext}"
            if p.exists():
                p.unlink()
