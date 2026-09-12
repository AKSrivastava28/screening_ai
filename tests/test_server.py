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
