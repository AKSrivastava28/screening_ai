"""Unit tests for Exotel telephony client."""

from __future__ import annotations

import asyncio
import pytest

from app.telephony import ExotelClient


def test_exotel_missing_numbers_raises() -> None:
    client = ExotelClient(account_sid="AC123", api_key="k", api_token="t")
    with pytest.raises(ValueError, match="Destination phone number"):
        asyncio.run(client.trigger_screening_call(to_number="", caller_id=""))


def test_exotel_unconfigured_credentials_returns_simulation() -> None:
    client = ExotelClient(account_sid="", api_key="", api_token="")
    res = asyncio.run(
        client.trigger_screening_call(
            to_number="+919876543210",
            caller_id="08047491899",
        )
    )
    assert not res["success"]
    assert res.get("simulated") is True
    assert "SIMULATED_EXOTEL_CALL_SID" in res["call"]["sid"]
