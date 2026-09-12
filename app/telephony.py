"""Exotel Cloud Telephony Client for placing outbound screening calls."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ExotelClient:
    """Client for Exotel Call Connect API."""

    def __init__(
        self,
        account_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        api_token: Optional[str] = None,
        subdomain: Optional[str] = None,
    ) -> None:
        self.account_sid = account_sid or settings.EXOTEL_ACCOUNT_SID
        self.api_key = api_key or settings.EXOTEL_API_KEY
        self.api_token = api_token or settings.EXOTEL_API_TOKEN
        self.subdomain = subdomain or settings.EXOTEL_SUBDOMAIN

    @property
    def base_url(self) -> str:
        sub = self.subdomain.strip()
        if not sub.startswith("http"):
            sub = f"https://{sub}"
        return sub.rstrip("/")

    async def trigger_screening_call(
        self,
        to_number: Optional[str] = None,
        caller_id: Optional[str] = None,
        stream_url: Optional[str] = None,
        app_id: Optional[str] = None,
        time_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trigger an outbound screening call via Exotel's Calls/connect API.

        Connects either directly to a bidirectional WebSocket stream (Voicebot)
        or via an Exotel Call Flow (Applet).
        """
        destination = (to_number or settings.EXOTEL_CALLER_NUMBER).strip()
        from_phone = (caller_id or settings.EXOTEL_EXOPHONE).strip()
        flow_id = app_id or settings.EXOTEL_APP_ID
        ws_url = stream_url or settings.public_ws_url
        max_duration = time_limit or settings.TOTAL_CALL_TIMEOUT_SECONDS

        if not destination:
            raise ValueError("Destination phone number (to_number / EXOTEL_CALLER_NUMBER) is required")
        if not from_phone:
            raise ValueError("ExoPhone caller ID (caller_id / EXOTEL_EXOPHONE) is required")
        if not self.account_sid or not self.api_key or not self.api_token:
            logger.warning("Exotel credentials not configured. Returning simulated response for test/dev.")
            return {
                "success": False,
                "simulated": True,
                "message": "Exotel credentials not set in .env",
                "call": {
                    "sid": "SIMULATED_EXOTEL_CALL_SID",
                    "status": "queued",
                    "from": destination,
                    "caller_id": from_phone,
                    "stream_url": ws_url,
                },
            }

        endpoint = f"{self.base_url}/v1/Accounts/{self.account_sid}/Calls/connect.json"

        # Build form payload according to Exotel API specification
        data: Dict[str, Any] = {
            "From": destination,
            "CallerId": from_phone,
            "CallType": "trans",
            "TimeLimit": str(max_duration),
            "StatusCallback": settings.public_status_callback_url,
            "StatusCallbackEvents[]": "terminal",
        }

        if flow_id:
            # Flow-based connection
            data["Url"] = f"http://my.exotel.com/{self.account_sid}/exoml/start_voice/{flow_id}"
        else:
            # Direct stream connection (AgentStream Voicebot)
            data["StreamUrl"] = ws_url
            data["StreamType"] = "bidirectional"

        logger.info(
            "Triggering Exotel call to %s from %s (Endpoint: %s)",
            destination,
            from_phone,
            endpoint,
        )

        auth = (self.api_key, self.api_token)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                auth=auth,
                data=data,
            )

        if response.status_code in (200, 201):
            json_resp = response.json()
            logger.info("Exotel call triggered successfully: %s", json_resp)
            return {"success": True, "data": json_resp}
        else:
            err_msg = f"Exotel API error ({response.status_code}): {response.text}"
            logger.error(err_msg)
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text,
            }


exotel_client = ExotelClient()
