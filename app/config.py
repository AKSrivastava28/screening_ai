"""Configuration settings loaded from environment variables and .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Exotel Configuration
    EXOTEL_ACCOUNT_SID: str = ""
    EXOTEL_API_KEY: str = ""
    EXOTEL_API_TOKEN: str = ""
    EXOTEL_SUBDOMAIN: str = "api.exotel.com"
    EXOTEL_EXOPHONE: str = ""
    EXOTEL_CALLER_NUMBER: str = ""
    EXOTEL_APP_ID: Optional[str] = None

    # Groq Configuration
    GROQ_API_KEY: str = ""
    GROQ_LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    GROQ_TTS_MODEL: str = "canopylabs/orpheus-v1-english"
    GROQ_TTS_VOICE: str = "hannah"

    # Server Configuration
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    PORT: int = 8000

    # Call Behavior & Guardrails
    MAX_SILENCE_SECONDS: float = 1.3
    MAX_ANSWER_SECONDS: float = 45.0
    TOTAL_CALL_TIMEOUT_SECONDS: int = 480
    STREAM_CHUNK_INTERVAL_SECONDS: float = 0.19

    # Dynamic Follow-Up Settings
    ENABLE_DYNAMIC_FOLLOWUP: bool = False
    FOLLOWUP_MAX_TIMEOUT_SECONDS: float = 7.5
    FOLLOWUP_VOICE: str = "en-IN-NeerjaNeural"

    # Cost Estimation Rates (USD)
    TELEPHONY_RATE_PER_MINUTE: float = 0.015
    GROQ_WHISPER_RATE_PER_MINUTE: float = 0.00111
    GROQ_LLAMA_INPUT_PER_1M: float = 0.59
    GROQ_LLAMA_OUTPUT_PER_1M: float = 0.79

    # File Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    QUESTIONS_FILE: Path = Path("questions.json")
    AUDIO_DIR: Path = Path("audio")
    REPORTS_DIR: Path = Path("reports")

    @property
    def exotel_base_url(self) -> str:
        subdomain = self.EXOTEL_SUBDOMAIN.strip()
        if not subdomain.startswith("http"):
            subdomain = f"https://{subdomain}"
        return subdomain.rstrip("/")

    @property
    def public_ws_url(self) -> str:
        """Convert public HTTP base URL to WebSocket URL."""
        url = self.PUBLIC_BASE_URL.strip().rstrip("/")
        if url.startswith("https://"):
            return "wss://" + url[len("https://") :] + "/media"
        elif url.startswith("http://"):
            return "ws://" + url[len("http://") :] + "/media"
        elif url.startswith("wss://") or url.startswith("ws://"):
            return f"{url}/media"
        return f"wss://{url}/media"

    @property
    def public_status_callback_url(self) -> str:
        url = self.PUBLIC_BASE_URL.strip().rstrip("/")
        if not url.startswith("http"):
            url = f"https://{url}"
        return f"{url}/call/status"


settings = Settings()
