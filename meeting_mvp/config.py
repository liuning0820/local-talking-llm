"""Configuration with safe local defaults."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .errors import MeetingError, PrivacyError


def _target_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_data_dir() -> Path:
    override = os.getenv("MEETING_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Windows":
        root = os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(root) / "local-talking-llm"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "local-talking-llm"
    return Path.home() / ".local" / "share" / "local-talking-llm"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunk_seconds: int = 300
    queue_chunks: int = 32
    title_max_length: int = 200
    stt_backend: str = "whisper"
    whisper_model: str = "base"
    asr_url: str = "http://127.0.0.1:8080/v1"
    asr_model: str = ""
    llm_backend: str = "ollama"
    llm_url: str = "http://127.0.0.1:11434"
    llm_model: str = "gemma3:1b"
    allow_network: bool = False
    request_timeout: float = 120.0
    language: str | None = None
    network_confirmed: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        chunk = int(os.getenv("MEETING_CHUNK_SECONDS", "300"))
        if not 30 <= chunk <= 900:
            raise MeetingError("MEETING_CHUNK_SECONDS must be between 30 and 900", "INVALID_CHUNK_SECONDS")
        title_max = int(os.getenv("MEETING_TITLE_MAX_LENGTH", "200"))
        if not 1 <= title_max <= 500:
            raise MeetingError("MEETING_TITLE_MAX_LENGTH must be between 1 and 500", "INVALID_TITLE_LENGTH")
        return cls(
            data_dir=default_data_dir(),
            chunk_seconds=chunk,
            queue_chunks=max(4, int(os.getenv("MEETING_QUEUE_CHUNKS", "32"))),
            title_max_length=title_max,
            stt_backend=os.getenv("MEETING_STT_BACKEND", "whisper").lower(),
            whisper_model=os.getenv("MEETING_WHISPER_MODEL", "base"),
            asr_url=os.getenv("MEETING_ASR_URL", "http://127.0.0.1:8080/v1"),
            asr_model=os.getenv("MEETING_ASR_MODEL", ""),
            llm_backend=os.getenv("MEETING_LLM_BACKEND", "ollama").lower(),
            llm_url=os.getenv("MEETING_LLM_URL", "http://127.0.0.1:11434"),
            llm_model=os.getenv("MEETING_LLM_MODEL", "gemma3:1b"),
            allow_network=_bool(os.getenv("MEETING_ALLOW_NETWORK"), False),
            request_timeout=float(os.getenv("MEETING_REQUEST_TIMEOUT", "120")),
            language=os.getenv("MEETING_LANGUAGE") or None,
        )

    def require_network_confirmation(self, url: str, data_type: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        local = host in {"127.0.0.1", "localhost", "::1"}
        self.validate_url(url)
        if not local and not self.network_confirmed:
            raise PrivacyError(
                f"Remote {data_type} upload to {_target_host(url)} is not confirmed; "
                "rerun with --yes after reviewing the target.",
                "REMOTE_NETWORK_CONFIRMATION_REQUIRED",
            )

    def validate_url(self, url: str | None = None) -> None:
        target = url or self.llm_url
        parsed = urlparse(target)
        host = (parsed.hostname or "").lower()
        local = host in {"127.0.0.1", "localhost", "::1"}
        if not parsed.scheme or not host:
            raise MeetingError(f"Invalid model URL: {target}", "INVALID_MODEL_URL")
        if not local and not self.allow_network:
            raise PrivacyError(
                "Remote model URL rejected while MEETING_ALLOW_NETWORK=false; "
                "use a loopback URL or explicitly enable networking.",
                "REMOTE_NETWORK_DISABLED",
            )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "meetings").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
