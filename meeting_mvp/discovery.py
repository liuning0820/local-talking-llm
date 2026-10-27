"""Model discovery for ASR/LLM endpoints (sends no meeting data)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import Settings
from .errors import MeetingError
from .llm import llm_endpoint, safe_endpoint

_API_SUFFIXES = ("/audio/transcriptions", "/chat/completions", "/models")


def is_local_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def normalize_openai_base(url: str) -> str:
    """Turn a host, /v1 base, or full endpoint into the OpenAI-compatible /v1 base."""
    url = url.strip().rstrip("/")
    for suffix in _API_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    parsed = urlsplit(url)
    if parsed.path in ("", "/"):
        parsed = parsed._replace(path="/v1")
    return urlunsplit(parsed).rstrip("/")


def asr_models_endpoint(asr_url: str) -> str:
    return normalize_openai_base(asr_url) + "/models"


def llm_models_endpoint(settings: Settings) -> str:
    endpoint = llm_endpoint(settings)
    if settings.llm_backend == "openai":
        return endpoint[: -len("/chat/completions")] + "/models"
    return endpoint[: -len("/api/chat")] + "/api/tags"


def parse_model_list(payload: Any) -> list[str]:
    """Accept OpenAI ``data[].id``, Ollama ``models[].name``, and llama-server variants."""
    names: list[str] = []
    if isinstance(payload, dict):
        for key in ("data", "models"):
            items = payload.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    name = item.get("id") or item.get("name") or item.get("model")
                else:
                    name = None
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
    return list(dict.fromkeys(names))


def fetch_models(endpoint: str, settings: Settings, timeout: float = 5.0) -> list[str]:
    settings.validate_url(endpoint)
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MeetingError(
            f"模型列表接口返回 HTTP {exc.code}（{safe_endpoint(endpoint)}）；请检查地址和接口类型",
            "MODEL_LIST_HTTP_ERROR",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MeetingError(
            f"无法连接 {safe_endpoint(endpoint)}：{getattr(exc, 'reason', exc)}；请确认服务已启动",
            "MODEL_LIST_UNAVAILABLE",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeetingError(f"{safe_endpoint(endpoint)} 返回的不是 JSON", "MODEL_LIST_INVALID") from exc
    models = parse_model_list(payload)
    if not models:
        raise MeetingError(f"{safe_endpoint(endpoint)} 没有返回任何模型", "MODEL_LIST_EMPTY")
    return models


def list_asr_models(settings: Settings, timeout: float = 5.0) -> list[str]:
    return fetch_models(asr_models_endpoint(settings.asr_url), settings, timeout)


def list_llm_models(settings: Settings, timeout: float = 5.0) -> list[str]:
    return fetch_models(llm_models_endpoint(settings), settings, timeout)


def whisper_models() -> list[tuple[str, bool]]:
    """Return ``(name, downloaded)`` for every openai-whisper model."""
    try:
        import whisper
    except ImportError as exc:
        raise MeetingError(f"openai-whisper 无法加载：{exc}", "WHISPER_NOT_INSTALLED") from exc
    cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "whisper"
    urls = getattr(whisper, "_MODELS", {})
    result = []
    for name in whisper.available_models():
        filename = os.path.basename(urls.get(name, f"{name}.pt"))
        result.append((name, (cache / filename).is_file()))
    return result
