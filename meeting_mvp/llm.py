"""Small standard-library HTTP adapters for local Ollama/OpenAI-compatible LLMs."""

from __future__ import annotations
import json
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import Settings
from .errors import MeetingError


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise MeetingError("LLM response must be a JSON object", "INVALID_LLM_JSON")
    return payload


def _without_v1(url: str) -> str:
    parsed = urlsplit(url.rstrip("/"))
    if parsed.path.rstrip("/") == "/v1":
        parsed = parsed._replace(path="")
    return urlunsplit(parsed).rstrip("/")


def llm_endpoint(settings: Settings) -> str:
    """Resolve a configured base URL or full endpoint exactly once."""
    url = settings.llm_url.rstrip("/")
    if settings.llm_backend == "openai":
        if url.endswith("/chat/completions"):
            return url
        return url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
    if url.endswith("/api/chat"):
        return url
    return _without_v1(url) + "/api/chat"


def safe_endpoint(url: str) -> str:
    """Return an endpoint without credentials, query strings, or fragments."""
    parsed = urlsplit(url)
    host = parsed.hostname or "?"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


class LocalLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.validate_url()

    def complete(self, prompt: str, system: str = "") -> str:
        self.settings.require_network_confirmation(self.settings.llm_url, "转写文本和会议内容")
        endpoint = llm_endpoint(self.settings)
        if self.settings.llm_backend == "openai":
            body = {"model": self.settings.llm_model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0}
        else:
            body = {"model": self.settings.llm_model, "stream": False, "format": "json", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                detail = (
                    f"LLM rejected the request with HTTP {exc.code} "
                    f"(endpoint={safe_endpoint(endpoint)}, model={self.settings.llm_model}). "
                    "Check the backend setting, run `ollama list`, pull the model if missing, "
                    "and verify the endpoint path."
                )
                raise MeetingError(detail, "LLM_CLIENT_ERROR") from exc
            raise MeetingError(f"LLM server error at {safe_endpoint(endpoint)}: {exc}", "LLM_TEMPORARY_ERROR") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MeetingError(f"LLM connection failed at {safe_endpoint(endpoint)}: {exc}", "LLM_CONNECTION_ERROR") from exc
        if self.settings.llm_backend == "openai":
            return str(result["choices"][0]["message"]["content"])
        return str(result.get("message", {}).get("content", result.get("response", "")))
