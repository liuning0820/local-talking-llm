"""Per-file transcription adapters and durable transcript writes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .errors import PermanentJobError, RecoverableJobError
from .jobs import run_with_retry
from .store import Store


class Transcriber(Protocol):
    backend: str
    model: str

    def transcribe(self, wav_path: Path) -> str | list[dict[str, Any]]: ...


class WhisperTranscriber:
    backend = "whisper"

    def __init__(self, model_name: str = "base"):
        try:
            import whisper
        except ModuleNotFoundError as exc:
            if exc.name == "whisper":
                raise PermanentJobError(
                    "Whisper is not installed. Install the project dependency "
                    "openai-whisper (it imports as whisper), not the unrelated "
                    "whisper package. Use Python 3.11-3.13, then run "
                    "python -m pip install -r requirements.txt; alternatively "
                    "set MEETING_STT_BACKEND=llama-asr and configure "
                    "MEETING_ASR_URL and MEETING_ASR_MODEL.",
                    "WHISPER_NOT_INSTALLED",
                ) from exc
            raise
        except ImportError as exc:
            raise PermanentJobError(
                "The openai-whisper dependency could not load its torch runtime: "
                f"{exc}. Use Python 3.11-3.13 and reinstall the pinned "
                "requirements, or use the configured llama-asr backend.",
                "WHISPER_RUNTIME_UNAVAILABLE",
            ) from exc
        self.model = model_name
        self._model = whisper.load_model(model_name)

    def transcribe(self, wav_path: Path) -> list[dict[str, Any]]:
        result = self._model.transcribe(str(wav_path), fp16=False)
        segments = result.get("segments") if isinstance(result, dict) else None
        if isinstance(segments, list):
            return [
                {"start_ms": max(0, int(float(item["start"]) * 1000)),
                 "end_ms": max(0, int(float(item["end"]) * 1000)),
                 "text": str(item.get("text", "")).strip()}
                for item in segments if isinstance(item, dict) and float(item.get("end", 0)) >= float(item.get("start", 0))
            ]
        return str(result.get("text", "")).strip() if isinstance(result, dict) else ""


class LlamaASRTranscriber:
    backend = "llama-asr"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.asr_model

    def transcribe(self, wav_path: Path) -> str | list[dict[str, Any]]:
        import requests
        self.settings.require_network_confirmation(self.settings.asr_url, "音频 WAV")
        timeout = float(self.settings.request_timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise PermanentJobError("MEETING_REQUEST_TIMEOUT must be finite and positive", "ASR_TIMEOUT_CONFIG")
        endpoint = self.settings.asr_url.rstrip("/") + "/audio/transcriptions"
        try:
            with wav_path.open("rb") as handle:
                response = requests.post(
                    endpoint,
                    files={"file": (wav_path.name, handle, "audio/wav")},
                    data={"model": self.model, "response_format": "json"},
                    # Keep connect failures short while bounding socket read inactivity.
                    timeout=(min(10.0, timeout), timeout),
                )
        except requests.Timeout as exc:
            raise RecoverableJobError(
                f"ASR request timed out after {timeout:g}s", "ASR_TIMEOUT"
            ) from exc
        except requests.ConnectionError as exc:
            raise RecoverableJobError(
                f"ASR server is unavailable at {endpoint}", "ASR_UNAVAILABLE"
            ) from exc
        except requests.RequestException as exc:
            raise RecoverableJobError(
                f"ASR request failed: {exc}", "ASR_UNAVAILABLE"
            ) from exc
        if response.status_code >= 400:
            if 400 <= response.status_code < 500:
                raise PermanentJobError(f"ASR rejected the request with HTTP {response.status_code}", "ASR_CLIENT_ERROR")
            raise RecoverableJobError(
                f"ASR server returned HTTP {response.status_code}", "ASR_UNAVAILABLE"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RecoverableJobError(
                f"ASR returned HTTP 200 with invalid JSON (content_type={response.headers.get('content-type', 'unknown')})",
                "ASR_INVALID_RESPONSE",
            ) from exc
        return parse_llama_response(payload)


def parse_llama_response(payload: Any) -> str | list[dict[str, Any]]:
    """Normalize llama-server's JSON transcription response."""
    if not isinstance(payload, dict):
        raise RecoverableJobError(
            "ASR returned HTTP 200 with an unexpected JSON shape (expected an object)",
            "ASR_INVALID_RESPONSE",
        )
    segments = payload.get("segments")
    if isinstance(segments, list):
        normalized: list[dict[str, Any]] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                start = max(0, int(float(item.get("start", 0)) * 1000))
                end = max(start, int(float(item.get("end", item.get("start", 0))) * 1000))
            except (TypeError, ValueError):
                continue
            normalized.append({"start_ms": start, "end_ms": end, "text": text})
        if normalized:
            return normalized
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RecoverableJobError(
        "ASR returned HTTP 200 without usable text or segments",
        "ASR_EMPTY_RESPONSE",
    )


def transcriber_from_settings(settings: Settings) -> Transcriber:
    if settings.stt_backend == "llama-asr":
        settings.validate_url(settings.asr_url)
        return LlamaASRTranscriber(settings)
    if settings.stt_backend != "whisper":
        raise PermanentJobError(f"Unsupported STT backend: {settings.stt_backend}", "INVALID_STT_BACKEND")
    return WhisperTranscriber(settings.whisper_model)


def _segments(result: str | list[dict[str, Any]], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result, list):
        output = []
        for item in result:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = chunk["start_ms"] + max(0, int(item.get("start_ms", 0)))
            end = chunk["start_ms"] + max(0, int(item.get("end_ms", item.get("start_ms", 0))))
            output.append({"start_ms": start, "end_ms": max(start, end), "text": text})
        if output:
            return output
    elif isinstance(result, str) and result.strip():
        return [{"start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"], "text": result.strip()}]
    raise RecoverableJobError("ASR returned no usable transcription text", "ASR_EMPTY_RESPONSE")


def _error_text(error: Exception) -> str:
    code = getattr(error, "code", None)
    return f"{code}: {error}" if code else str(error)


def transcribe_meeting(store: Store, meeting_id: str, transcriber: Transcriber) -> tuple[int, int]:
    succeeded = failed = 0
    for chunk in store.chunks(meeting_id):
        if chunk["status"] != "closed":
            continue
        existing = store.transcripts_for_chunk(meeting_id, chunk["sequence"])
        if existing and all(row["status"] == "succeeded" for row in existing):
            succeeded += 1
            continue
        result, attempts, error = run_with_retry(lambda: transcriber.transcribe(store.root / chunk["relative_path"]))
        if error:
            failed += 1
            store.replace_transcripts(meeting_id, chunk["sequence"], [{"start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"], "text": ""}], transcriber.backend, getattr(transcriber, "model", None), attempts, "failed", _error_text(error))
        else:
            succeeded += 1
            store.replace_transcripts(meeting_id, chunk["sequence"], _segments(result, chunk), transcriber.backend, getattr(transcriber, "model", None), attempts)
    return succeeded, failed
