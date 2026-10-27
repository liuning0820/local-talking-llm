"""Evidence-validated, bounded map/reduce meeting summaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from .errors import MeetingError
from .llm import LocalLLM, parse_json_response
from .markdown import render_summary
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, summary_prompt
from .store import Store

SUMMARY_LIMIT = 12000
REDUCE_LIMIT = 30000
LIST_FIELDS = (
    "summary",
    "topics",
    "decisions",
    "action_items",
    "unresolved_questions",
    "inferences",
    "missing_information",
)


def transcript_text(segments: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{item['id']}] ({item['start_ms']}-{item['end_ms']}ms) {item['text']}"
        for item in segments if item.get("text")
    )


def _normalize_list_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model scalar/list drift without discarding returned content."""
    result = dict(payload)
    for key in LIST_FIELDS:
        value = result.get(key)
        if value is None or value == []:
            result[key] = ["未提及"]
        elif isinstance(value, list):
            result[key] = value
        elif isinstance(value, (str, int, float, bool, dict)):
            # A scalar is one item; an object remains one item so its content is
            # still validated or rendered instead of being silently discarded.
            result[key] = [value]
        else:
            raise MeetingError(
                f"Summary field '{key}' must be a list or a scalar item, got {type(value).__name__}",
                "INVALID_SUMMARY_SCHEMA",
            )
    return result


INFERENCE_LABEL = "模型推断（需确认）"
_INFERENCE_TEXT_KEYS = ("text", "inference", "content", "statement", "description", "value")


def _normalize_inferences(items: list[Any]) -> list[Any]:
    """Keep inference content, drop empty shells, and guarantee both labels on each item.

    Small models often return ``{"labels": ["模型推断", "需确认"]}`` without any text, or
    plain text without labels. Inferences are only ever rendered under the
    "模型推断（需确认）" heading, so adding the label keeps them clearly unconfirmed.
    """
    result: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            text = next((item[key] for key in _INFERENCE_TEXT_KEYS if isinstance(item.get(key), str) and item[key].strip()), "")
        else:
            text = "" if item is None else str(item)
        text = text.strip()
        if not text or text == "未提及":
            continue
        if "模型推断" not in text or "需确认" not in text:
            text = f"{INFERENCE_LABEL}：{text}"
        result.append(text)
    return result or ["未提及"]


def _defaults(payload: dict[str, Any]) -> dict[str, Any]:
    result = _normalize_list_fields(payload)
    result["inferences"] = _normalize_inferences(result["inferences"])
    if not result.get("summary"):
        result["summary"] = ["未提及"]
    for key in LIST_FIELDS[1:]:
        if not result.get(key):
            result[key] = ["未提及"]
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        result["metadata"] = {"source": "transcript", "version": 1}
    else:
        result["metadata"] = _normalize_metadata(metadata)
    return result


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Absorb harmless model drift such as "version": "1" or "Transcript".

    Values that cannot be coerced are left as-is so validation still rejects them.
    """
    result = dict(metadata)
    result.setdefault("source", "transcript")
    result.setdefault("version", 1)
    source = result["source"]
    if isinstance(source, str):
        result["source"] = source.strip().lower()
    version = result["version"]
    if isinstance(version, float) and version.is_integer():
        result["version"] = int(version)
    elif isinstance(version, str) and version.strip().lstrip("vV").isdigit():
        result["version"] = int(version.strip().lstrip("vV"))
    return result


def _segment_lines(segments: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get("text") or "")
        prefix = f"[{segment['id']}] ({segment['start_ms']}-{segment['end_ms']}ms) "
        if not text:
            continue
        available = max(1, limit - len(prefix))
        for offset in range(0, len(text), available):
            lines.append(prefix + text[offset:offset + available])
    return lines


def _windows_from_lines(lines: list[str], limit: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for original in lines:
        pieces = [original[offset:offset + limit] for offset in range(0, len(original), limit)] or [""]
        for line in pieces:
            if current and size + len(line) + 1 > limit:
                windows.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
    if current:
        windows.append("\n".join(current))
    return windows


def _segment_windows(segments: list[dict[str, Any]], limit: int = SUMMARY_LIMIT) -> list[str]:
    return _windows_from_lines(_segment_lines(segments, limit), limit)


def _valid_evidence(item: dict[str, Any], segment_ids: set[str], ranges: dict[str, tuple[int, int]]) -> None:
    evidence = item.get("evidence_segment_ids")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(value, str) for value in evidence):
        raise MeetingError("Every fact needs a non-empty evidence_segment_ids list", "SUMMARY_EVIDENCE_INVALID")
    if not set(evidence) <= segment_ids:
        raise MeetingError("Summary evidence refers to a segment outside this meeting", "SUMMARY_EVIDENCE_INVALID")
    start, end = item.get("start_ms"), item.get("end_ms")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise MeetingError("Every fact needs a valid start_ms/end_ms range", "SUMMARY_EVIDENCE_INVALID")
    if ranges and not any(start <= ranges[segment][1] and end >= ranges[segment][0] for segment in evidence):
        raise MeetingError("Fact time range does not overlap its transcript evidence", "SUMMARY_EVIDENCE_INVALID")


def validate_summary(payload: dict[str, Any], segment_ids: set[str] | list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the model contract before a summary can become ready."""
    if not isinstance(payload, dict):
        raise MeetingError("LLM summary must be a JSON object", "INVALID_SUMMARY_SCHEMA")
    supplied_keys = set(payload)
    payload = _defaults(payload)
    if supplied_keys and "metadata" not in supplied_keys:
        raise MeetingError("Summary metadata is required", "INVALID_SUMMARY_SCHEMA")
    if isinstance(segment_ids, set):
        known_ids = segment_ids
        ranges: dict[str, tuple[int, int]] = {}
    else:
        known_ids = {str(item["id"]) for item in segment_ids}
        ranges = {str(item["id"]): (int(item["start_ms"]), int(item["end_ms"])) for item in segment_ids}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("source") != "transcript" or not isinstance(metadata.get("version"), int):
        raise MeetingError("Summary metadata must identify transcript source and schema version", "INVALID_SUMMARY_SCHEMA")
    for key in LIST_FIELDS:
        if not isinstance(payload[key], list):
            raise MeetingError(f"Summary field '{key}' must be a list", "INVALID_SUMMARY_SCHEMA")
    for collection in ("decisions", "action_items"):
        for item in payload[collection]:
            if isinstance(item, str):
                if item != "未提及":
                    raise MeetingError("Factual strings must be backed by an evidence object", "SUMMARY_EVIDENCE_INVALID")
                continue
            if not isinstance(item, dict):
                raise MeetingError("Summary facts must be objects or exactly 未提及", "INVALID_SUMMARY_SCHEMA")
            field = "decision" if collection == "decisions" else "action"
            if not str(item.get(field) or "").strip() or item.get(field) == "未提及":
                raise MeetingError("A factual summary object must contain a fact", "INVALID_SUMMARY_SCHEMA")
            _valid_evidence(item, known_ids, ranges)
            if collection == "action_items":
                for name in ("owner", "due_date", "status"):
                    if item.get(name) is None or item.get(name) == "":
                        item[name] = "未提及"
    for item in payload["inferences"]:
        if item == "未提及":
            continue
        text = item.get("text") if isinstance(item, dict) else str(item)
        if not isinstance(text, str) or "模型推断" not in text or "需确认" not in text:
            raise MeetingError("Every inference must be marked 模型推断/需确认", "INVALID_SUMMARY_SCHEMA")
    return payload


def demote_unbacked_facts(payload: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Move decisions/actions without valid transcript evidence into labelled inferences.

    A model claim that cannot be traced to a segment is never shown as a decision or
    action; it is kept under "模型推断（需确认）" instead of failing the whole summary.
    """
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)
    known = {str(item["id"]) for item in segments}
    ranges = {str(item["id"]): (int(item["start_ms"]), int(item["end_ms"])) for item in segments}
    demoted: list[str] = []
    for collection, field, label in (("decisions", "decision", "决策"), ("action_items", "action", "行动项")):
        value = result.get(collection)
        items = value if isinstance(value, list) else ([] if value is None else [value])
        kept: list[Any] = []
        for item in items:
            if item == "未提及":
                kept.append(item)
                continue
            text = str((item.get(field) if isinstance(item, dict) else item) or "").strip()
            if isinstance(item, dict) and text and text != "未提及":
                try:
                    _valid_evidence(item, known, ranges)
                    kept.append(item)
                    continue
                except MeetingError:
                    pass
            if text and text != "未提及":
                demoted.append(f"{INFERENCE_LABEL}：（转写中找不到依据的{label}）{text}")
        result[collection] = kept or ["未提及"]
    if demoted:
        existing = result.get("inferences")
        existing = existing if isinstance(existing, list) else ([] if existing is None else [existing])
        result["inferences"] = [item for item in existing if item != "未提及"] + demoted
    return result


class SummaryEngine:
    def __init__(self, summarizer: Callable[[str, str], dict[str, Any]], model: str = "local"):
        self.summarizer, self.model = summarizer, model

    def _call(self, prompt: str, purpose: str) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(2):
            try:
                result = self.summarizer(prompt, purpose)
                if not isinstance(result, dict):
                    raise MeetingError("LLM response must be a JSON object", "INVALID_LLM_JSON")
                return result
            except Exception as exc:
                last = exc
        assert last is not None
        raise last

    def generate(self, segments: list[dict[str, Any]]) -> tuple[dict[str, Any], str, str]:
        text = transcript_text(segments)
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if len(text) <= SUMMARY_LIMIT:
            raw = self._call(summary_prompt(text), "final")
        else:
            maps = [self._call(summary_prompt(window, "map"), "map") for window in _segment_windows(segments)]
            map_lines = [json.dumps(item, ensure_ascii=False) for item in maps]
            reduce_windows = _windows_from_lines(map_lines, REDUCE_LIMIT)
            reduced = "\n--- MAP WINDOW ---\n".join(reduce_windows)
            raw = self._call(summary_prompt(reduced, "reduce"), "reduce")
        payload = validate_summary(demote_unbacked_facts(raw, segments), segments)
        return payload, render_summary(payload), source_hash


def engine_from_settings(settings) -> SummaryEngine:
    llm = LocalLLM(settings)

    def call(prompt: str, _purpose: str) -> dict[str, Any]:
        return parse_json_response(llm.complete(prompt, SYSTEM_PROMPT))

    return SummaryEngine(call, settings.llm_model)


def summarize_meeting(store: Store, meeting_id: str, engine: SummaryEngine) -> dict[str, Any]:
    segments = store.transcripts(meeting_id)
    failed = [item for item in segments if item["status"] != "succeeded"]
    try:
        payload, markdown, source_hash = engine.generate(segments)
    except Exception as exc:
        source = transcript_text(segments)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        failed_payload = _defaults({"summary": "未提及"})
        failed_payload["missing_information"] = [f"纪要生成失败: {exc}"]
        store.save_summary(meeting_id, failed_payload, None, engine.model, source_hash, PROMPT_VERSION, status="failed", error=str(exc))
        raise
    missing = list(payload.get("missing_information") or [])
    if failed:
        missing.append("转写失败的音频片段: " + ", ".join(str(item["chunk_sequence"]) for item in failed))
    meeting = store.get_meeting(meeting_id) or {}
    try:
        ranges = json.loads(meeting.get("missing_ranges") or "[]")
    except (TypeError, json.JSONDecodeError):
        ranges = []
    if ranges:
        missing.append("录音缺失时间范围: " + ", ".join(f"{item['start_ms']}-{item['end_ms']}ms" for item in ranges))
    if int(meeting.get("dropped_blocks") or 0):
        missing.append(f"录音队列丢失音频块: {meeting['dropped_blocks']} 个")
    if missing != payload.get("missing_information"):
        payload["missing_information"] = missing
        markdown = render_summary(payload)
    store.save_summary(meeting_id, payload, markdown, engine.model, source_hash, PROMPT_VERSION)
    return payload
