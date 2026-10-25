"""Deterministic no-device/no-model smoke pipeline."""

from __future__ import annotations
import wave
from pathlib import Path
from .config import Settings
from .export import export_meeting
from .service import create_store, process_meeting
from .store import Store
from .summary import SummaryEngine


class FakeTranscriber:
    backend = "fake"
    model = "synthetic"
    def transcribe(self, wav_path: Path) -> str:
        return {"chunk-000001.wav": "我们决定在周五前发布版本。", "chunk-000002.wav": "未解决问题是测试环境何时可用。"}[wav_path.name]


class FakeSummary:
    def __call__(self, _prompt: str, _purpose: str) -> dict:
        return {"summary": "会议决定在周五前发布版本。", "topics": [{"topic": "发布计划"}], "decisions": [{"decision": "周五前发布版本", "evidence_segment_ids": ["synthetic-1"], "start_ms": 0, "end_ms": 1000}], "action_items": [{"action": "发布版本", "owner": "未提及", "due_date": "周五", "status": "未提及", "evidence_segment_ids": ["synthetic-1"], "start_ms": 0, "end_ms": 1000}], "unresolved_questions": ["测试环境何时可用"], "inferences": ["未提及"], "missing_information": ["未提及"]}


def _wav(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes((value.to_bytes(2, "little", signed=True)) * 16000)


def run(settings: Settings) -> dict[str, object]:
    store = create_store(settings)
    meeting = store.create_meeting(title="Synthetic smoke", input_mode="synthetic", device_name="fake")
    audio = store.root / "meetings" / meeting["id"] / "audio"
    for number, value in ((1, 100), (2, 200)):
        path = audio / f"chunk-{number:06d}.wav"; _wav(path, value)
        store.add_audio_chunk(meeting["id"], number, path.relative_to(store.root).as_posix(), (number - 1) * 1000, number * 1000, path.stat().st_size, store.file_hash(path))
    store.update_meeting(meeting["id"], status="stopping", ended_at=__import__("meeting_mvp.store", fromlist=["utc_now"]).utc_now())
    # The fake output has stable IDs after store insertion; adapt it to the persisted IDs.
    class Engine(SummaryEngine):
        def generate(self, segments):
            sid = segments[0]["id"]
            payload = FakeSummary()("", "final"); payload["decisions"][0]["evidence_segment_ids"] = [sid]; payload["action_items"][0]["evidence_segment_ids"] = [sid]
            from .markdown import render_summary
            import hashlib, json
            text = json.dumps(segments, ensure_ascii=False)
            return payload, render_summary(payload), hashlib.sha256(text.encode()).hexdigest()
    process_meeting(store, meeting["id"], settings, FakeTranscriber(), Engine(FakeSummary(), "synthetic"))
    json_path = export_meeting(store, meeting["id"], "json")
    markdown_path = export_meeting(store, meeting["id"], "markdown")
    result = {"meeting_id": meeting["id"], "status": store.get_meeting(meeting["id"])["status"], "chunks": len(store.chunks(meeting["id"])), "transcripts": len(store.transcripts(meeting["id"])), "json_export": str(json_path), "markdown_export": str(markdown_path), "limitation": "Synthetic only; no physical audio device, Whisper, or Ollama was used."}
    store.close()
    return result
