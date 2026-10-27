"""Step-by-step meeting workflow (record -> ASR -> LLM) used by the Streamlit UI.

Each step can be run and re-run independently with its own settings. Nothing in
this module imports Streamlit, so it can be tested and reused directly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .audio import DeviceInfo, probe_source, record_meeting, source_for
from .config import Settings
from .errors import MeetingError
from .service import _process_transcription_jobs, run_summary_job
from .store import Store, utc_now
from .transcription import WhisperTranscriber, _error_text, transcriber_from_settings

Progress = Callable[[str], None]

_WHISPER_CACHE: dict[str, WhisperTranscriber] = {}
_WHISPER_LOCK = threading.Lock()


def error_text(exc: BaseException) -> str:
    return _error_text(exc) if isinstance(exc, Exception) else str(exc)


class BackgroundTask:
    """Runs one step in a daemon thread so page reruns never interrupt it."""

    def __init__(self, kind: str, meeting_id: str, target: Callable[["BackgroundTask"], Any]):
        self.kind, self.meeting_id = kind, meeting_id
        self.progress = ""
        self.error: str | None = None
        self.result: Any = None
        self.started = time.monotonic()
        self.finished: float | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, name=f"meeting-{kind}", daemon=True)

    def start(self) -> "BackgroundTask":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.result = self._target(self)
        except Exception as exc:  # surfaced to the UI instead of killing the thread silently
            self.error = error_text(exc)
        finally:
            self.finished = time.monotonic()

    def report(self, message: str) -> None:
        self.progress = message

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started


def recording_finished(meeting: dict[str, Any]) -> bool:
    """True once the capture loop has exited (``ended_at`` is written on exit)."""
    if meeting["status"] == "recording":
        return False
    return not (meeting["status"] == "stopping" and not meeting.get("ended_at"))


def start_recording(store: Store, settings: Settings, device: DeviceInfo, title: str | None = None) -> tuple[dict[str, Any], BackgroundTask]:
    """Probe the device first (no meeting is created if it fails), then record in the background."""
    probe_source(source_for(device, settings), settings)
    meeting = store.create_meeting(
        title=(title or "").strip() or None, input_mode=device.source, device_id=device.id,
        device_name=device.name, sample_rate=settings.sample_rate, channels=settings.channels,
    )
    meeting_id = meeting["id"]

    def target(_task: BackgroundTask) -> int:
        try:
            return record_meeting(store, meeting_id, source_for(device, settings), settings)
        finally:
            # The CLI processes a meeting right after recording; the UI waits for the user.
            # Leave "stopping" so the next recording can use the one-active-meeting slot.
            current = store.get_meeting(meeting_id)
            if current and current["status"] == "stopping":
                store.update_meeting(
                    meeting_id, status="partial", ended_at=current.get("ended_at") or utc_now(),
                    error="录音已结束，等待第 ② 步转写",
                )

    return meeting, BackgroundTask("record", meeting_id, target).start()


def make_transcriber(settings: Settings):
    """Whisper models are loaded once per model name; HTTP transcribers are cheap."""
    if settings.stt_backend != "whisper":
        return transcriber_from_settings(settings)
    with _WHISPER_LOCK:
        cached = _WHISPER_CACHE.get(settings.whisper_model)
        if cached is None:
            cached = WhisperTranscriber(settings.whisper_model)
            _WHISPER_CACHE[settings.whisper_model] = cached
        return cached


def transcribe_step(
    store: Store, meeting_id: str, settings: Settings, *, rerun_all: bool = False,
    transcriber=None, progress: Progress | None = None,
) -> dict[str, int]:
    """Step 2: transcribe pending (or, with ``rerun_all``, every) audio chunk."""
    report = progress or (lambda _message: None)
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
    if not recording_finished(meeting):
        raise MeetingError("录音尚未结束，请先停止录音", "MEETING_ACTIVE")
    chunks = store.chunks(meeting_id)
    if not chunks:
        raise MeetingError("这场会议没有可转写的录音片段", "NO_AUDIO")
    store.enqueue_transcription_jobs(meeting_id)
    store.requeue_jobs(meeting_id, "transcribe", include_completed=rerun_all)
    if rerun_all:
        store.delete_transcripts(meeting_id)
    queued = [job for job in store.jobs(meeting_id) if job["kind"] == "transcribe" and job["status"] == "queued"]
    if not queued:
        report("所有片段都已转写；勾选“重新转写全部片段”可用新模型重跑")
        return {"chunks": len(chunks), "transcribed": 0, "failed": 0, "segments": len(store.transcripts(meeting_id))}
    ended_at = meeting.get("ended_at") or utc_now()
    store.update_meeting(meeting_id, status="transcribing")
    try:
        report(f"正在加载 ASR 模型（{settings.whisper_model if settings.stt_backend == 'whisper' else settings.asr_model or settings.asr_url}）…")
        transcriber = transcriber or make_transcriber(settings)
    except Exception as exc:
        message = error_text(exc)
        for job in queued:
            store.fail_job(job["id"], message)
        store.update_meeting(meeting_id, status="partial", ended_at=ended_at, error=f"ASR 初始化失败: {message}")
        raise

    done = 0

    def on_chunk(sequence: int) -> None:
        nonlocal done
        done += 1
        report(f"正在转写第 {done}/{len(queued)} 段（片段 #{sequence}）…")

    try:
        _process_transcription_jobs(store, meeting_id, transcriber, on_chunk)
    except Exception as exc:
        store.update_meeting(meeting_id, status="partial", ended_at=ended_at, error=f"转写中断: {error_text(exc)}")
        raise
    jobs = [job for job in store.jobs(meeting_id) if job["kind"] == "transcribe"]
    failed = [job for job in jobs if job["status"] == "failed"]
    transcripts = store.transcripts(meeting_id)
    usable = any(item["status"] == "succeeded" and item.get("text") for item in transcripts)
    if failed and not usable:
        error = failed[0].get("error") or "所有片段转写失败"
    elif failed:
        error = f"{len(failed)} 个片段转写失败；可重试转写，或直接生成纪要"
    else:
        error = "已转写，尚未生成纪要"
    # There is no dedicated "transcribed" status; partial + message means "waiting for step 3".
    store.update_meeting(meeting_id, status="partial", ended_at=ended_at, error=error)
    if failed and not usable:
        raise MeetingError(error, "TRANSCRIPTION_FAILED")
    return {"chunks": len(chunks), "transcribed": len(queued) - len(failed), "failed": len(failed), "segments": len(transcripts)}


def summarize_step(store: Store, meeting_id: str, settings: Settings, *, engine=None, progress: Progress | None = None) -> dict[str, Any]:
    """Step 3: (re)generate minutes from the stored transcript with the chosen LLM."""
    report = progress or (lambda _message: None)
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
    if not recording_finished(meeting):
        raise MeetingError("录音尚未结束，请先停止录音", "MEETING_ACTIVE")
    pending = [job for job in store.jobs(meeting_id) if job["kind"] == "transcribe" and job["status"] in {"queued", "running"}]
    if pending:
        raise MeetingError(f"还有 {len(pending)} 个片段未完成转写，请先完成第 2 步", "TRANSCRIPTION_PENDING")
    if not any(item["status"] == "succeeded" and item.get("text") for item in store.transcripts(meeting_id)):
        raise MeetingError("没有可用的转写文本，请先完成第 2 步", "NO_TRANSCRIPT")
    if meeting["status"] == "stopping":
        store.update_meeting(meeting_id, status="partial")
    store.enqueue_job(meeting_id, "summarize")
    store.requeue_jobs(meeting_id, "summarize", include_completed=True)
    report(f"正在调用 LLM（{settings.llm_model}）生成纪要…")
    return run_summary_job(store, meeting_id, settings, engine)
