"""Durable meeting lifecycle and single-process job worker."""

from __future__ import annotations

import threading

from .config import Settings
from .errors import MeetingError
from .jobs import run_with_retry
from .store import Store, utc_now
from .summary import engine_from_settings, summarize_meeting
from .transcription import _error_text, transcriber_from_settings


def create_store(settings: Settings) -> Store:
    settings.ensure_dirs()
    return Store(settings.data_dir)


def enqueue_transcription_jobs(store: Store, meeting_id: str) -> None:
    """Create only chunk jobs at the recording boundary; summary waits for them."""
    store.enqueue_transcription_jobs(meeting_id)


def _transcription_done(jobs: list[dict]) -> bool:
    jobs = [job for job in jobs if job["kind"] == "transcribe"]
    return all(job["status"] in {"completed", "failed"} for job in jobs)


def _has_transcription_failures(jobs: list[dict]) -> bool:
    return any(job["kind"] == "transcribe" and job["status"] == "failed" for job in jobs)


def _start_lease_heartbeat(store: Store, job_id: int, lease_seconds: int = 600):
    stop = threading.Event()

    def renew() -> None:
        while not stop.wait(min(30, max(1, lease_seconds // 3))):
            store.renew_job(job_id, lease_seconds)

    thread = threading.Thread(target=renew, name=f"meeting-job-lease-{job_id}", daemon=True)
    thread.start()
    return stop, thread


def _process_transcription_jobs(store: Store, meeting_id: str, transcriber) -> None:
    while True:
        job = store.claim_job(meeting_id, "transcribe")
        if not job:
            return
        sequence = job["chunk_sequence"]
        chunk = store.get_chunk(meeting_id, sequence)
        if not chunk or chunk["status"] != "closed":
            store.fail_job(job["id"], "Audio chunk is missing or not closed")
            continue
        existing = store.transcripts_for_chunk(meeting_id, sequence)
        if existing and all(row["status"] == "succeeded" for row in existing):
            store.complete_job(job["id"])
            continue
        attempts = 1
        lease_stop, lease_thread = _start_lease_heartbeat(store, job["id"])
        try:
            result, attempts, error = run_with_retry(
                lambda: transcriber.transcribe(store.root / chunk["relative_path"])
            )
            if error:
                raise error
            from .transcription import _segments
            store.replace_transcripts(
                meeting_id, sequence, _segments(result, chunk), transcriber.backend,
                getattr(transcriber, "model", None), attempts,
            )
            store.complete_job(job["id"])
        except Exception as exc:
            store.replace_transcripts(
                meeting_id, sequence,
                [{"start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"], "text": ""}],
                getattr(transcriber, "backend", "unknown"), getattr(transcriber, "model", None),
                attempts, "failed", _error_text(exc),
            )
            store.fail_job(job["id"], _error_text(exc))
        finally:
            lease_stop.set()
            lease_thread.join(timeout=1)


def process_meeting(store: Store, meeting_id: str, settings: Settings, transcriber=None, engine=None) -> dict:
    """Consume durable jobs and leave failed work retryable after a restart."""
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
    enqueue_transcription_jobs(store, meeting_id)
    if not store.chunks(meeting_id):
        store.update_meeting(meeting_id, status="failed", error="No audio was recorded", ended_at=meeting.get("ended_at") or utc_now())
        return {"status": "failed", "meeting_id": meeting_id, "reason": "empty_recording"}
    jobs = store.jobs(meeting_id)
    if any(job["kind"] == "transcribe" and job["status"] in {"queued", "running"} for job in jobs):
        store.update_meeting(meeting_id, status="transcribing")
        try:
            transcriber = transcriber or transcriber_from_settings(settings)
        except Exception as exc:
            message = (
                f"{exc} Audio is retained and transcription jobs are failed for retry. "
                f"After fixing the ASR setup, run: python -m meeting_mvp retry {meeting_id}"
            )
            for job in store.jobs(meeting_id):
                if job["kind"] == "transcribe" and job["status"] in {"queued", "running"}:
                    store.fail_job(job["id"], message)
            store.update_meeting(
                meeting_id, status="partial", ended_at=meeting.get("ended_at") or utc_now(), error=message
            )
            if isinstance(exc, MeetingError):
                raise MeetingError(message, exc.code) from exc
            raise MeetingError(message, "TRANSCRIPTION_SETUP_FAILED") from exc
        _process_transcription_jobs(store, meeting_id, transcriber)
    jobs = store.jobs(meeting_id)
    if not _transcription_done(jobs):
        running = [job for job in jobs if job["kind"] == "transcribe" and job["status"] == "running"]
        reason = f"Transcription jobs are still running ({len(running)}); retry will reclaim abandoned leases"
        return {"status": "transcribing", "meeting_id": meeting_id, "reason": reason}

    transcription_failures = _has_transcription_failures(jobs)
    transcripts = store.transcripts(meeting_id)
    if transcription_failures and not any(item["status"] == "succeeded" and item.get("text") for item in transcripts):
        message = next(
            (job.get("error") for job in jobs if job["kind"] == "transcribe" and job["status"] == "failed" and job.get("error")),
            "Transcription failed; audio is retained and can be retried",
        )
        store.update_meeting(
            meeting_id, status="partial", ended_at=meeting.get("ended_at") or utc_now(), error=message,
        )
        return {"status": "partial", "meeting_id": meeting_id, "reason": message}

    summary_job = next((job for job in jobs if job["kind"] == "summarize"), None)
    if summary_job is None:
        store.enqueue_job(meeting_id, "summarize")
        summary_job = next(job for job in store.jobs(meeting_id) if job["kind"] == "summarize")
    if summary_job["status"] == "failed":
        error = summary_job.get("error") or "Summary job failed without an error"
        store.update_meeting(
            meeting_id, status="partial", ended_at=meeting.get("ended_at") or utc_now(), error=error
        )
        raise MeetingError(f"Summary failed: {error}; audio and transcripts are retained for retry", "SUMMARY_FAILED")
    if summary_job["status"] == "completed":
        summary = store.summary(meeting_id)
        return summary["payload"] if summary else {"status": "completed"}

    store.update_meeting(meeting_id, status="summarizing")
    claimed = store.claim_job(meeting_id, "summarize")
    if not claimed:
        return {"status": "summarizing", "meeting_id": meeting_id}
    try:
        engine = engine or engine_from_settings(settings)
        payload = summarize_meeting(store, meeting_id, engine)
        store.complete_job(claimed["id"])
    except Exception as exc:
        store.fail_job(claimed["id"], str(exc))
        status = "partial" if _has_transcription_failures(store.jobs(meeting_id)) else "failed"
        store.update_meeting(meeting_id, status=status, ended_at=meeting.get("ended_at") or utc_now(), error=str(exc))
        raise MeetingError(f"Summary generation failed: {exc}", "SUMMARY_FAILED") from exc

    jobs = store.jobs(meeting_id)
    partial = _has_transcription_failures(jobs) or bool((store.get_meeting(meeting_id) or {}).get("missing_ranges") not in (None, "", "[]"))
    store.update_meeting(
        meeting_id, status="partial" if partial else "completed",
        ended_at=meeting.get("ended_at") or utc_now(),
        error=("Some audio chunks could not be transcribed" if _has_transcription_failures(jobs) else None),
    )
    return payload
