"""Crash recovery for recordings, WAV registration, and processing leases."""

from __future__ import annotations

import re
import wave
from datetime import datetime, timezone

from .paths import safe_relative
from .store import Store, utc_now

_CHUNK_NAME = re.compile(r"chunk-(\d+)\.wav$")


def _stale(heartbeat: str | None, seconds: int) -> bool:
    if not heartbeat:
        return True
    try:
        value = datetime.fromisoformat(heartbeat)
        return (datetime.now(timezone.utc) - value).total_seconds() > seconds
    except ValueError:
        return True


def _recover_deleting(store: Store, meeting_id: str) -> None:
    """Finish a delete whose database transaction was interrupted."""
    meeting_dir = store.root / "meetings" / meeting_id
    try:
        if meeting_dir.exists():
            import shutil
            shutil.rmtree(meeting_dir)
        with store._lock, store.db:
            store.db.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
    except OSError as exc:
        store.update_meeting(meeting_id, error=f"Delete incomplete: {exc}")


def recover(store: Store, stale_after_seconds: int = 30) -> list[str]:
    recovered: list[str] = []
    for meeting in store.list_meetings():
        if meeting["status"] == "deleting":
            _recover_deleting(store, meeting["id"])
            continue
        if meeting["status"] in {"recording", "stopping"} and _stale(meeting.get("last_heartbeat"), stale_after_seconds):
            store.update_meeting(
                meeting["id"], status="interrupted", ended_at=utc_now(),
                error="Recording process stopped before a clean finish",
            )
            recovered.append(meeting["id"])
        validate_closed_audio(store, meeting["id"])
        refreshed = store.get_meeting(meeting["id"]) or meeting
        if refreshed["status"] in {"stopping", "interrupted", "partial", "failed"}:
            store.enqueue_transcription_jobs(meeting["id"])
        # A fresh recording may still have its writer's open temporary WAV.
        # Keep it until the writer has stopped; stale/interrupted meetings are
        # safe to clean up because their process can no longer replace it.
        if refreshed["status"] not in {"recording", "stopping"} or _stale(
            refreshed.get("last_heartbeat"), stale_after_seconds
        ):
            _remove_temporary_audio(store, refreshed["id"])
    store.reset_expired_jobs()
    return recovered


def _remove_temporary_audio(store: Store, meeting_id: str) -> None:
    audio_dir = store.root / "meetings" / meeting_id / "audio"
    if not audio_dir.exists():
        return
    for temporary in audio_dir.glob("*.tmp"):
        try:
            temporary.unlink()
        except OSError:
            # A later recovery can retry after the writer or another process
            # releases the file. Never turn recovery into data loss.
            continue


def validate_closed_audio(store: Store, meeting_id: str) -> list[str]:
    """Register complete WAVs left after replace-before-SQLite crash, idempotently."""
    found: list[str] = []
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        return found
    audio_dir = store.root / "meetings" / meeting_id / "audio"
    if not audio_dir.exists():
        return found
    registered = {row["relative_path"] for row in store.chunks(meeting_id, include_deleted=True)}
    registered_sequences = {row["sequence"] for row in store.chunks(meeting_id, include_deleted=True)}
    rate = int(meeting["sample_rate"])
    channels = int(meeting["channels"])
    width = 2
    for wav_path in sorted(audio_dir.glob("*.wav")):
        relative = safe_relative(store.root, wav_path)
        if relative in registered:
            continue
        match = _CHUNK_NAME.match(wav_path.name)
        if not match:
            continue
        sequence = int(match.group(1))
        if sequence in registered_sequences:
            continue
        try:
            with wave.open(str(wav_path), "rb") as handle:
                frames = handle.getnframes()
                if frames <= 0 or handle.getframerate() != rate or handle.getnchannels() != channels or handle.getsampwidth() != width:
                    continue
                duration = int(frames * 1000 / rate)
            size = wav_path.stat().st_size
            digest = store.file_hash(wav_path)
            previous = store.get_chunk(meeting_id, sequence - 1)
            start = int(previous["end_ms"]) if previous else max(0, (sequence - 1) * duration)
            store.add_audio_chunk(meeting_id, sequence, relative, start, start + duration, size, digest)
            found.append(relative)
            registered.add(relative)
            registered_sequences.add(sequence)
        except (OSError, wave.Error):
            continue
    return found
