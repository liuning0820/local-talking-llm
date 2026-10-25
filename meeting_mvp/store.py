"""SQLite persistence for meetings, chunks, jobs, transcripts, and summaries."""

from __future__ import annotations

import hashlib
import json
import platform as platform_module
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import MeetingError
from .paths import resolve_relative, safe_relative


MEETING_STATUSES = {
    "recording", "stopping", "interrupted", "transcribing", "summarizing",
    "completed", "partial", "failed", "deleting",
}

_STATUS_TRANSITIONS = {
    "recording": {"recording", "stopping", "interrupted", "partial", "failed"},
    "stopping": {"stopping", "interrupted", "partial", "failed", "transcribing"},
    "interrupted": {"interrupted", "transcribing", "summarizing", "partial", "deleting"},
    "partial": {"partial", "transcribing", "summarizing", "completed", "failed", "deleting"},
    "transcribing": {"transcribing", "summarizing", "partial", "failed"},
    "summarizing": {"summarizing", "completed", "partial", "failed"},
    "completed": {"completed", "deleting"},
    "failed": {"failed", "transcribing", "summarizing", "partial", "deleting"},
    "deleting": {"deleting"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _valid_transition(current: str, target: str) -> bool:
    return target in _STATUS_TRANSITIONS.get(current, set())


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Store:
    """Thread-safe local SQLite store. The database never contains audio bytes."""

    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meetings").mkdir(exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        self.db_path = self.root / "meetings.db"
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _migrate(self) -> None:
        with self._lock, self.db:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meetings (
                  id TEXT PRIMARY KEY, title TEXT, started_at TEXT NOT NULL,
                  ended_at TEXT, status TEXT NOT NULL CHECK(status IN ('recording','stopping','interrupted','transcribing','summarizing','completed','partial','failed','deleting')), input_mode TEXT NOT NULL,
                  device_id TEXT, device_name TEXT, platform TEXT NOT NULL,
                  sample_rate INTEGER NOT NULL CHECK(sample_rate > 0), channels INTEGER NOT NULL CHECK(channels > 0),
                  stop_requested INTEGER NOT NULL DEFAULT 0 CHECK(stop_requested IN (0,1)),
                  last_heartbeat TEXT, written_bytes INTEGER NOT NULL DEFAULT 0 CHECK(written_bytes >= 0),
                  dropped_blocks INTEGER NOT NULL DEFAULT 0 CHECK(dropped_blocks >= 0),
                  missing_ranges TEXT NOT NULL DEFAULT '[]', error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_meeting
                  ON meetings(status) WHERE status IN ('recording','stopping');
                CREATE INDEX IF NOT EXISTS meetings_started ON meetings(started_at DESC);
                CREATE TABLE IF NOT EXISTS audio_chunks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL, relative_path TEXT NOT NULL,
                  start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL,
                  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0), sha256 TEXT NOT NULL CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9A-Fa-f]*'),
                  status TEXT NOT NULL CHECK(status IN ('closed','deleted','failed','partial')), error TEXT,
                  CHECK(sequence > 0), CHECK(start_ms >= 0), CHECK(end_ms >= start_ms),
                  UNIQUE(meeting_id, sequence), UNIQUE(meeting_id, relative_path),
                  FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS transcript_segments (
                  id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL,
                  chunk_sequence INTEGER NOT NULL, start_ms INTEGER NOT NULL CHECK(start_ms >= 0),
                  end_ms INTEGER NOT NULL CHECK(end_ms >= start_ms), text TEXT NOT NULL, backend TEXT NOT NULL,
                  model TEXT, attempts INTEGER NOT NULL DEFAULT 1 CHECK(attempts > 0),
                  status TEXT NOT NULL CHECK(status IN ('succeeded','failed')), error TEXT,
                  FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS transcript_segments_order
                  ON transcript_segments(meeting_id, start_ms, chunk_sequence);
                CREATE TABLE IF NOT EXISTS processing_jobs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL,
                  kind TEXT NOT NULL, chunk_sequence INTEGER, status TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0, lease_until TEXT,
                  error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                  FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS processing_jobs_ready
                  ON processing_jobs(status, lease_until, id);
                CREATE TABLE IF NOT EXISTS summary_versions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL,
                  version INTEGER NOT NULL, model TEXT, prompt_version TEXT NOT NULL,
                  source_hash TEXT NOT NULL, json_text TEXT NOT NULL, markdown TEXT,
                  status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
                  UNIQUE(meeting_id, version),
                  FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(meetings)")}
            if "missing_ranges" not in columns:
                self.db.execute("ALTER TABLE meetings ADD COLUMN missing_ranges TEXT NOT NULL DEFAULT '[]'")
            audio_columns = {row[1] for row in self.db.execute("PRAGMA table_info(audio_chunks)")}
            if "status" not in audio_columns:
                self.db.execute("ALTER TABLE audio_chunks ADD COLUMN status TEXT NOT NULL DEFAULT 'closed'")
            if "error" not in audio_columns:
                self.db.execute("ALTER TABLE audio_chunks ADD COLUMN error TEXT")
            self.db.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS meetings_status_guard
                BEFORE INSERT ON meetings
                WHEN NEW.status NOT IN ('recording','stopping','interrupted','transcribing','summarizing','completed','partial','failed','deleting')
                BEGIN SELECT RAISE(ABORT, 'invalid meeting status'); END;
                CREATE TRIGGER IF NOT EXISTS meetings_status_update_guard
                BEFORE UPDATE OF status ON meetings
                WHEN NEW.status NOT IN ('recording','stopping','interrupted','transcribing','summarizing','completed','partial','failed','deleting')
                BEGIN SELECT RAISE(ABORT, 'invalid meeting status'); END;
                CREATE TRIGGER IF NOT EXISTS audio_chunk_metadata_guard
                BEFORE INSERT ON audio_chunks
                WHEN NEW.sequence <= 0 OR NEW.start_ms < 0 OR NEW.end_ms < NEW.start_ms OR NEW.size_bytes < 0
                  OR length(NEW.sha256) <> 64 OR NEW.sha256 GLOB '*[^0-9A-Fa-f]*'
                  OR NEW.status NOT IN ('closed','deleted','failed','partial')
                BEGIN SELECT RAISE(ABORT, 'invalid audio chunk metadata'); END;
                CREATE TRIGGER IF NOT EXISTS audio_chunk_metadata_update_guard
                BEFORE UPDATE ON audio_chunks
                WHEN NEW.sequence <= 0 OR NEW.start_ms < 0 OR NEW.end_ms < NEW.start_ms OR NEW.size_bytes < 0
                  OR length(NEW.sha256) <> 64 OR NEW.sha256 GLOB '*[^0-9A-Fa-f]*'
                  OR NEW.status NOT IN ('closed','deleted','failed','partial')
                BEGIN SELECT RAISE(ABORT, 'invalid audio chunk metadata'); END;
                """)
            self.db.execute("PRAGMA user_version=4")

    def create_meeting(
        self, *, title: str | None = None, input_mode: str = "microphone",
        device_id: str | None = None, device_name: str | None = None,
        sample_rate: int = 16000, channels: int = 1,
        platform: str | None = None,
    ) -> dict[str, Any]:
        if title is not None and len(title) > 200:
            raise MeetingError("Meeting title is limited to 200 characters", "TITLE_TOO_LONG")
        if input_mode not in {"microphone", "system", "synthetic"}:
            raise MeetingError("Invalid meeting input mode", "INVALID_AUDIO_SOURCE")
        if sample_rate <= 0 or channels <= 0:
            raise MeetingError("Sample rate and channels must be positive", "INVALID_AUDIO_FORMAT")
        meeting_id = str(uuid.uuid4())
        started = utc_now()
        platform = platform or platform_module.system()
        with self._lock, self.db:
            try:
                self.db.execute(
                    "INSERT INTO meetings(id,title,started_at,status,input_mode,device_id,device_name,platform,sample_rate,channels,last_heartbeat) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (meeting_id, title, started, "recording", input_mode, device_id,
                     device_name, platform, sample_rate, channels, started),
                )
            except sqlite3.IntegrityError as exc:
                raise MeetingError(
                    "Another meeting is recording; stop it before starting a new one",
                    "MEETING_ALREADY_ACTIVE",
                ) from exc
        (self.root / "meetings" / meeting_id / "audio").mkdir(parents=True)
        (self.root / "meetings" / meeting_id / "exports").mkdir()
        return self.get_meeting(meeting_id)  # type: ignore[return-value]

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        with self._lock:
            return _dict(self.db.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone())

    def list_meetings(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM meetings"
        args: tuple[Any, ...] = ()
        if status:
            query += " WHERE status=?"
            args = (status,)
        query += " ORDER BY started_at DESC"
        with self._lock:
            return [dict(row) for row in self.db.execute(query, args).fetchall()]

    def update_meeting(self, meeting_id: str, **fields: Any) -> None:
        allowed = {
            "status", "ended_at", "stop_requested", "last_heartbeat", "written_bytes",
            "dropped_blocks", "missing_ranges", "error",
        }
        if not fields or not set(fields) <= allowed:
            raise MeetingError("Invalid meeting update", "INVALID_MEETING_UPDATE")
        if "status" in fields:
            status = str(fields["status"])
            if status not in MEETING_STATUSES:
                raise MeetingError(f"Invalid meeting status: {status}", "INVALID_MEETING_STATUS")
            current = self.get_meeting(meeting_id)
            if not current:
                raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
            if not _valid_transition(current["status"], status):
                raise MeetingError(
                    f"Cannot change meeting status from {current['status']} to {status}",
                    "INVALID_MEETING_TRANSITION",
                )
        for name in ("written_bytes", "dropped_blocks"):
            if name in fields and int(fields[name]) < 0:
                raise MeetingError(f"{name} cannot be negative", "INVALID_MEETING_COUNTER")
        if "stop_requested" in fields and fields["stop_requested"] not in (0, 1, False, True):
            raise MeetingError("stop_requested must be 0 or 1", "INVALID_MEETING_UPDATE")
        if "missing_ranges" in fields:
            try:
                ranges = json.loads(fields["missing_ranges"])
                if not isinstance(ranges, list):
                    raise ValueError
                for item in ranges:
                    if int(item["start_ms"]) < 0 or int(item["end_ms"]) <= int(item["start_ms"]):
                        raise ValueError
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise MeetingError("missing_ranges must contain valid time ranges", "INVALID_MEETING_UPDATE") from exc
        sets = ", ".join(f"{key}=?" for key in fields)
        with self._lock, self.db:
            self.db.execute(f"UPDATE meetings SET {sets} WHERE id=?", (*fields.values(), meeting_id))

    def request_stop(self, meeting_id: str) -> None:
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
        if meeting["status"] in {"completed", "partial", "failed", "interrupted", "deleting"}:
            raise MeetingError("Meeting is no longer recording", "MEETING_NOT_ACTIVE")
        self.update_meeting(meeting_id, stop_requested=1, status="stopping")

    def heartbeat(self, meeting_id: str, written_bytes: int | None = None,
                  dropped_blocks: int | None = None) -> None:
        values: dict[str, Any] = {"last_heartbeat": utc_now()}
        if written_bytes is not None:
            values["written_bytes"] = written_bytes
        if dropped_blocks is not None:
            values["dropped_blocks"] = dropped_blocks
        self.update_meeting(meeting_id, **values)

    def note_gap(self, meeting_id: str, start_ms: int, end_ms: int, reason: str = "audio gap") -> None:
        if end_ms <= start_ms:
            return
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
        try:
            ranges = json.loads(meeting.get("missing_ranges") or "[]")
        except (TypeError, json.JSONDecodeError):
            ranges = []
        ranges.append({"start_ms": int(start_ms), "end_ms": int(end_ms), "reason": reason})
        self.update_meeting(meeting_id, missing_ranges=json.dumps(ranges, ensure_ascii=False), error="Audio contains missing ranges")

    def mark_partial(self, meeting_id: str, error: str | None = None) -> None:
        meeting = self.get_meeting(meeting_id)
        if meeting and meeting["status"] not in {"recording", "stopping", "deleting"}:
            self.update_meeting(meeting_id, status="partial", error=error or meeting.get("error"))
        elif meeting and error:
            self.update_meeting(meeting_id, error=error)

    def add_audio_chunk(
        self, meeting_id: str, sequence: int, relative_path: str, start_ms: int,
        end_ms: int, size_bytes: int, sha256: str, status: str = "closed",
        error: str | None = None,
    ) -> None:
        resolve_relative(self.root, relative_path)
        if sequence <= 0 or start_ms < 0 or end_ms < start_ms or size_bytes < 0:
            raise MeetingError("Invalid audio chunk metadata", "INVALID_AUDIO_CHUNK")
        if len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256):
            if sha256 == "hash" and resolve_relative(self.root, relative_path).is_file():
                sha256 = self.file_hash(resolve_relative(self.root, relative_path))
            else:
                raise MeetingError("Audio chunk SHA-256 must contain 64 hexadecimal characters", "INVALID_AUDIO_HASH")
        if status not in {"closed", "deleted", "failed", "partial"}:
            raise MeetingError("Invalid audio chunk status", "INVALID_AUDIO_CHUNK_STATUS")
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
        if end_ms == start_ms and status == "closed":
            raise MeetingError("Closed audio chunks must contain audio", "INVALID_AUDIO_CHUNK")
        with self._lock, self.db:
            try:
                self.db.execute(
                    "INSERT INTO audio_chunks(meeting_id,sequence,relative_path,start_ms,end_ms,size_bytes,sha256,status,error) VALUES(?,?,?,?,?,?,?,?,?)",
                    (meeting_id, sequence, relative_path, start_ms, end_ms, size_bytes, sha256.lower(), status, error),
                )
            except sqlite3.IntegrityError as exc:
                raise MeetingError("Audio chunk sequence or path already exists", "DUPLICATE_AUDIO_CHUNK") from exc

    def get_chunk(self, meeting_id: str, sequence: int) -> dict[str, Any] | None:
        with self._lock:
            return _dict(self.db.execute("SELECT * FROM audio_chunks WHERE meeting_id=? AND sequence=?", (meeting_id, sequence)).fetchone())

    def chunks(self, meeting_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM audio_chunks WHERE meeting_id=?"
        if not include_deleted:
            query += " AND status != 'deleted'"
        query += " ORDER BY sequence"
        with self._lock:
            return [dict(row) for row in self.db.execute(query, (meeting_id,)).fetchall()]

    def add_transcript(
        self, meeting_id: str, chunk_sequence: int, start_ms: int, end_ms: int,
        text: str, backend: str, model: str | None, attempts: int = 1,
        status: str = "succeeded", error: str | None = None,
        segment_id: str | None = None,
    ) -> str:
        if start_ms < 0 or end_ms < start_ms or attempts <= 0:
            raise MeetingError("Invalid transcript segment metadata", "INVALID_TRANSCRIPT_SEGMENT")
        if status not in {"succeeded", "failed"}:
            raise MeetingError("Invalid transcript status", "INVALID_TRANSCRIPT_STATUS")
        segment_id = segment_id or str(uuid.uuid4())
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO transcript_segments(id,meeting_id,chunk_sequence,start_ms,end_ms,text,backend,model,attempts,status,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (segment_id, meeting_id, chunk_sequence, start_ms, end_ms, text, backend,
                 model, attempts, status, error),
            )
        return segment_id

    def replace_transcripts(
        self, meeting_id: str, chunk_sequence: int,
        segments: list[dict[str, Any]], backend: str, model: str | None,
        attempts: int, status: str = "succeeded", error: str | None = None,
    ) -> None:
        """Replace one chunk's result atomically, allowing many timed segments."""
        with self._lock, self.db:
            self.db.execute(
                "DELETE FROM transcript_segments WHERE meeting_id=? AND chunk_sequence=?",
                (meeting_id, chunk_sequence),
            )
            for segment in segments:
                self.add_transcript(
                    meeting_id, chunk_sequence, int(segment["start_ms"]),
                    int(segment["end_ms"]), str(segment.get("text") or ""),
                    backend, model, attempts, status, error,
                )

    def transcripts_for_chunk(self, meeting_id: str, sequence: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.db.execute(
                "SELECT * FROM transcript_segments WHERE meeting_id=? AND chunk_sequence=? ORDER BY start_ms, id",
                (meeting_id, sequence),
            ).fetchall()]

    def get_transcript_for_chunk(self, meeting_id: str, sequence: int) -> dict[str, Any] | None:
        rows = self.transcripts_for_chunk(meeting_id, sequence)
        return rows[0] if rows else None

    def transcripts(self, meeting_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM transcript_segments WHERE meeting_id=? ORDER BY start_ms,chunk_sequence", (meeting_id,)).fetchall()]

    def enqueue_job(self, meeting_id: str, kind: str, chunk_sequence: int | None = None) -> None:
        now = utc_now()
        with self._lock, self.db:
            exists = self.db.execute(
                "SELECT id FROM processing_jobs WHERE meeting_id=? AND kind=? AND (chunk_sequence=? OR (chunk_sequence IS NULL AND ? IS NULL))",
                (meeting_id, kind, chunk_sequence, chunk_sequence),
            ).fetchone()
            if exists:
                return
            self.db.execute(
                "INSERT INTO processing_jobs(meeting_id,kind,chunk_sequence,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (meeting_id, kind, chunk_sequence, "queued", now, now),
            )

    def enqueue_transcription_jobs(self, meeting_id: str) -> int:
        chunks = self.chunks(meeting_id)
        with self._lock, self.db:
            now = utc_now()
            added = 0
            for chunk in chunks:
                if chunk["status"] != "closed":
                    continue
                exists = self.db.execute(
                    "SELECT id FROM processing_jobs WHERE meeting_id=? AND kind='transcribe' AND chunk_sequence=?",
                    (meeting_id, chunk["sequence"]),
                ).fetchone()
                if not exists:
                    self.db.execute(
                        "INSERT INTO processing_jobs(meeting_id,kind,chunk_sequence,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                        (meeting_id, "transcribe", chunk["sequence"], "queued", now, now),
                    )
                    added += 1
            return added

    def jobs(self, meeting_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM processing_jobs WHERE meeting_id=? ORDER BY id", (meeting_id,)).fetchall()]

    def claim_job(self, meeting_id: str | None = None, kind: str | None = None,
                  lease_seconds: int = 600) -> dict[str, Any] | None:
        now = utc_now()
        lease = datetime.now(timezone.utc).timestamp() + lease_seconds
        lease_until = datetime.fromtimestamp(lease, timezone.utc).isoformat(timespec="milliseconds")
        with self._lock, self.db:
            where = "(status='queued' OR (status='running' AND lease_until < ?))"
            args: list[Any] = [now]
            if meeting_id:
                where += " AND meeting_id=?"; args.append(meeting_id)
            if kind:
                where += " AND kind=?"; args.append(kind)
            row = self.db.execute(f"SELECT * FROM processing_jobs WHERE {where} ORDER BY id LIMIT 1", args).fetchone()
            if not row:
                return None
            self.db.execute(
                "UPDATE processing_jobs SET status='running', attempts=attempts+1, lease_until=?, updated_at=?, error=NULL WHERE id=?",
                (lease_until, now, row["id"]),
            )
            updated = self.db.execute("SELECT * FROM processing_jobs WHERE id=?", (row["id"],)).fetchone()
            return _dict(updated)

    def renew_job(self, job_id: int, lease_seconds: int = 600) -> None:
        lease = datetime.now(timezone.utc).timestamp() + lease_seconds
        lease_until = datetime.fromtimestamp(lease, timezone.utc).isoformat(timespec="milliseconds")
        with self._lock, self.db:
            self.db.execute("UPDATE processing_jobs SET lease_until=?,updated_at=? WHERE id=? AND status='running'", (lease_until, utc_now(), job_id))

    def complete_job(self, job_id: int) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE processing_jobs SET status='completed',lease_until=NULL,updated_at=? WHERE id=?", (utc_now(), job_id))

    def fail_job(self, job_id: int, error: str, retryable: bool = False) -> None:
        with self._lock, self.db:
            self.db.execute(
                "UPDATE processing_jobs SET status=?,lease_until=NULL,error=?,updated_at=? WHERE id=?",
                ("queued" if retryable else "failed", error, utc_now(), job_id),
            )

    def reset_expired_jobs(self) -> int:
        with self._lock, self.db:
            cur = self.db.execute(
                "UPDATE processing_jobs SET status='queued',lease_until=NULL,updated_at=? WHERE status='running' AND lease_until < ?",
                (utc_now(), utc_now()),
            )
            return cur.rowcount

    def retry_jobs(self, meeting_id: str) -> int:
        """Requeue failed work and only expired abandoned leases."""
        now = utc_now()
        with self._lock, self.db:
            active = self.db.execute(
                "SELECT id, kind, lease_until FROM processing_jobs "
                "WHERE meeting_id=? AND status='running' AND lease_until >= ? LIMIT 1",
                (meeting_id, now),
            ).fetchone()
            if active:
                raise MeetingError(
                    f"{active['kind']} job is active in another process until {active['lease_until']}; "
                    "wait for it to finish or retry after its lease expires",
                    "JOB_ACTIVE",
                )
            cur = self.db.execute(
                "UPDATE processing_jobs SET status='queued',lease_until=NULL,error=NULL,updated_at=? "
                "WHERE meeting_id=? AND status IN ('failed','running')",
                (now, meeting_id),
            )
            return cur.rowcount

    def save_summary(
        self, meeting_id: str, payload: dict[str, Any], markdown: str | None,
        model: str, source_hash: str, prompt_version: str = "1",
        status: str = "ready", error: str | None = None,
    ) -> int:
        with self._lock, self.db:
            version = self.db.execute("SELECT COALESCE(MAX(version),0)+1 FROM summary_versions WHERE meeting_id=?", (meeting_id,)).fetchone()[0]
            cur = self.db.execute(
                "INSERT INTO summary_versions(meeting_id,version,model,prompt_version,source_hash,json_text,markdown,status,error,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (meeting_id, version, model, prompt_version, source_hash,
                 json.dumps(payload, ensure_ascii=False), markdown, status, error, utc_now()),
            )
            return int(cur.lastrowid)

    def summary(self, meeting_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM summary_versions WHERE meeting_id=? ORDER BY version DESC LIMIT 1", (meeting_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            item["payload"] = json.loads(item.pop("json_text"))
            return item

    def delete_audio(self, meeting_id: str) -> int:
        count = 0
        for chunk in self.chunks(meeting_id, include_deleted=True):
            path = resolve_relative(self.root, chunk["relative_path"])
            if path.exists():
                path.unlink()
                count += 1
        with self._lock, self.db:
            self.db.execute("UPDATE audio_chunks SET status='deleted' WHERE meeting_id=?", (meeting_id,))
        return count

    def delete_meeting(self, meeting_id: str, audio_only: bool = False) -> None:
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
        if meeting["status"] in {"recording", "stopping", "transcribing", "summarizing"}:
            raise MeetingError("Stop active recording and processing before deleting it", "MEETING_ACTIVE")
        if audio_only:
            self.delete_audio(meeting_id)
            return
        meeting_dir = self.root / "meetings" / meeting_id
        safe_relative(self.root, meeting_dir)
        self.update_meeting(meeting_id, status="deleting", error=None)
        try:
            if meeting_dir.exists():
                shutil.rmtree(meeting_dir)
            with self._lock, self.db:
                self.db.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        except OSError as exc:
            self.update_meeting(meeting_id, error=f"Delete incomplete: {exc}")
            raise MeetingError(f"Meeting deletion is incomplete and can be retried: {exc}", "DELETE_INCOMPLETE") from exc

    @staticmethod
    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
