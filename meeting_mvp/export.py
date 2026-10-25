"""Safe JSON and Markdown exports."""

from __future__ import annotations
import json
from pathlib import Path
from .paths import atomic_replace, safe_relative
from .store import Store
from .errors import MeetingError


def export_meeting(store: Store, meeting_id: str, fmt: str, output: str | None = None) -> Path:
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
    summary = store.summary(meeting_id)
    data = {"meeting": meeting, "transcript": store.transcripts(meeting_id), "summary": summary["payload"] if summary else None}
    if fmt == "json":
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        suffix = ".json"
    elif fmt == "markdown":
        content = (summary["markdown"] if summary else "# 会议纪要\n\n未生成纪要。\n")
        suffix = ".md"
    else:
        raise MeetingError("format must be markdown or json", "INVALID_EXPORT_FORMAT")
    destination = Path(output).expanduser() if output else store.root / "meetings" / meeting_id / "exports" / ("meeting" + suffix)
    if not destination.is_absolute():
        destination = store.root / destination
    temp = destination.with_name(destination.name + ".tmp")
    if destination.parent != store.root and store.root not in destination.resolve().parents:
        # Explicit absolute paths are supported, but never accept a traversal path.
        if not Path(output or destination).is_absolute():
            raise MeetingError("Relative exports must stay inside the data directory", "PATH_OUTSIDE_DATA_DIR")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(content, encoding="utf-8")
    atomic_replace(temp, destination)
    return destination
