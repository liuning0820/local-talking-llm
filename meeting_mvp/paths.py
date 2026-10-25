"""Data-root and path traversal safeguards."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import MeetingError


def safe_relative(root: Path, path: Path) -> str:
    root = root.expanduser().resolve()
    candidate = path.expanduser().resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise MeetingError("Path must remain inside the meeting data directory", "PATH_OUTSIDE_DATA_DIR") from exc
    return relative.as_posix()


def resolve_relative(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise MeetingError("Stored path must be a non-empty relative path", "INVALID_STORED_PATH")
    candidate = (root / relative).resolve()
    safe_relative(root, candidate)
    return candidate


def atomic_replace(source: Path, destination: Path, root: Path | None = None) -> None:
    if root is not None:
        safe_relative(root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
