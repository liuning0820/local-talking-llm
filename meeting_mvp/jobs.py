"""Small retry policy shared by transcription and summary jobs."""

from __future__ import annotations
import time
from collections.abc import Callable
from typing import TypeVar
from .errors import PermanentJobError

T = TypeVar("T")
RETRY_DELAYS = (5, 30, 300)


def run_with_retry(operation: Callable[[], T], *, attempts: int = 3, sleep: Callable[[float], None] = time.sleep) -> tuple[T | None, int, Exception | None]:
    last: Exception | None = None
    for number in range(1, attempts + 1):
        try:
            return operation(), number, None
        except PermanentJobError as exc:
            return None, number, exc
        except Exception as exc:
            last = exc
            if number < attempts:
                sleep(RETRY_DELAYS[number - 1])
    return None, attempts, last
