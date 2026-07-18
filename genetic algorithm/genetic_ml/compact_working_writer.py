from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from genetic_ml.compact_format import encode_candidate

Candidate = dict[str, Any]


class CompactWorkingWriter:
    """Appends every newly discovered working flying machine as one compact binary record
    onto a single file (compact-working/flyers.data by default), instead of
    WorkingFolderWriter's one-JSON-file-per-discovery. Same save(candidate) -> Path interface
    as WorkingFolderWriter, so ga_loop.py's call site doesn't need to know which one it has.

    Records are buffered in memory and written to disk at most once every
    flush_interval_seconds (or on an explicit flush()), rather than opening the file and
    flushing after every single discovery - a long run can find candidates in rapid bursts, and
    a disk write+flush per discovery adds up to a lot of small writes over a long run.
    Time-based rather than count-based so the write rate stays bounded regardless of discovery
    rate. Callers must call flush() before the process exits (run_ga does this in a finally
    block) or buffered-but-unflushed discoveries are lost."""

    def __init__(
        self, directory: str | Path, filename: str = "flyers.data", flush_interval_seconds: float = 1.0
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename
        self.flush_interval_seconds = flush_interval_seconds
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._pending: list[bytes] = []

    def save(self, candidate: Candidate) -> Path:
        with self._lock:
            self._pending.append(encode_candidate(candidate))
            if time.monotonic() - self._last_flush >= self.flush_interval_seconds:
                self._flush_locked()
        return self.path

    def flush(self) -> None:
        """Writes every buffered record to disk in one batch. Safe to call with nothing
        pending (no-op)."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._last_flush = time.monotonic()
        if not self._pending:
            return
        with self.path.open("ab") as handle:
            handle.write(b"".join(self._pending))
            handle.flush()
        self._pending.clear()
