from __future__ import annotations

import time
from pathlib import Path


class TriedLog:
    """Append-only, compact log of every distinct candidate shape the GA has ever simulated,
    working or not - just the hash and a one-char outcome flag per line, no candidate/result
    payload. Archive already keeps the full record of working discoveries; this log exists
    purely so the GA can cheaply skip re-simulating a candidate (or a structurally-identical
    one from an unrelated lineage) whose shape was already tried and failed. Without it, as
    the population converges, mutation increasingly proposes offspring near already-tried dead
    ends, wasting a growing share of each generation's simulator budget on repeats.

    Writes are buffered in memory and flushed to disk at most once every flush_interval_seconds
    (or on an explicit flush() call), same time-based throttle as Archive - has()/outcome() are
    correct as soon as record() returns regardless of when the disk catches up. Callers must
    flush() before exit.

    The stored outcome isn't just a dedup flag - outcome() lets a caller (e.g. the RL reward
    function) recover the working/not-working result of an already-tried shape and compute its
    reward directly, without paying for another simulation."""

    def __init__(self, path: str | Path, flush_interval_seconds: float = 1.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval_seconds = flush_interval_seconds
        self._last_flush = time.monotonic()
        self._tried: dict[str, bool] = {}
        self._pending: list[str] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    candidate_hash, _, outcome = line.strip().partition(",")
                    if candidate_hash:
                        self._tried[candidate_hash] = outcome == "1"

    def __len__(self) -> int:
        return len(self._tried)

    def has(self, candidate_hash: str) -> bool:
        return candidate_hash in self._tried

    def outcome(self, candidate_hash: str) -> bool | None:
        """Returns the recorded working/not-working outcome for a previously-tried hash, or
        None if it has never been simulated."""
        return self._tried.get(candidate_hash)

    def record(self, candidate_hash: str, working: bool) -> bool:
        """Marks a hash as tried if it's new, flushing to disk once flush_interval_seconds has
        elapsed since the last flush. Returns True if it was newly recorded."""
        if candidate_hash in self._tried:
            return False
        self._tried[candidate_hash] = working
        self._pending.append(f"{candidate_hash},{'1' if working else '0'}")
        if time.monotonic() - self._last_flush >= self.flush_interval_seconds:
            self.flush()
        return True

    def flush(self) -> None:
        """Writes every buffered record to disk in one batch. Safe to call with nothing
        pending (no-op). Always resets the flush-interval clock, whether or not anything was
        actually pending, so an explicit flush() also restarts the throttle window."""
        self._last_flush = time.monotonic()
        if not self._pending:
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(self._pending) + "\n")
        self._pending.clear()
