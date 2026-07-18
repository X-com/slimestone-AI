from __future__ import annotations

import time
from pathlib import Path


class HashLog:
    """Append-only, fixed-width binary log of candidate hashes - just the raw hash bytes,
    back-to-back, no delimiter and no outcome flag. A hash's presence in a given HashLog
    instance already tells you everything the old Archive/TriedLog's per-record metadata used
    to encode: which file it's in (working vs. not-working) IS the outcome. The actual candidate
    data lives elsewhere (flyers.data / data/working/*.json, written by WorkingFolderWriter or
    CompactWorkingWriter) - this file only ever needs to answer "have I seen this hash before".

    hash_bytes controls how many leading bytes of canonical_hash()'s 32-byte SHA-256 digest are
    kept: 32 (the full digest, lossless, zero collision risk) for the working-hash log, where
    exactness matters because a false "yes, this works" would corrupt an RL reward or silently
    merge two distinct machines. 8 (64 bits) for the not-working log, where the volume is what
    actually caused the old text-hex tried.log to bloat, and a false positive there only ever
    causes a fresh candidate to be skipped as already-tried - a missed opportunity, never a wrong
    "confirmed working" result. At 64 bits, the birthday-bound collision probability stays
    negligible (roughly 1-in-several-million) even across tens of millions of records.

    Writes are buffered in memory and flushed to disk at most once every flush_interval_seconds
    (or on an explicit flush() call), same time-based throttle as the writers this replaces.
    has() is correct as soon as record() returns regardless of when the disk catches up. Callers
    must flush() before exit."""

    def __init__(self, path: str | Path, hash_bytes: int = 32, flush_interval_seconds: float = 1.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_bytes = hash_bytes
        self.flush_interval_seconds = flush_interval_seconds
        self._last_flush = time.monotonic()
        self._seen: set[bytes] = set()
        self._pending: list[bytes] = []
        if self.path.exists():
            with self.path.open("rb") as handle:
                while True:
                    chunk = handle.read(hash_bytes)
                    if len(chunk) < hash_bytes:
                        break  # clean EOF, or a truncated trailing record from an unclean exit
                    self._seen.add(chunk)

    def __len__(self) -> int:
        return len(self._seen)

    def _key(self, candidate_hash: str) -> bytes:
        # canonical_hash() returns a 64-char hex string (the full SHA-256 digest); this is the
        # one place that turns it into the compact on-disk key, truncated to hash_bytes.
        return bytes.fromhex(candidate_hash)[: self.hash_bytes]

    def has(self, candidate_hash: str) -> bool:
        return self._key(candidate_hash) in self._seen

    def record(self, candidate_hash: str) -> bool:
        """Marks a hash as seen if it's new, flushing to disk once flush_interval_seconds has
        elapsed since the last flush. Returns True if it was newly recorded."""
        key = self._key(candidate_hash)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._pending.append(key)
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
        with self.path.open("ab") as handle:
            handle.write(b"".join(self._pending))
        self._pending.clear()
