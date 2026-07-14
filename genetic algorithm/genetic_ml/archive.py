from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Candidate = dict[str, Any]


class Archive:
    """Append-only JSONL log of every distinct working candidate the GA has ever
    found, independent of whether it's still in the live population. This is the
    actual output of a run - the live population is just the search frontier.

    Writes are buffered in memory and flushed to disk in batches (flush_every records, or
    an explicit flush() call) rather than opening/writing/closing the file on every single
    discovery - a long run can find candidates in rapid bursts, and a disk write+flush per
    discovery adds up to a lot of small writes over a long run. Dedup (has()) is unaffected:
    _known_hashes is updated immediately in record(), before the write is ever buffered or
    flushed, so in-memory correctness doesn't depend on when the disk catches up. Callers
    must call flush() before the process exits (run_ga does this in a finally block) or
    buffered-but-unflushed discoveries are lost."""

    def __init__(self, path: str | Path, flush_every: int = 20) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._known_hashes: set[str] = set()
        self._pending: list[str] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._known_hashes.add(record["hash"])

    def __len__(self) -> int:
        return len(self._known_hashes)

    def has(self, candidate_hash: str) -> bool:
        return candidate_hash in self._known_hashes

    def record(
        self,
        candidate: Candidate,
        candidate_hash: str,
        result: dict[str, Any],
        generation: int,
        origin: str,
    ) -> bool:
        """Buffers a discovery if its hash hasn't been recorded before, flushing to disk once
        flush_every have accumulated. Returns True if it was newly recorded."""

        if candidate_hash in self._known_hashes:
            return False

        entry = {
            "hash": candidate_hash,
            "generation": generation,
            "origin": origin,
            "block_count": len(candidate["blocks"]),
            "candidate": candidate,
            "result": result,
            "found_at": datetime.now(UTC).isoformat(),
        }
        self._pending.append(json.dumps(entry, separators=(",", ":")))
        self._known_hashes.add(candidate_hash)
        if len(self._pending) >= self.flush_every:
            self.flush()
        return True

    def flush(self) -> None:
        """Writes every buffered record to disk in one batch. Safe to call with nothing
        pending (no-op)."""
        if not self._pending:
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(self._pending) + "\n")
        self._pending.clear()
