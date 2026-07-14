from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Literal

Candidate = dict[str, Any]
FailureKind = Literal["crash", "hang"]


def _next_index(directory: Path, stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.json$")
    max_index = 0
    for path in directory.glob(f"{stem}_*.json"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


class FailureLogger:
    """Writes one JSON file per crash- or hang-inducing mutated candidate the
    simulator hits, so they can be replayed later to harden the (still
    work-in-progress) C++ simulator. Crashes and hangs go into separate
    directories since they're different classes of bug: a native process crash
    (e.g. bad memory access) versus an unbounded update cascade that never
    returns. Each file is nothing but the candidate JSON itself, in the same
    format a working machine would be saved in, so it can be re-fed straight
    into the simulator or the GA as a repro case. Thread-safe since SimulatorPool
    logs from multiple worker threads."""

    def __init__(self, crash_dir: str | Path, hang_dir: str | Path) -> None:
        self.crash_dir = Path(crash_dir)
        self.hang_dir = Path(hang_dir)
        self.crash_dir.mkdir(parents=True, exist_ok=True)
        self.hang_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_index = {
            "crash": _next_index(self.crash_dir, "crash"),
            "hang": _next_index(self.hang_dir, "hung"),
        }
        self._counters = {"crash": 0, "hang": 0}

    def log(self, kind: FailureKind, candidate: Candidate) -> Path:
        directory = self.crash_dir if kind == "crash" else self.hang_dir
        stem = "crash" if kind == "crash" else "hung"

        with self._lock:
            index = self._next_index[kind]
            self._next_index[kind] += 1
            self._counters[kind] += 1

        path = directory / f"{stem}_{index:04d}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(candidate, handle, separators=(",", ":"))
        return path

    @property
    def crash_count(self) -> int:
        return self._counters["crash"]

    @property
    def hang_count(self) -> int:
        return self._counters["hang"]
