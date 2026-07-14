from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

Candidate = dict[str, Any]


class WorkingFolderWriter:
    """Saves every newly discovered working flying machine as its own JSON file in
    data/working, using an editable name prefix plus an ascending postfix
    (`<prefix>_0001.json`, `<prefix>_0002.json`, ...). The postfix continues from
    whatever's already on disk, so re-running never collides with or overwrites an
    earlier discovery. `prefix` is meant to be tweaked in main_ga.py - it's just the
    default name a newly discovered machine gets until someone renames it."""

    def __init__(self, directory: str | Path, prefix: str = "discovered") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._lock = threading.Lock()
        self._next_index = self._scan_next_index()

    def _scan_next_index(self) -> int:
        pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d+)\.json$")
        max_index = 0
        for path in self.directory.glob(f"{self.prefix}_*.json"):
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def save(self, candidate: Candidate) -> Path:
        with self._lock:
            index = self._next_index
            self._next_index += 1

        path = self.directory / f"{self.prefix}_{index:04d}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(candidate, handle, separators=(",", ":"))
        return path
