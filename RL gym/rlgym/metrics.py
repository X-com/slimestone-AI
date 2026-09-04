"""One JSONL row per evaluation - the build plan's "answerable retrospectively".

Deliberately not a metrics library. The rule is that every row carries the **whole config** as
well as the numbers, so a question nobody thought to ask during the run - "did `reward_cargo`
below 1 actually move non-cargo discoveries?" - is answerable afterwards by grouping rows,
rather than requiring the run to be repeated with the right counter added.

Append-only. Rows are never rewritten, so a crashed run leaves valid data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Metrics:
    def __init__(self, path: Path | None, context: dict[str, Any] | None = None):
        self.path = Path(path) if path is not None else None
        self.context = context or {}
        self.rows: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **values: Any) -> dict[str, Any]:
        row = {"kind": kind, "wall": round(time.time(), 3), **self.context, **values}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        return row

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["kind"] == kind]


def read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
