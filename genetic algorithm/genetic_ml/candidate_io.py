from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


Candidate = dict[str, Any]


def load_candidates_from_glob(pattern: str | Path) -> list[Candidate]:
    paths = sorted(Path(path) for path in glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(f"No candidate files matched: {pattern}")

    candidates: list[Candidate] = []
    for path in paths:
        candidates.extend(load_candidates_from_file(path))
    return candidates


def load_candidates_from_file(path: str | Path) -> list[Candidate]:
    file_path = Path(path)
    candidates: list[Candidate] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            candidate = json.loads(stripped)
            if not isinstance(candidate, dict):
                raise ValueError(f"{file_path}:{line_number} is not a JSON object")
            validate_candidate(candidate, file_path, line_number)
            candidates.append(candidate)
    return candidates


def validate_candidate(candidate: Candidate, source: Path | None = None, line_number: int | None = None) -> None:
    location = ""
    if source is not None:
        location = str(source)
        if line_number is not None:
            location += f":{line_number}"
        location += " "

    if "id" not in candidate:
        raise ValueError(f"{location}candidate is missing id")
    if "trigger" not in candidate:
        raise ValueError(f"{location}candidate is missing trigger")
    if "blocks" not in candidate:
        raise ValueError(f"{location}candidate is missing blocks")

