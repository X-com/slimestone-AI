from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_dataset_jsonl(
    output_path: str | Path,
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    simulator_path: str | Path,
    include_candidate: bool = True,
) -> None:
    if len(candidates) != len(results):
        raise ValueError(f"candidate/result length mismatch: {len(candidates)} != {len(results)}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written_at = datetime.now(UTC).isoformat()

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate, result in zip(candidates, results, strict=True):
            record: dict[str, Any] = {
                "candidate_id": candidate.get("id"),
                "result": result,
                "simulator": {
                    "kind": "mcp1122_cpp_stream",
                    "path": str(simulator_path),
                    "written_at": written_at,
                },
            }
            if include_candidate:
                record["candidate"] = candidate
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

