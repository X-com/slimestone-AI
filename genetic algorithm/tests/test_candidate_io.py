from __future__ import annotations

import json

from genetic_ml.candidate_io import load_candidates_from_file


def test_load_candidates_from_jsonl(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps({"id": 1, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": []}) + "\n"
        + "\n"
        + json.dumps({"id": 2, "trigger": {"x": 1, "y": 2, "z": 3}, "blocks": []}) + "\n",
        encoding="utf-8",
    )

    candidates = load_candidates_from_file(path)

    assert [candidate["id"] for candidate in candidates] == [1, 2]

