from __future__ import annotations

import json

from genetic_ml.working_folder import WorkingFolderWriter


def make_candidate():
    return {"id": 3, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": [{"x": 0, "y": 0, "z": 0, "state": 165}]}


def test_save_writes_bare_candidate_json_with_ascending_names(tmp_path):
    writer = WorkingFolderWriter(tmp_path, prefix="discovered")
    candidate = make_candidate()

    first = writer.save(candidate)
    second = writer.save(candidate)

    assert first.name == "discovered_0001.json"
    assert second.name == "discovered_0002.json"
    assert json.loads(first.read_text(encoding="utf-8")) == candidate


def test_save_resumes_from_existing_files(tmp_path):
    (tmp_path / "discovered_0001.json").write_text("{}", encoding="utf-8")
    (tmp_path / "discovered_0005.json").write_text("{}", encoding="utf-8")

    writer = WorkingFolderWriter(tmp_path, prefix="discovered")
    path = writer.save(make_candidate())

    assert path.name == "discovered_0006.json"


def test_save_uses_custom_prefix(tmp_path):
    writer = WorkingFolderWriter(tmp_path, prefix="mutant")

    path = writer.save(make_candidate())

    assert path.name == "mutant_0001.json"


def test_save_ignores_files_matching_other_prefixes(tmp_path):
    (tmp_path / "simple_machine2.json").write_text("{}", encoding="utf-8")

    writer = WorkingFolderWriter(tmp_path, prefix="discovered")
    path = writer.save(make_candidate())

    assert path.name == "discovered_0001.json"
