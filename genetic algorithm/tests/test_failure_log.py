from __future__ import annotations

import json

from genetic_ml.failure_log import FailureLogger


def make_candidate():
    return {"id": 7, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": [{"x": 0, "y": 0, "z": 0, "state": 33}]}


def test_log_writes_separate_folders_for_crash_and_hang(tmp_path):
    crash_dir = tmp_path / "crash"
    hang_dir = tmp_path / "hangs"
    logger = FailureLogger(crash_dir, hang_dir)

    crash_path = logger.log("crash", make_candidate())
    hang_path = logger.log("hang", make_candidate())

    assert crash_path.parent == crash_dir
    assert hang_path.parent == hang_dir
    assert logger.crash_count == 1
    assert logger.hang_count == 1


def test_log_writes_only_the_bare_candidate_json(tmp_path):
    logger = FailureLogger(tmp_path / "crash", tmp_path / "hangs")
    candidate = make_candidate()

    crash_path = logger.log("crash", candidate)
    hang_path = logger.log("hang", candidate)

    assert crash_path.name == "crash_0001.json"
    assert hang_path.name == "hung_0001.json"
    assert json.loads(crash_path.read_text(encoding="utf-8")) == candidate
    assert json.loads(hang_path.read_text(encoding="utf-8")) == candidate


def test_log_does_not_overwrite_across_repeated_calls(tmp_path):
    logger = FailureLogger(tmp_path / "crash", tmp_path / "hangs")
    candidate = make_candidate()

    paths = {logger.log("crash", candidate) for _ in range(5)}

    assert len(paths) == 5
    assert logger.crash_count == 5


def test_log_resumes_postfix_from_existing_files(tmp_path):
    crash_dir = tmp_path / "crash"
    crash_dir.mkdir()
    (crash_dir / "crash_0001.json").write_text("{}", encoding="utf-8")
    (crash_dir / "crash_0003.json").write_text("{}", encoding="utf-8")

    logger = FailureLogger(crash_dir, tmp_path / "hangs")
    path = logger.log("crash", make_candidate())

    assert path.name == "crash_0004.json"
