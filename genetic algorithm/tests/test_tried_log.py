from __future__ import annotations

import time

from genetic_ml.tried_log import TriedLog


def _bare_log() -> TriedLog:
    """A TriedLog with a huge flush interval so record() never auto-flushes mid-test - avoids
    touching disk for pure in-memory checks."""
    log = TriedLog.__new__(TriedLog)
    log._tried = {}
    log._pending = []
    log.flush_interval_seconds = 3600.0
    log._last_flush = time.monotonic()
    return log


def test_record_is_compact_and_dedupes():
    log = _bare_log()

    assert log.record("abc", working=False) is True
    assert log.record("abc", working=True) is False  # already tried, second attempt ignored
    assert log.has("abc") is True
    assert log.has("xyz") is False
    assert log._pending == ["abc,0"]  # one-char outcome flag, no candidate/result payload


def test_outcome_lets_a_caller_recover_the_cached_result_without_resimulating():
    log = _bare_log()

    log.record("working-hash", working=True)
    log.record("failed-hash", working=False)

    assert log.outcome("working-hash") is True
    assert log.outcome("failed-hash") is False
    assert log.outcome("never-tried") is None


def test_resumes_tried_hashes_from_disk_regardless_of_outcome(tmp_path):
    path = tmp_path / "tried.log"
    path.write_text("workinghash,1\nfailedhash,0\n", encoding="utf-8")

    log = TriedLog(path)

    assert log.has("workinghash") is True
    assert log.has("failedhash") is True
    assert len(log) == 2


def test_flush_writes_pending_records_once(tmp_path):
    path = tmp_path / "tried.log"
    log = TriedLog(path, flush_interval_seconds=3600.0)

    log.record("h1", working=False)
    log.record("h2", working=True)
    assert not path.exists()  # buffered, not yet on disk

    log.flush()

    assert path.read_text(encoding="utf-8") == "h1,0\nh2,1\n"


def test_auto_flushes_once_the_interval_elapses(tmp_path):
    path = tmp_path / "tried.log"
    log = TriedLog(path, flush_interval_seconds=0.05)

    log.record("h1", working=True)
    assert not path.exists()  # interval hasn't elapsed yet

    time.sleep(0.06)
    log.record("h2", working=False)  # this record() call is the one that notices and flushes

    assert path.read_text(encoding="utf-8") == "h1,1\nh2,0\n"
