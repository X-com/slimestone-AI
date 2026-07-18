from __future__ import annotations

import time

from genetic_ml.hash_log import HashLog

_HASH_A = "aa" * 32  # 64 hex chars = a valid canonical_hash()-shaped SHA-256 digest
_HASH_B = "bb" * 32
_HASH_A_TWIN = "aa" * 8 + "cc" * 24  # shares hash_bytes=8's leading prefix with _HASH_A


def _bare_log(hash_bytes: int) -> HashLog:
    """A HashLog with a huge flush interval so record() never auto-flushes mid-test - avoids
    touching disk for pure in-memory checks."""
    log = HashLog.__new__(HashLog)
    log.hash_bytes = hash_bytes
    log._seen = set()
    log._pending = []
    log.flush_interval_seconds = 3600.0
    log._last_flush = time.monotonic()
    return log


def test_record_is_compact_and_dedupes():
    log = _bare_log(hash_bytes=32)

    assert log.record(_HASH_A) is True
    assert log.record(_HASH_A) is False  # already seen, second attempt ignored
    assert log.has(_HASH_A) is True
    assert log.has(_HASH_B) is False
    assert log._pending == [bytes.fromhex(_HASH_A)]  # raw hash bytes, no delimiter, no flag


def test_truncation_keeps_only_the_leading_hash_bytes():
    log = _bare_log(hash_bytes=8)

    log.record(_HASH_A)

    assert log.has(_HASH_A) is True
    # A hash sharing only the first 8 bytes with _HASH_A is indistinguishable at this
    # truncation length - this is the accepted collision trade-off for the not-working log.
    assert log.has(_HASH_A_TWIN) is True


def test_resumes_seen_hashes_from_disk(tmp_path):
    path = tmp_path / "working_hashes.log"
    path.write_bytes(bytes.fromhex(_HASH_A) + bytes.fromhex(_HASH_B))

    log = HashLog(path, hash_bytes=32)

    assert log.has(_HASH_A) is True
    assert log.has(_HASH_B) is True
    assert len(log) == 2


def test_flush_writes_pending_records_once(tmp_path):
    path = tmp_path / "not_working_hashes.log"
    log = HashLog(path, hash_bytes=8, flush_interval_seconds=3600.0)

    log.record(_HASH_A)
    log.record(_HASH_B)
    assert not path.exists()  # buffered, not yet on disk

    log.flush()

    assert path.read_bytes() == bytes.fromhex(_HASH_A)[:8] + bytes.fromhex(_HASH_B)[:8]


def test_auto_flushes_once_the_interval_elapses(tmp_path):
    path = tmp_path / "hashes.log"
    log = HashLog(path, hash_bytes=8, flush_interval_seconds=0.05)

    log.record(_HASH_A)
    assert not path.exists()  # interval hasn't elapsed yet

    time.sleep(0.06)
    log.record(_HASH_B)  # this record() call is the one that notices and flushes

    assert path.read_bytes() == bytes.fromhex(_HASH_A)[:8] + bytes.fromhex(_HASH_B)[:8]
