from __future__ import annotations

import itertools
import time

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.compact_format import encode_candidate, read_compact_file
from genetic_ml.compact_working_writer import CompactWorkingWriter
from genetic_ml.hash_log import HashLog
from genetic_ml.population import canonical_hash

from rl_ml.tasks.dummy_task import DummyContext, DummyTask
from rl_ml.train_loop import _simulate_rewards

_MACHINE = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [{"x": 0, "y": 0, "z": 0, "state": 1}],
}


def _bare_hash_log(hash_bytes: int) -> HashLog:
    """A HashLog with a huge flush interval so record() never auto-flushes mid-test - avoids
    touching disk for pure in-memory checks."""
    log = HashLog.__new__(HashLog)
    log.hash_bytes = hash_bytes
    log._seen = set()
    log._pending = []
    log.flush_interval_seconds = 3600.0
    log._last_flush = time.monotonic()
    return log


class _StubPool:
    """Fakes genetic_ml.simulator_pool.SimulatorPool.run_all() without touching the real
    simulator - _simulate_rewards only ever calls pool.run_all(candidates), so this is all the
    interface it needs."""

    def __init__(self, results_by_candidate_id: dict[int, dict]) -> None:
        self.results_by_candidate_id = results_by_candidate_id

    def run_all(self, candidates: list[dict]) -> list[dict]:
        return [self.results_by_candidate_id[candidate["id"]] for candidate in candidates]


class _StubHub:
    """Fakes genetic_ml.stream_hub.StreamHub's publish(frame) - _simulate_rewards only ever
    calls that one method."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def publish(self, frame: bytes) -> None:
        self.frames.append(frame)


class _ExplodingPool:
    """A pool whose run_all() must never be called - used to prove a cache hit skips
    simulation entirely rather than merely stubbing out what the (unreachable) result would be."""

    def run_all(self, candidates: list[dict]) -> list[dict]:
        raise AssertionError(f"run_all() should not have been called for {candidates!r} - expected a cache hit")


def test_rewards_align_with_contexts_regardless_of_which_actions_were_simulated():
    task = DummyTask()
    contexts = [
        DummyContext(machine=_MACHINE, position=(1, 0, 0)),
        DummyContext(machine=_MACHINE, position=(2, 0, 0)),
    ]
    actions = [True, False]  # only the first gets simulated (build_candidate(False, ...) is None)
    pool = _StubPool({1: {"validCycle": True}})

    rewards = _simulate_rewards(task, contexts, actions, pool, itertools.count(1), working_writer=None)

    assert rewards == [1.0, 0.0]


def test_a_failed_addition_is_never_recorded_as_working_or_saved(tmp_path):
    task = DummyTask()
    contexts = [DummyContext(machine=_MACHINE, position=(1, 0, 0))]
    working_hashes = HashLog(tmp_path / "working_hashes.log", hash_bytes=32)
    not_working_hashes = HashLog(tmp_path / "not_working_hashes.log", hash_bytes=8)
    writer = CompactWorkingWriter(tmp_path / "compact")
    pool = _StubPool({1: {"validCycle": False}})

    rewards = _simulate_rewards(
        task, contexts, [True], pool, itertools.count(1), writer, working_hashes, not_working_hashes
    )

    assert rewards == [-1.0]
    assert len(working_hashes) == 0
    assert len(not_working_hashes) == 1
    assert writer._pending == []


def test_rediscovering_the_same_candidate_saves_to_working_writer_only_once(tmp_path):
    task = DummyTask()
    contexts = [DummyContext(machine=_MACHINE, position=(1, 0, 0))]
    working_hashes = HashLog(tmp_path / "working_hashes.log", hash_bytes=32)
    not_working_hashes = HashLog(tmp_path / "not_working_hashes.log", hash_bytes=8)
    writer = CompactWorkingWriter(tmp_path / "compact")
    next_id = itertools.count(1)

    first = _simulate_rewards(
        task, contexts, [True], _StubPool({1: {"validCycle": True}}), next_id, writer,
        working_hashes, not_working_hashes,
    )
    second = _simulate_rewards(
        task, contexts, [True], _StubPool({2: {"validCycle": True}}), next_id, writer,
        working_hashes, not_working_hashes,
    )

    assert first == [1.0]
    assert second == [1.0]
    assert len(working_hashes) == 1  # deduped by canonical_hash despite the different candidate ids
    assert len(writer._pending) == 1  # working_writer.save() only called on the first discovery

    working_hashes.flush()
    writer.flush()
    assert len(read_compact_file(tmp_path / "compact" / "flyers.data")) == 1


def test_hash_logs_record_every_simulated_outcome_in_the_matching_file():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(1, 0, 0))
    working_hashes = _bare_hash_log(hash_bytes=32)
    not_working_hashes = _bare_hash_log(hash_bytes=8)
    pool = _StubPool({1: {"validCycle": True}})

    _simulate_rewards(
        task, [context], [True], pool, itertools.count(1), working_writer=None,
        working_hashes=working_hashes, not_working_hashes=not_working_hashes,
    )

    candidate = task.build_candidate(context, True, candidate_id=99)
    assert working_hashes.has(canonical_hash(candidate)) is True
    assert not_working_hashes.has(canonical_hash(candidate)) is False


def test_working_cache_hit_skips_resimulation_and_recovers_the_same_reward():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(1, 0, 0))
    candidate = task.build_candidate(context, True, candidate_id=1)
    candidate_hash = canonical_hash(candidate)

    working_hashes = _bare_hash_log(hash_bytes=32)
    working_hashes._seen = {bytes.fromhex(candidate_hash)}
    not_working_hashes = _bare_hash_log(hash_bytes=8)

    rewards = _simulate_rewards(
        task, [context], [True], _ExplodingPool(), itertools.count(1), working_writer=None,
        working_hashes=working_hashes, not_working_hashes=not_working_hashes,
    )

    assert rewards == [1.0]  # same reward task.reward_of(True, {"validCycle": True}) would give


def test_not_working_cache_hit_skips_resimulation_and_recovers_the_same_reward():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(1, 0, 0))
    candidate = task.build_candidate(context, True, candidate_id=1)
    candidate_hash = canonical_hash(candidate)

    working_hashes = _bare_hash_log(hash_bytes=32)
    not_working_hashes = _bare_hash_log(hash_bytes=8)
    not_working_hashes._seen = {bytes.fromhex(candidate_hash)[:8]}

    rewards = _simulate_rewards(
        task, [context], [True], _ExplodingPool(), itertools.count(1), working_writer=None,
        working_hashes=working_hashes, not_working_hashes=not_working_hashes,
    )

    assert rewards == [-1.0]  # same reward task.reward_of(True, {"validCycle": False}) would give


def test_a_newly_discovered_working_candidate_is_published_as_one_frame():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(1, 0, 0))
    hub = _StubHub()
    pool = _StubPool({1: {"validCycle": True}})

    _simulate_rewards(
        task, [context], [True], pool, itertools.count(1), working_writer=None,
        working_hashes=_bare_hash_log(hash_bytes=32), stream_hub=hub,
    )

    candidate = task.build_candidate(context, True, candidate_id=1)
    assert hub.frames == [encode_candidate(candidate)]


def test_failed_and_cache_hit_candidates_never_publish():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(1, 0, 0))
    hub = _StubHub()

    _simulate_rewards(
        task, [context], [True], _StubPool({1: {"validCycle": False}}), itertools.count(1),
        working_writer=None, stream_hub=hub,
    )
    assert hub.frames == []

    candidate = task.build_candidate(context, True, candidate_id=1)
    working_hashes = _bare_hash_log(hash_bytes=32)
    working_hashes._seen = {bytes.fromhex(canonical_hash(candidate))}

    _simulate_rewards(
        task, [context], [True], _ExplodingPool(), itertools.count(1), working_writer=None,
        working_hashes=working_hashes, stream_hub=hub,
    )
    assert hub.frames == []
