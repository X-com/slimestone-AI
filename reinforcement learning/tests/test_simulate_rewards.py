from __future__ import annotations

import itertools

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.archive import Archive
from genetic_ml.compact_format import read_compact_file
from genetic_ml.compact_working_writer import CompactWorkingWriter

from rl_ml.tasks.dummy_task import DummyContext, DummyTask
from rl_ml.train_loop import _simulate_rewards

_MACHINE = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [{"x": 0, "y": 0, "z": 0, "state": 1}],
}


class _StubPool:
    """Fakes genetic_ml.simulator_pool.SimulatorPool.run_all() without touching the real
    simulator - _simulate_rewards only ever calls pool.run_all(candidates), so this is all the
    interface it needs."""

    def __init__(self, results_by_candidate_id: dict[int, dict]) -> None:
        self.results_by_candidate_id = results_by_candidate_id

    def run_all(self, candidates: list[dict]) -> list[dict]:
        return [self.results_by_candidate_id[candidate["id"]] for candidate in candidates]


def test_rewards_align_with_contexts_regardless_of_which_actions_were_simulated():
    task = DummyTask()
    contexts = [
        DummyContext(machine=_MACHINE, position=(1, 0, 0)),
        DummyContext(machine=_MACHINE, position=(2, 0, 0)),
    ]
    actions = [True, False]  # only the first gets simulated (build_candidate(False, ...) is None)
    pool = _StubPool({1: {"working": True}})

    rewards = _simulate_rewards(
        task, contexts, actions, pool, itertools.count(1), archive=None, working_writer=None, generation=1
    )

    assert rewards == [1.0, 0.0]


def test_a_failed_addition_is_never_archived_or_saved(tmp_path):
    task = DummyTask()
    contexts = [DummyContext(machine=_MACHINE, position=(1, 0, 0))]
    archive = Archive(tmp_path / "archive.jsonl")
    writer = CompactWorkingWriter(tmp_path / "compact")
    pool = _StubPool({1: {"working": False}})

    rewards = _simulate_rewards(
        task, contexts, [True], pool, itertools.count(1), archive, writer, generation=1
    )

    assert rewards == [-1.0]
    assert len(archive) == 0
    assert writer._pending == []


def test_rediscovering_the_same_candidate_saves_to_working_writer_only_once(tmp_path):
    task = DummyTask()
    contexts = [DummyContext(machine=_MACHINE, position=(1, 0, 0))]
    archive = Archive(tmp_path / "archive.jsonl")
    writer = CompactWorkingWriter(tmp_path / "compact")
    next_id = itertools.count(1)

    first = _simulate_rewards(
        task, contexts, [True], _StubPool({1: {"working": True}}), next_id, archive, writer, generation=1
    )
    second = _simulate_rewards(
        task, contexts, [True], _StubPool({2: {"working": True}}), next_id, archive, writer, generation=2
    )

    assert first == [1.0]
    assert second == [1.0]
    assert len(archive) == 1  # deduped by canonical_hash despite the different candidate ids
    assert len(writer._pending) == 1  # working_writer.save() only called on the first discovery

    archive.flush()
    writer.flush()
    assert len(read_compact_file(tmp_path / "compact" / "flyers.data")) == 1
