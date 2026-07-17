from __future__ import annotations

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import BLOCK_SLIME, make_state

from rl_ml.tasks.dummy_task import DummyContext, DummyTask, _candidate_positions

_MACHINE = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [
        {"x": 0, "y": 0, "z": 0, "state": 1},
        {"x": 1, "y": 0, "z": 0, "state": 1},
    ],
}


def test_candidate_positions_are_air_neighbors_excluding_occupied_cells():
    positions = _candidate_positions(_MACHINE)

    assert (0, 0, 0) not in positions
    assert (1, 0, 0) not in positions
    # (2,0,0) is a face-neighbor of the block at (1,0,0) and currently air.
    assert (2, 0, 0) in positions
    assert len(positions) == len(set(positions))  # deduped


def test_features_are_bias_only():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(2, 0, 0))

    assert task.features(context) == [1.0]


def test_build_candidate_returns_none_for_action_false():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(2, 0, 0))

    assert task.build_candidate(context, False, candidate_id=99) is None


def test_build_candidate_appends_one_slime_block_for_action_true():
    task = DummyTask()
    context = DummyContext(machine=_MACHINE, position=(2, 0, 0))

    candidate = task.build_candidate(context, True, candidate_id=99)

    assert candidate is not None
    assert candidate["id"] == 99
    assert candidate["trigger"] == _MACHINE["trigger"]
    assert len(candidate["blocks"]) == len(_MACHINE["blocks"]) + 1
    added = candidate["blocks"][-1]
    assert (added["x"], added["y"], added["z"]) == (2, 0, 0)
    assert added["state"] == make_state(BLOCK_SLIME)
    # original blocks list must not be mutated in place
    assert len(_MACHINE["blocks"]) == 2


def test_reward_of_matches_the_incentive_compatible_table():
    task = DummyTask()

    assert task.reward_of(False, None) == 0.0
    assert task.reward_of(True, None) == 0.0
    assert task.reward_of(True, {"validCycle": True}) == 1.0
    assert task.reward_of(True, {"validCycle": False}) == -1.0
