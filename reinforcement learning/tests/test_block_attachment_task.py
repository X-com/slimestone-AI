from __future__ import annotations

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    make_state,
)

from rl_ml.tasks.block_attachment import BlockAttachmentContext, BlockAttachmentTask

_FACING_EAST = 5  # matches genetic_ml.mutation._FACING_OFFSETS[5] == (1, 0, 0)

_MACHINE = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [
        {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, _FACING_EAST)},
        {"x": 10, "y": 0, "z": 0, "state": make_state(BLOCK_SLIME)},
        {"x": 20, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},
        {"x": 30, "y": 0, "z": 0, "state": make_state(BLOCK_OBSERVER)},
    ],
}


def _features_at(position, candidate_block_ids=None, block_id=BLOCK_SLIME):
    task = BlockAttachmentTask(candidate_block_ids)
    context = BlockAttachmentContext(machine=_MACHINE, position=position, block_id=block_id)
    return task.features(context)


def test_push_face_position_is_flagged_but_not_other_face():
    push_face_features = _features_at((1, 0, 0))  # directly in front of the piston (east)
    assert push_face_features[1] == 1.0  # push_face
    assert push_face_features[2] == 0.0  # other_piston_face


def test_non_push_face_position_is_flagged_as_other_face():
    other_face_features = _features_at((0, -1, 0))  # below the piston, not its push face
    assert other_face_features[1] == 0.0  # push_face
    assert other_face_features[2] == 1.0  # other_piston_face


def test_adjacency_indicators_for_slime_redstone_and_observer():
    assert _features_at((11, 0, 0))[3] == 1.0  # near_slime
    assert _features_at((21, 0, 0))[4] == 1.0  # near_redstone
    assert _features_at((31, 0, 0))[5] == 1.0  # near_observer


def test_isolated_position_has_no_structural_indicators_set():
    features = _features_at((100, 100, 100))
    assert features[0] == 1.0  # bias
    assert features[1:6] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_block_type_one_hot_matches_context_block_id():
    features = _features_at(
        (100, 100, 100),
        candidate_block_ids=[BLOCK_SLIME, BLOCK_REDSTONE_BLOCK],
        block_id=BLOCK_REDSTONE_BLOCK,
    )
    assert features[-2:] == [0.0, 1.0]


def test_build_candidate_returns_none_for_action_false():
    task = BlockAttachmentTask([BLOCK_SLIME])
    context = BlockAttachmentContext(machine=_MACHINE, position=(1, 0, 0), block_id=BLOCK_SLIME)

    assert task.build_candidate(context, False, candidate_id=99) is None


def test_build_candidate_appends_the_requested_block_type_without_mutating_the_original():
    task = BlockAttachmentTask([BLOCK_SLIME, BLOCK_OBSERVER])
    context = BlockAttachmentContext(machine=_MACHINE, position=(1, 0, 0), block_id=BLOCK_OBSERVER)

    candidate = task.build_candidate(context, True, candidate_id=42)

    assert candidate["id"] == 42
    assert candidate["trigger"] == _MACHINE["trigger"]
    added = candidate["blocks"][-1]
    assert (added["x"], added["y"], added["z"]) == (1, 0, 0)
    assert added["state"] == make_state(BLOCK_OBSERVER)
    assert len(_MACHINE["blocks"]) == 4  # original untouched


def test_reward_of_matches_the_incentive_compatible_table():
    task = BlockAttachmentTask()

    assert task.reward_of(False, None) == 0.0
    assert task.reward_of(True, None) == 0.0
    assert task.reward_of(True, {"validCycle": True}) == 1.0
    assert task.reward_of(True, {"validCycle": False}) == -1.0
    # the older working/cycles field alone must not be enough - validCycle is the real signal.
    assert task.reward_of(True, {"working": True, "validCycle": False}) == -1.0


def test_default_candidate_block_ids_is_slime_only():
    assert BlockAttachmentTask().candidate_block_ids == [BLOCK_SLIME]
