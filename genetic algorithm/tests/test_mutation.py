from __future__ import annotations

import random

from genetic_ml.blocks import block_id, is_palette_state
from genetic_ml.mutation import mutate


def make_candidate():
    return {
        "id": 1,
        "trigger": {"x": 0, "y": 0, "z": 0},
        "blocks": [
            {"x": 0, "y": 0, "z": 0, "state": 33},  # piston
            {"x": 0, "y": 1, "z": 0, "state": 152},  # redstone block
            {"x": 0, "y": 2, "z": 0, "state": 165},  # slime
        ],
    }


def test_mutate_does_not_touch_original_candidate():
    original = make_candidate()
    snapshot = {
        "id": original["id"],
        "trigger": dict(original["trigger"]),
        "blocks": [dict(block) for block in original["blocks"]],
    }

    rng = random.Random(0)
    mutate(original, rng, op_count=3)

    assert original == snapshot


def test_mutate_preserves_trigger_position():
    candidate = make_candidate()
    rng = random.Random(1)

    for _ in range(20):
        candidate = mutate(candidate, rng, op_count=2)
        trigger_key = (candidate["trigger"]["x"], candidate["trigger"]["y"], candidate["trigger"]["z"])
        block_keys = {(b["x"], b["y"], b["z"]) for b in candidate["blocks"]}
        assert trigger_key in block_keys or len(candidate["blocks"]) == 0


def test_mutate_only_emits_palette_block_ids():
    candidate = make_candidate()
    rng = random.Random(2)

    for _ in range(50):
        candidate = mutate(candidate, rng, op_count=3)
        for block in candidate["blocks"]:
            assert is_palette_state(block["state"]), block_id(block["state"])


def test_mutate_is_deterministic_for_a_given_seed():
    candidate = make_candidate()

    result_a = mutate(candidate, random.Random(42), op_count=3)
    result_b = mutate(candidate, random.Random(42), op_count=3)

    assert result_a == result_b
