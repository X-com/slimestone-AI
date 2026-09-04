"""canonical_hash - DECISIONS.md point 23.

The asymmetry that governs every test here: missing a duplicate costs one wasted simulation,
while wrongly declaring two different machines identical means a valid machine is skipped and
never discovered, silently and permanently. Everything below is biased toward under-merging.
"""
from __future__ import annotations

import copy

from rlgym.game import Machine, canonical_hash


def _shift(candidate: dict, dx: int, dy: int, dz: int) -> dict:
    out = copy.deepcopy(candidate)
    for block in out["blocks"]:
        block["x"] += dx
        block["y"] += dy
        block["z"] += dz
    out["trigger"]["x"] += dx
    out["trigger"]["y"] += dy
    out["trigger"]["z"] += dz
    return out


def test_translation_invariant(engine_candidate):
    """Correct: a flying machine is the same machine wherever it is - and it does not stay
    where it started, so absolute position cannot be part of identity."""
    base = canonical_hash(engine_candidate)
    assert canonical_hash(_shift(engine_candidate, 100, 100, 100)) == base
    assert canonical_hash(_shift(engine_candidate, -7, 0, 3)) == base


def test_block_order_does_not_matter(engine_candidate):
    shuffled = copy.deepcopy(engine_candidate)
    shuffled["blocks"] = list(reversed(shuffled["blocks"]))
    assert canonical_hash(shuffled) == canonical_hash(engine_candidate)


def test_candidate_id_does_not_matter(engine_candidate):
    other = copy.deepcopy(engine_candidate)
    other["id"] = 999999
    assert canonical_hash(other) == canonical_hash(engine_candidate)


def test_trigger_is_part_of_identity(engine_candidate):
    """Identical blocks with a different trigger is a different machine - it starts somewhere
    else and so behaves differently."""
    moved = copy.deepcopy(engine_candidate)
    moved["trigger"] = {"x": 1, "y": 0, "z": 2}
    assert canonical_hash(moved) != canonical_hash(engine_candidate)


def test_changing_one_block_changes_the_hash(engine_candidate):
    changed = copy.deepcopy(engine_candidate)
    changed["blocks"][0]["state"] = 165
    assert canonical_hash(changed) != canonical_hash(engine_candidate)


def test_adding_a_block_changes_the_hash(engine_candidate):
    grown = copy.deepcopy(engine_candidate)
    grown["blocks"].append({"x": 50, "y": 50, "z": 50, "state": 1})
    assert canonical_hash(grown) != canonical_hash(engine_candidate)


def test_removing_a_block_changes_the_hash(engine_candidate):
    shrunk = copy.deepcopy(engine_candidate)
    shrunk["blocks"] = shrunk["blocks"][:-1]
    assert canonical_hash(shrunk) != canonical_hash(engine_candidate)


def test_rotation_is_a_different_machine(engine_candidate):
    """Deliberately NOT rotation-invariant. Rotating changes which block is the anchor
    (simulator.cpp:2052 orders by y, then z, then x) and changes neighbour update order, which
    is fixed in world terms - so rotations genuinely can behave differently."""
    rotated = copy.deepcopy(engine_candidate)
    for block in rotated["blocks"]:
        block["x"], block["z"] = block["z"], -block["x"]
    rotated["trigger"]["x"], rotated["trigger"]["z"] = (
        rotated["trigger"]["z"],
        -rotated["trigger"]["x"],
    )
    assert canonical_hash(rotated) != canonical_hash(engine_candidate)


def test_hash_is_stable_across_calls(engine_candidate):
    assert canonical_hash(engine_candidate) == canonical_hash(engine_candidate)


def test_hash_is_a_sha256_hex_digest(engine_candidate):
    digest = canonical_hash(engine_candidate)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_empty_machine_does_not_crash():
    assert canonical_hash({"id": 0, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": []})


def test_machine_round_trip_preserves_hash(engine_candidate):
    """Going through Machine and back must not change identity - otherwise every hash computed
    from a rebuilt machine would disagree with one computed from the fixture."""
    machine = Machine.from_candidate(engine_candidate)
    assert canonical_hash(machine.to_candidate()) == canonical_hash(engine_candidate)
