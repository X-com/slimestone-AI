from __future__ import annotations

import random
from typing import Any

from genetic_ml.blocks import (
    BLOCK_PISTON,
    BLOCK_PISTON_HEAD,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_STICKY_PISTON,
    FACING_KINDS,
    INSERTABLE_KINDS,
    PISTON_EXTENDED_BIT,
    block_id,
    block_meta,
    kind_for_state,
    make_state,
)

Candidate = dict[str, Any]

_PISTON_BLOCK_IDS = {BLOCK_PISTON, BLOCK_STICKY_PISTON}

# down, up, north, south, west, east - matches blocks.py's FACING_DOWN..FACING_EAST (0-5) order.
_FACING_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
)

# How far a newly-added block may land from an existing block, in blocks per axis (used by
# add_block only - move_block always moves exactly 1). 2 instead of 1 allows a gap between an
# added block and its anchor, e.g. a piston with empty space before the block it pushes/pulls -
# needed for machines that grab a block from a distance rather than only ever touching it.
NEIGHBOR_RADIUS = 2

# Hard cap on how far any block may end up from the trigger, in blocks per axis (a 33x33x33 box
# centered on the trigger - 16 in every direction: up/down, left/right, forward/backward).
# add_block/move_block are otherwise an uncapped random walk (equal weight to grow vs. shrink,
# no size/count limit), so without this a machine that keeps "working" after each addition can
# grow arbitrarily large over an indefinite run. The trigger is the natural fixed anchor for
# this, since it's the one position mutations never move or remove.
BOUNDING_BOX_RADIUS = 3


def _block_key(block: dict[str, Any]) -> tuple[int, int, int]:
    return (block["x"], block["y"], block["z"])


def _within_bounds(position: tuple[int, int, int], trigger: tuple[int, int, int]) -> bool:
    return all(abs(position[i] - trigger[i]) <= BOUNDING_BOX_RADIUS for i in range(3))


def _pick_neighbor_position(blocks: list[dict[str, Any]], rng: random.Random) -> tuple[int, int, int]:
    anchor = rng.choice(blocks)
    dx = rng.randint(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS)
    dy = rng.randint(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS)
    dz = rng.randint(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS)
    return (anchor["x"] + dx, anchor["y"] + dy, anchor["z"] + dz)


def add_block(candidate: Candidate, rng: random.Random) -> bool:
    blocks: list[dict[str, Any]] = candidate["blocks"]
    if not blocks:
        return False

    trigger = _block_key(candidate["trigger"])
    occupied = {_block_key(block) for block in blocks}
    for _ in range(8):
        position = _pick_neighbor_position(blocks, rng)
        if position in occupied or not _within_bounds(position, trigger):
            continue
        kind = rng.choice(INSERTABLE_KINDS)
        blocks.append(
            {
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "state": kind.random_state(rng),
            }
        )
        return True
    return False


def remove_block(candidate: Candidate, rng: random.Random) -> bool:
    blocks: list[dict[str, Any]] = candidate["blocks"]
    trigger = _block_key(candidate["trigger"])
    removable = [block for block in blocks if _block_key(block) != trigger]
    if len(removable) <= 1:
        return False

    victim = rng.choice(removable)
    blocks.remove(victim)
    return True


def move_block(candidate: Candidate, rng: random.Random) -> bool:
    blocks: list[dict[str, Any]] = candidate["blocks"]
    trigger = _block_key(candidate["trigger"])
    movable = [block for block in blocks if _block_key(block) != trigger]
    if not movable:
        return False

    occupied = {_block_key(block) for block in blocks}
    block = rng.choice(movable)
    axis = rng.choice(("x", "y", "z"))
    delta = rng.choice((-1, 1))
    new_key = list(_block_key(block))
    new_key["xyz".index(axis)] += delta
    new_key_tuple = tuple(new_key)
    if new_key_tuple in occupied or not _within_bounds(new_key_tuple, trigger):
        return False

    block["x"], block["y"], block["z"] = new_key_tuple
    return True


def retype_block(candidate: Candidate, rng: random.Random) -> bool:
    blocks: list[dict[str, Any]] = candidate["blocks"]
    trigger = _block_key(candidate["trigger"])
    retypable = [block for block in blocks if _block_key(block) != trigger]
    if not retypable:
        return False

    block = rng.choice(retypable)
    kind = rng.choice(INSERTABLE_KINDS)
    block["state"] = kind.random_state(rng)
    return True


def reface_block(candidate: Candidate, rng: random.Random) -> bool:
    blocks: list[dict[str, Any]] = candidate["blocks"]
    candidates_to_reface = [block for block in blocks if kind_for_state(block["state"]) in FACING_KINDS]
    if not candidates_to_reface:
        return False

    block = rng.choice(candidates_to_reface)
    kind = kind_for_state(block["state"])
    assert kind is not None
    new_facing = rng.choice(kind.facings)
    block["state"] = make_state(kind.block_id, new_facing)
    return True


def _settle_piston_extensions(candidate: Candidate) -> None:
    """Keeps every piston's extended/retracted representation consistent with whether it's
    directly touching a permanent power source (a redstone block) on one of its non-front
    faces - matching Simulator::shouldPistonBeExtended's power rule (simulator.cpp:459-499),
    simplified to just redstone-block adjacency since that's the only static/permanent power
    source the mutation palette can produce (no redstone dust wiring exists to generate).

    This matters because block placement bypasses the normal setBlockState()-triggered
    neighbor-update cascade (see world.setBlock/directSetBlock in all three engines) - a
    piston loaded from disk already touching a redstone block never gets a notification to
    discover it should extend, and just sits there incorrectly retracted for the whole run
    instead of starting extended like an already-settled real world would.

    Called once at the end of mutate(), after piston-head blocks have been stripped and the
    normal operators have run - so this always re-derives extension state fresh from the
    final piston/redstone positions rather than trying to incrementally patch stale state,
    which for free also retracts (and removes the head of) any piston that lost its power
    source to this mutation."""
    blocks: list[dict[str, Any]] = candidate["blocks"]
    occupied = {_block_key(block) for block in blocks}
    redstone_positions = {
        _block_key(block) for block in blocks if block_id(block["state"]) == BLOCK_REDSTONE_BLOCK
    }

    for block in list(blocks):
        bid = block_id(block["state"])
        if bid not in _PISTON_BLOCK_IDS:
            continue

        facing = block_meta(block["state"]) & 0b111
        pos = _block_key(block)
        powered = any(
            (pos[0] + dx, pos[1] + dy, pos[2] + dz) in redstone_positions
            for direction, (dx, dy, dz) in enumerate(_FACING_OFFSETS)
            if direction != facing  # a piston's own front face never counts as a power source
        )

        if not powered:
            block["state"] = make_state(bid, facing)
            continue

        head_dx, head_dy, head_dz = _FACING_OFFSETS[facing]
        head_pos = (pos[0] + head_dx, pos[1] + head_dy, pos[2] + head_dz)
        if head_pos in occupied:
            # Can't represent "extended" - nothing occupies the head's cell in real Minecraft
            # either, but here there's no room to place it, so leave this one retracted.
            block["state"] = make_state(bid, facing)
            continue

        block["state"] = make_state(bid, facing | (1 << PISTON_EXTENDED_BIT))
        sticky_bit = (1 << PISTON_EXTENDED_BIT) if bid == BLOCK_STICKY_PISTON else 0
        blocks.append(
            {
                "x": head_pos[0],
                "y": head_pos[1],
                "z": head_pos[2],
                "state": make_state(BLOCK_PISTON_HEAD, facing | sticky_bit),
            }
        )
        occupied.add(head_pos)


# Weighted so structural changes (add/remove/move) happen a bit more often than
# state-only tweaks, since block-set topology is what actually creates new machines.
MUTATION_OPERATORS: tuple[tuple[str, Any, float], ...] = (
    ("add_block", add_block, 1.0),
    ("remove_block", remove_block, 1.0),
    ("move_block", move_block, 1.0),
    ("retype_block", retype_block, 0.75),
    ("reface_block", reface_block, 0.75),
)


def mutate(candidate: Candidate, rng: random.Random, op_count: int = 1) -> Candidate:
    """Return a deep-copied, mutated candidate. Applies op_count operators in sequence,
    retrying a different operator if the chosen one is a no-op (e.g. remove_block on a
    single-block machine), up to a small budget.

    Piston-head blocks are stripped before mutating (they're always derived, never a "real"
    independently-placed block - see _settle_piston_extensions) and re-derived fresh
    afterward, so the four operators above never need to know piston heads exist at all."""

    mutated: Candidate = {
        "id": candidate["id"],
        "trigger": dict(candidate["trigger"]),
        "blocks": [
            dict(block) for block in candidate["blocks"] if block_id(block["state"]) != BLOCK_PISTON_HEAD
        ],
    }
    for key, value in candidate.items():
        if key not in mutated:
            mutated[key] = value

    names = [name for name, _, _ in MUTATION_OPERATORS]
    ops = [op for _, op, _ in MUTATION_OPERATORS]
    weights = [weight for _, _, weight in MUTATION_OPERATORS]

    applied = 0
    attempts = 0
    max_attempts = op_count * 6
    while applied < op_count and attempts < max_attempts:
        attempts += 1
        op = rng.choices(ops, weights=weights, k=1)[0]
        if op(mutated, rng):
            applied += 1

    _settle_piston_extensions(mutated)
    return mutated
