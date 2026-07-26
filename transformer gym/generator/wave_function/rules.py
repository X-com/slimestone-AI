"""Hand-authored adjacency rules ("set specific rules", not learned/extracted from fixtures - see
the plan's future-extensions section for that variant) over an alphabet built from
genetic_ml.blocks.MUTATION_PALETTE's kinds.

Almost every placement is "physically legal" in this simulator (nothing crashes regardless of
adjacency), so these rules aren't about avoiding impossible states - they're a small, explicit,
deliberately minimal set of *plausibility* constraints, kept easy to read and extend:

  1. A piston's front face never wants another piston's body there (pushing directly into
     another piston is a placement that's never structurally useful).
  2. An observer's SENSED face (offset(pos, facing) - the corrected geometry from gen_forward.py/
     tiles.py, not offset(pos, opposite(facing))) never wants permanent air - an observer facing
     nothing but air can never fire.

Everything else defaults to compatible. `compatible` is symmetric by construction (checked in
tests/test_wfc.py over the full alphabet): compatible(a, d, b) == compatible(b, opposite(d), a).
"""
from __future__ import annotations

from genetic_ml.blocks import (
    BLOCK_AIR,
    BLOCK_GLAZED_TERRACOTTA,
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
    BLOCK_STONE,
    block_id,
    block_meta,
    make_state,
)
from geometry import FACING_OPPOSITE

_NON_FACING_KINDS = (BLOCK_AIR, BLOCK_STONE, BLOCK_GLAZED_TERRACOTTA, BLOCK_SLIME, BLOCK_REDSTONE_BLOCK)
_FACING_KINDS = (BLOCK_PISTON, BLOCK_STICKY_PISTON, BLOCK_OBSERVER)
_PISTON_KINDS = (BLOCK_PISTON, BLOCK_STICKY_PISTON)


def build_alphabet() -> list[int]:
    """Every legal `state` int a WFC cell can collapse to: one state per non-facing kind, plus
    one state per (facing kind, facing 0-5) pair."""
    states = [make_state(k) for k in _NON_FACING_KINDS]
    states += [make_state(k, f) for k in _FACING_KINDS for f in range(6)]
    return states


def compatible(state_a: int, direction: int, state_b: int) -> bool:
    """Is `state_b` allowed to sit at the `direction` neighbor of a cell holding `state_a`?"""
    kind_a, facing_a = block_id(state_a), block_meta(state_a) & 0b111
    kind_b, facing_b = block_id(state_b), block_meta(state_b) & 0b111
    back_direction = FACING_OPPOSITE[direction]

    # Rule 1: piston front-face vs. piston front-face (checked from both sides for symmetry).
    if kind_a in _PISTON_KINDS and direction == facing_a and kind_b in _PISTON_KINDS:
        return False
    if kind_b in _PISTON_KINDS and back_direction == facing_b and kind_a in _PISTON_KINDS:
        return False

    # Rule 2: observer sensed-face vs. air (checked from both sides for symmetry).
    if kind_a == BLOCK_OBSERVER and direction == facing_a and kind_b == BLOCK_AIR:
        return False
    if kind_b == BLOCK_OBSERVER and back_direction == facing_b and kind_a == BLOCK_AIR:
        return False

    return True
