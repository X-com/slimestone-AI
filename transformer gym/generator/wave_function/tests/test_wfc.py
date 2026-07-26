from __future__ import annotations

import random

from generator.geometry import FACING_OFFSETS, FACING_OPPOSITE
from generator.wave_function.rules import build_alphabet, compatible
from generator.wave_function.wfc import collapse
from genetic_ml.blocks import BLOCK_AIR, BLOCK_OBSERVER, FACING_EAST, FACING_WEST, make_state


def test_compatible_is_symmetric_over_full_alphabet():
    alphabet = build_alphabet()
    for a in alphabet:
        for d in range(6):
            for b in alphabet:
                assert compatible(a, d, b) == compatible(b, FACING_OPPOSITE[d], a), (a, d, b)


def test_observer_sensed_face_rejects_air():
    # Regression guard against reintroducing the exact facing-direction bug fixed earlier this
    # session in gen_forward.py/tiles.py: the sensed cell is offset(pos, facing), not
    # offset(pos, opposite(facing)).
    observer = make_state(BLOCK_OBSERVER, FACING_EAST)
    air = make_state(BLOCK_AIR)
    assert compatible(observer, FACING_EAST, air) is False
    assert compatible(air, FACING_WEST, observer) is False  # symmetric check


def test_small_grid_collapses_consistently():
    alphabet = build_alphabet()
    weights = {s: 1.0 for s in alphabet}
    rng = random.Random(0)
    grid = collapse((3, 3, 3), alphabet, compatible, weights, rng, max_restarts=50)

    assert len(grid) == 27
    for (x, y, z), state in grid.items():
        for direction, (dx, dy, dz) in enumerate(FACING_OFFSETS):
            npos = (x + dx, y + dy, z + dz)
            if npos in grid:
                assert compatible(state, direction, grid[npos])
