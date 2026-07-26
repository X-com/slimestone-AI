"""Generic Wave Function Collapse over a 3D grid of opaque integer states - no domain knowledge
here at all (see rules.py for the block alphabet + compatibility function this operates on).

Contradiction handling: restarts the whole grid with a fresh RNG draw rather than backtracking.
Simplest correct option and cheap at this grid size (≤512 cells) - true backtracking WFC is real
added complexity, noted in the plan as the upgrade if restart-thrashing turns out to matter in
practice (measure it via run_wfc.py's stats before optimizing).
"""
from __future__ import annotations

import random
from typing import Callable

from geometry import FACING_OFFSETS

Pos = tuple[int, int, int]
CompatFn = Callable[[int, int, int], bool]  # (state_a, direction, state_b) -> bool


class Contradiction(Exception):
    pass


def _neighbors(pos: Pos, size: Pos):
    x, y, z = pos
    sx, sy, sz = size
    for direction, (dx, dy, dz) in enumerate(FACING_OFFSETS):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
            yield direction, (nx, ny, nz)


def collapse(
    size: Pos,
    alphabet: list[int],
    compatible: CompatFn,
    weights: dict[int, float],
    rng: random.Random,
    max_restarts: int = 20,
) -> dict[Pos, int]:
    """Runs WFC over a `size` grid, returning {pos: collapsed_state}. Raises Contradiction if no
    restart within `max_restarts` succeeds."""
    for _ in range(max_restarts):
        try:
            return _collapse_once(size, alphabet, compatible, weights, rng)
        except Contradiction:
            continue
    raise Contradiction(f"no valid collapse found in {max_restarts} restarts")


def _collapse_once(
    size: Pos, alphabet: list[int], compatible: CompatFn, weights: dict[int, float], rng: random.Random,
) -> dict[Pos, int]:
    sx, sy, sz = size
    all_positions = [(x, y, z) for x in range(sx) for y in range(sy) for z in range(sz)]
    possible: dict[Pos, set[int]] = {p: set(alphabet) for p in all_positions}
    collapsed: dict[Pos, int] = {}

    def propagate(start: Pos) -> None:
        queue = [start]
        while queue:
            p = queue.pop()
            for direction, np in _neighbors(p, size):
                if np in collapsed:
                    continue
                allowed = {
                    sb for sb in possible[np]
                    if any(compatible(sa, direction, sb) for sa in possible[p])
                }
                if not allowed:
                    raise Contradiction(f"{np} has no possibilities left")
                if allowed != possible[np]:
                    possible[np] = allowed
                    queue.append(np)

    remaining = list(all_positions)
    while remaining:
        min_entropy = min(len(possible[p]) for p in remaining)
        tied = [p for p in remaining if len(possible[p]) == min_entropy]
        pos = rng.choice(tied)

        choices = list(possible[pos])
        pick_weights = [weights.get(s, 1.0) for s in choices]
        state = rng.choices(choices, weights=pick_weights, k=1)[0]

        collapsed[pos] = state
        possible[pos] = {state}
        remaining.remove(pos)
        propagate(pos)

    return collapsed
