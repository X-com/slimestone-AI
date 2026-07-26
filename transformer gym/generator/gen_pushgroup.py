"""Generator 2 (doc §4): targeted push-group cases. Group-size reasoning is the highest-value
concept in the domain and random placement almost never lands on the interesting cases, so this
builds them directly: a piston, a front column of N blocks (oversampled around the push-limit
boundary 10-14), with slime injected to drag perpendicular side blocks into the group - so group
size is NOT reliably readable off column length, which is exactly the case worth training on.
Some columns get an obsidian cap (guaranteed-immovable -> guaranteed failure event).
"""
from __future__ import annotations

import random
from typing import Any, Iterator

import generator  # noqa: F401
from genetic_ml.blocks import (
    BLOCK_OBSIDIAN,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STONE,
    FACING_EAST,
    FACING_WEST,
    make_state,
)
from geometry import FACING_OFFSETS

Candidate = dict[str, Any]

# Push-limit boundary lives at 12 in this simulator's registry (mirrors real Minecraft); sample
# heavily around it so failure/success both show up near the edge, not just deep in either side.
_N_CHOICES = list(range(1, 17))
_N_WEIGHTS = [1] * 9 + [4] * 5 + [1] * 2  # N=1-9 low, N=10-14 high, N=15-16 low

# Perpendicular-to-push axes for a piston facing EAST/WEST (push axis = x).
_SIDE_OFFSETS = ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _build(rng: random.Random) -> Candidate:
    n = rng.choices(_N_CHOICES, weights=_N_WEIGHTS, k=1)[0]
    piston = {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, FACING_EAST)}
    dx, dy, dz = FACING_OFFSETS[FACING_WEST]
    redstone = {"x": dx, "y": dy, "z": dz, "state": make_state(BLOCK_REDSTONE_BLOCK)}
    blocks = [piston, redstone]

    slime_slots = set(rng.sample(range(1, n + 1), k=min(rng.randint(0, 2), n))) if n else set()
    for i in range(1, n + 1):
        state = make_state(BLOCK_SLIME) if i in slime_slots else make_state(BLOCK_STONE)
        blocks.append({"x": i, "y": 0, "z": 0, "state": state})
        if i in slime_slots:
            for sdx, sdy, sdz in _SIDE_OFFSETS:
                blocks.append({"x": i + sdx, "y": sdy, "z": sdz, "state": make_state(BLOCK_STONE)})

    if rng.random() < 0.3:
        blocks.append({"x": n + 1, "y": 0, "z": 0, "state": make_state(BLOCK_OBSIDIAN)})

    return {"id": 0, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": blocks}


def generate(rng: random.Random) -> Iterator[Candidate]:
    while True:
        yield _build(rng)
