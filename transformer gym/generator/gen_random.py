"""Doc §9's 10% "uniform random, including dead structures" slice. Deliberately kept small - its
role is teaching the model that some structures do nothing at all, not generating variety (that's
what the other five generators are for).
"""
from __future__ import annotations

import random
from typing import Any, Iterator

import generator  # noqa: F401
from genetic_ml.blocks import INSERTABLE_KINDS, make_state
from geometry import pick_trigger

Candidate = dict[str, Any]

BOX_SIZE = (8, 8, 8)
FILL_RANGE = (0.10, 0.25)
TRIGGER_RETRIES = 10


def _build(rng: random.Random) -> Candidate | None:
    fill = rng.uniform(*FILL_RANGE)
    blocks = []
    for x in range(BOX_SIZE[0]):
        for y in range(BOX_SIZE[1]):
            for z in range(BOX_SIZE[2]):
                if rng.random() < fill:
                    kind = rng.choice(INSERTABLE_KINDS)
                    blocks.append({"x": x, "y": y, "z": z, "state": kind.random_state(rng)})
    trigger = pick_trigger(blocks, rng)
    if trigger is None:
        return None
    return {"id": 0, "trigger": {"x": trigger[0], "y": trigger[1], "z": trigger[2]}, "blocks": blocks}


def generate(rng: random.Random) -> Iterator[Candidate]:
    while True:
        for _ in range(TRIGGER_RETRIES):
            result = _build(rng)
            if result is not None:
                yield result
                break
