"""Doc §5: splice two structures adjacent along a random axis. Mostly broken, structurally
plausible, multiplies variety with no design work. Reuses gen_perturb.generate for the two
pieces (already-sized small-fixture or cropped-large-machine candidates) rather than re-deriving
its small/crop split - splicing is just "take two of those and glue them together."
"""
from __future__ import annotations

import random
from typing import Any, Iterator

import generator  # noqa: F401
import gen_perturb
from geometry import bbox, translate

Candidate = dict[str, Any]


def _splice(a: Candidate, b: Candidate, rng: random.Random) -> Candidate:
    (a_min, a_max) = bbox(a["blocks"])
    (b_min, _b_max) = bbox(b["blocks"])
    axis = rng.randrange(3)

    offset = [a_min[i] - b_min[i] for i in range(3)]
    offset[axis] = (a_max[axis] + 1) - b_min[axis]  # b starts one cell past a's far face
    b_blocks = translate(b["blocks"], tuple(offset))
    b_trigger = (b["trigger"]["x"] + offset[0], b["trigger"]["y"] + offset[1], b["trigger"]["z"] + offset[2])

    keep_a_trigger = rng.random() < 0.5
    trigger = a["trigger"] if keep_a_trigger else {"x": b_trigger[0], "y": b_trigger[1], "z": b_trigger[2]}
    return {"id": 0, "trigger": trigger, "blocks": [dict(x) for x in a["blocks"]] + b_blocks}


def generate(rng: random.Random) -> Iterator[Candidate]:
    pieces = gen_perturb.generate(rng)
    while True:
        a, b = next(pieces), next(pieces)
        yield _splice(a, b, rng)
