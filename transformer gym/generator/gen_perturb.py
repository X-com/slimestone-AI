"""Generator 3 (doc) + the user's crop-and-mutate idea: perturbations of real fixtures.

Small fixtures are perturbed whole. Large machines (tank ~4.7k blocks, complex_machine ~3.1k, ...)
are first cropped to a random 8x8x8 window - that's what makes them usable at training size and
gives genuine local redstone structure to mutate, instead of either skipping them or perturbing
something too big for encode.py's node cap.

Every candidate - cropped or not - is routed through genetic_ml.mutation.mutate() (op_count=0 is
a pure crop/copy, no-op otherwise): its _settle_piston_extensions cleans up boundary-severed
pistons in a crop for free (a piston that lost its power source or head-space outside the window
gets correctly re-derived to retracted, rather than left in an inconsistent state).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import generator  # noqa: F401
from genetic_ml.candidate_io import load_candidates_from_file
from genetic_ml.mutation import mutate
from geometry import bbox, crop_window, pick_trigger
from holdout import HOLDOUT_FIXTURES

Candidate = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "flying machines" / "json"

WINDOW_SIZE = (8, 8, 8)
SMALL_BLOCK_LIMIT = 40  # whole-fixture path below this; crop path at/above it
CROP_RETRIES = 20


def _seed_fixtures() -> list[Candidate]:
    seeds = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        if path.stem in HOLDOUT_FIXTURES:
            continue
        seeds.extend(load_candidates_from_file(path))
    return seeds


def _op_count_ladder(block_count: int, rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.6:
        return rng.randint(1, 3)
    if roll < 0.9:
        return rng.randint(5, 15)
    frac = rng.uniform(0.2, 0.5)
    return max(1, int(block_count * frac))


def _crop_candidate(seed: Candidate, rng: random.Random) -> Candidate | None:
    (minx, miny, minz), (maxx, maxy, maxz) = bbox(seed["blocks"])
    for _ in range(CROP_RETRIES):
        ox = rng.randint(minx, max(minx, maxx - WINDOW_SIZE[0] + 1))
        oy = rng.randint(miny, max(miny, maxy - WINDOW_SIZE[1] + 1))
        oz = rng.randint(minz, max(minz, maxz - WINDOW_SIZE[2] + 1))
        cropped = crop_window(seed["blocks"], (ox, oy, oz), WINDOW_SIZE)
        if not cropped:
            continue
        trigger = pick_trigger(cropped, rng)
        if trigger is None:
            continue
        return {"id": 0, "trigger": {"x": trigger[0], "y": trigger[1], "z": trigger[2]}, "blocks": cropped}
    return None


def generate(rng: random.Random) -> Iterator[Candidate]:
    seeds = _seed_fixtures()
    while True:
        seed = rng.choice(seeds)
        block_count = len(seed["blocks"])
        if block_count < SMALL_BLOCK_LIMIT:
            base = {"id": 0, "trigger": dict(seed["trigger"]), "blocks": [dict(b) for b in seed["blocks"]]}
        else:
            base = _crop_candidate(seed, rng)
            if base is None:
                continue
        op_count = _op_count_ladder(len(base["blocks"]), rng)
        yield mutate(base, rng, op_count=op_count) if op_count > 0 else base
