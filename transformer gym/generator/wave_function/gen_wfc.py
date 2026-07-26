"""generate(rng) -> Iterator[Candidate], same interface as every gen_*.py in generator/, so wiring
this into corpus.py later (if the standalone stats in run_wfc.py justify it) is a one-line
addition - not done this pass per this session's decision to evaluate it standalone first.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

from genetic_ml.blocks import BLOCK_AIR, block_id
from geometry import pick_trigger
from wave_function.rules import build_alphabet, compatible
from wave_function.wfc import Contradiction, collapse

Candidate = dict[str, Any]
Pos = tuple[int, int, int]

BOX_SIZE = (8, 8, 8)
# Air must dominate the weighted draw, not just outweigh any single other symbol: the alphabet
# has 22 non-air states, so a naively "heavy" weight like 8 (vs 1 each) still only gives ~27% air
# - a pathologically dense, near-full board. Measured empirically (not guessed): at ~15% fill
# (~75-90 solid blocks in a 512-cell box), encode.py's node count (solid + synthesized interface-
# air tokens) came out at 320-400+ - almost always over the 300-node cap, meaning nearly every
# candidate was silently wasting a full simulate() subprocess call only to be rejected afterward
# (looks like a hang under a large --count; it's actually near-zero throughput). Tuned for ~91%
# air (~46 expected solid blocks -> comfortably under the cap with the ratio observed above).
AIR_WEIGHT = 220.0
TRIGGER_RETRIES = 10
# Reject oversized boards BEFORE spending a simulate() call on them (encode.py's node count -
# solid blocks plus synthesized interface-air - runs at ~4-4.5x the solid block count for these
# boards; this stays comfortably under the 300-node cap even on an unlucky denser-than-expected
# draw, rather than relying on AIR_WEIGHT tuning alone).
MAX_SOLID_BLOCKS = 55


def _weights(alphabet: list[int]) -> dict[int, float]:
    return {s: (AIR_WEIGHT if block_id(s) == BLOCK_AIR else 1.0) for s in alphabet}


def _to_candidate(grid: dict[Pos, int], rng: random.Random) -> Candidate | None:
    blocks = [{"x": p[0], "y": p[1], "z": p[2], "state": s} for p, s in grid.items() if block_id(s) != BLOCK_AIR]
    if len(blocks) > MAX_SOLID_BLOCKS:
        return None
    trigger = pick_trigger(blocks, rng)
    if trigger is None:
        return None
    return {"id": 0, "trigger": {"x": trigger[0], "y": trigger[1], "z": trigger[2]}, "blocks": blocks}


def generate(rng: random.Random) -> Iterator[Candidate]:
    alphabet = build_alphabet()
    weights = _weights(alphabet)
    while True:
        for _ in range(TRIGGER_RETRIES):
            try:
                grid = collapse(BOX_SIZE, alphabet, compatible, weights, rng)
            except Contradiction:
                continue
            candidate = _to_candidate(grid, rng)
            if candidate is not None:
                yield candidate
                break
