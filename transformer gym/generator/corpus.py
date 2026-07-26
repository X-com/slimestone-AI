"""Orchestrates the six generators at the approved corpus mix, dedups, applies the zero-event
rejection filter (sim_event_log.cpp:149's "zero events is the generator's free rejection filter"),
and writes accepted `.simlog` files to a shard directory that transformer_gym/dataset.py's
SimlogDirDataset can train from directly (no simulator run needed at train time).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import generator  # noqa: F401
import gen_forward
import gen_perturb
import gen_pushgroup
import gen_puzzle
import gen_random
import gen_splice
from genetic_ml.population import canonical_hash
from runner import SimRunError, simulate
from transformer_gym.encode import TooManyNodesError, encode
from transformer_gym.simlog_reader import read_footer

Candidate = dict[str, Any]

# Rebalanced mix from the approved plan: perturb 25 / forward 20 / puzzle 20 / push-group 15 /
# splice 10 / random 10.
GENERATOR_WEIGHTS: tuple[tuple[str, Any, float], ...] = (
    ("perturb", gen_perturb.generate, 25),
    ("forward", gen_forward.generate, 20),
    ("puzzle", gen_puzzle.generate, 20),
    ("pushgroup", gen_pushgroup.generate, 15),
    ("splice", gen_splice.generate, 10),
    ("random", gen_random.generate, 10),
)

# The doc explicitly wants some dead (zero-event) structures kept, not just filtered out - the
# model needs "does nothing" as a real, represented outcome, not an absence.
KEEP_DEAD_FRACTION = 0.10


def build_corpus(out_dir: Path, count: int, seed: int = 0, max_attempts: int | None = None) -> dict:
    """Writes up to `count` accepted .simlog files to out_dir. Returns a small stats dict
    (attempted/accepted/rejected-by-reason) for the caller (generate.py's CLI) to report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    names = [n for n, _, _ in GENERATOR_WEIGHTS]
    weights = [w for _, _, w in GENERATOR_WEIGHTS]
    iterators = {n: gen(rng) for n, gen, _ in GENERATOR_WEIGHTS}

    seen_hashes: set[str] = set()
    stats = {"attempted": 0, "accepted": 0, "dup": 0, "sim_error": 0, "dead": 0, "too_big": 0}
    max_attempts = max_attempts if max_attempts is not None else count * 20
    next_id = 1

    while stats["accepted"] < count and stats["attempted"] < max_attempts:
        stats["attempted"] += 1
        name = rng.choices(names, weights=weights, k=1)[0]
        try:
            candidate = next(iterators[name])
        except StopIteration:
            continue

        h = canonical_hash(candidate)
        if h in seen_hashes:
            stats["dup"] += 1
            continue

        candidate["id"] = next_id
        try:
            log = simulate(candidate)
        except SimRunError:
            stats["sim_error"] += 1
            continue

        event_count = read_footer(log)["eventCount"]
        if event_count == 0 and rng.random() > KEEP_DEAD_FRACTION:
            stats["dead"] += 1
            continue

        shard_path = out_dir / f"{name}_{next_id}.simlog"
        shard_path.write_bytes(log)
        try:
            encode(shard_path)  # cheap node-count check only; result discarded here
        except TooManyNodesError:
            stats["too_big"] += 1
            shard_path.unlink()
            continue

        seen_hashes.add(h)
        stats["accepted"] += 1
        next_id += 1

    return stats
