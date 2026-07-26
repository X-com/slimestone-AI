"""Verifies the generator package actually works - previously untested (no test anywhere called
build_corpus, any gen_*.generate, or runner.simulate; only ever exercised by hand via generate.py).

Two tiers: structural (fast, no simulator - candidate shape sanity) and integration (slow, runs
the real C++ simulator via corpus.build_corpus - the actual "does this generator produce
simulator-accepted output" proof). Run from `transformer gym/`:
    py -m pytest generator/tests/test_generators.py -v
Requires cpp_simulator_stream.exe built and MSYS2 UCRT64 on PATH (see runner.py/verify_simulation_data.py).
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generator  # noqa: F401,E402

import gen_forward  # noqa: E402
import gen_perturb  # noqa: E402
import gen_pushgroup  # noqa: E402
import gen_puzzle  # noqa: E402
import gen_random  # noqa: E402
import gen_splice  # noqa: E402
from corpus import build_corpus  # noqa: E402
from genetic_ml.blocks import BLOCK_OBSERVER, BLOCK_PISTON, BLOCK_STICKY_PISTON, block_id  # noqa: E402
from wave_function import gen_wfc  # noqa: E402

_TRIGGERABLE = {BLOCK_PISTON, BLOCK_STICKY_PISTON, BLOCK_OBSERVER}

GENERATORS = (
    ("perturb", gen_perturb.generate),
    ("forward", gen_forward.generate),
    ("pushgroup", gen_pushgroup.generate),
    ("puzzle", gen_puzzle.generate),
    ("splice", gen_splice.generate),
    ("random", gen_random.generate),
    ("wave_function", gen_wfc.generate),
)


def _assert_valid_structure(name: str, candidate: dict) -> None:
    blocks = candidate["blocks"]
    assert blocks, f"{name}: candidate has no blocks"
    positions = [(b["x"], b["y"], b["z"]) for b in blocks]
    assert len(positions) == len(set(positions)), f"{name}: duplicate block positions"

    trigger = candidate["trigger"]
    tpos = (trigger["x"], trigger["y"], trigger["z"])
    by_pos = {(b["x"], b["y"], b["z"]): b for b in blocks}
    assert tpos in by_pos, f"{name}: trigger position {tpos} has no block there"
    assert block_id(by_pos[tpos]["state"]) in _TRIGGERABLE, (
        f"{name}: trigger block is not a piston/sticky_piston/observer"
    )


@pytest.mark.parametrize("name,generate", GENERATORS)
def test_candidate_structure_is_sane(name: str, generate) -> None:
    rng = random.Random(0)
    gen = generate(rng)
    for _ in range(10):
        candidate = next(gen)
        _assert_valid_structure(name, candidate)


@pytest.mark.parametrize("name,generate", GENERATORS)
def test_generator_produces_simulator_accepted_output(name: str, generate) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats = build_corpus(
            Path(tmp), count=8, seed=3, max_attempts=250, generators=((name, generate, 1.0),),
        )
    assert stats["accepted"] >= 1, f"{name}: {stats}"
    assert stats["sim_error"] == 0, f"{name}: simulator raised SimRunError - {stats}"


def test_default_generator_mix_end_to_end() -> None:
    # Catches mix/dedup/dead-filter interaction bugs a single-generator run wouldn't - the real
    # weighted GENERATOR_WEIGHTS mix corpus.py ships with.
    with tempfile.TemporaryDirectory() as tmp:
        stats = build_corpus(Path(tmp), count=30, seed=7)
    assert stats["accepted"] == 30, stats
    assert stats["sim_error"] == 0, stats
