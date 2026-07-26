"""Comprehensive mechanic-tagged verification suite (design doc §2): one minimal, hand-built
candidate per mechanic in mechanics.MECHANIC_ORDER, each hand-tagged with the mechanic(s) it
provably exercises. "Provably" is checked directly - build_suite() simulates every candidate and
asserts the intended tag(s) actually appear via mechanics.derive_tags before handing it back, so
a fixture can never silently stop exercising what it claims to.

Coverage note (ponytail: known ceiling): this covers the 5 mechanic-order buckets that drive the
curriculum and eval, built from the push/drag/non-stick/observer primitives. It does not yet have
a dedicated fixture per raw simulator event kind (e.g. PistonExtendBlocked, BlockDestroyed,
ComponentSplit, RedstoneBlockAppeared/Removed) - add one under _CANDIDATES when a specific rare
kind's coverage is actually needed, following the same pattern.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import generator  # noqa: F401  (sys.path shim, must run before the generator-package imports below)
from genetic_ml.blocks import (
    BLOCK_GLAZED_TERRACOTTA,
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
    BLOCK_STONE,
    FACING_EAST,
    FACING_WEST,
    make_state,
)
from runner import simulate

from .encode import Sample, encode
from .mechanics import derive_tags

Candidate = dict[str, Any]

_CANDIDATES: tuple[tuple[str, Candidate, frozenset[str]], ...] = (
    (
        "push_order",
        {
            "trigger": {"x": 0, "y": 0, "z": 0},
            "blocks": [
                {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, FACING_EAST)},
                {"x": -1, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},
                {"x": 1, "y": 0, "z": 0, "state": make_state(BLOCK_STONE)},
            ],
        },
        frozenset({"push_order"}),
    ),
    (
        "sticky_drag",
        {
            "trigger": {"x": 0, "y": 0, "z": 0},
            "blocks": [
                {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_STICKY_PISTON, FACING_EAST)},
                {"x": -1, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},
                {"x": 1, "y": 0, "z": 0, "state": make_state(BLOCK_SLIME)},
                {"x": 2, "y": 0, "z": 0, "state": make_state(BLOCK_STONE)},
            ],
        },
        frozenset({"push_order", "sticky_drag", "composed"}),
    ),
    (
        "non_stick",
        {
            "trigger": {"x": 0, "y": 0, "z": 0},
            "blocks": [
                {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, FACING_EAST)},
                {"x": -1, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},
                {"x": 1, "y": 0, "z": 0, "state": make_state(BLOCK_SLIME)},
                {"x": 1, "y": 0, "z": -1, "state": make_state(BLOCK_GLAZED_TERRACOTTA)},
            ],
        },
        # The slime pushed here also drags nothing (only the terracotta sits beside it), but
        # slime's own stickiness class + movement still trips the sticky_drag tag - an accurate
        # multi-tag board, not a fixture bug (see build_suite()'s exact-match assertion below).
        frozenset({"push_order", "non_stick", "sticky_drag", "composed"}),
    ),
    (
        "observer_pulse",
        {
            "trigger": {"x": 3, "y": 0, "z": 0},
            "blocks": [
                {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_OBSERVER, FACING_EAST)},
                {"x": 2, "y": 0, "z": 0, "state": make_state(BLOCK_STONE)},
                {"x": 3, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, FACING_WEST)},
                {"x": 4, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},
            ],
        },
        # Naturally composed: firing an observer requires a state change, here a piston push -
        # this fixture is the suite's "composed" ground truth too, not a separate build.
        frozenset({"push_order", "observer_pulse", "composed"}),
    ),
)


def build_suite() -> list[tuple[str, Sample, frozenset[str]]]:
    """Simulates every hand-built candidate and returns (name, encoded Sample, expected_tags).
    Raises AssertionError if a fixture's simulated log no longer exercises its claimed tag(s) -
    this suite is verification code, so a silently-drifted fixture must fail loud, not warn."""
    out: list[tuple[str, Sample, frozenset[str]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, candidate, expected_tags in _CANDIDATES:
            candidate = dict(candidate, id=1)
            log = simulate(candidate, workdir=workdir)
            log_path = workdir / f"{name}.simlog"
            log_path.write_bytes(log)
            sample = encode(log_path)
            actual_tags = derive_tags(sample)
            assert actual_tags == expected_tags, (
                f"{name}: expected tags {sorted(expected_tags)}, got {sorted(actual_tags)}"
            )
            out.append((name, sample, expected_tags))
    return out
