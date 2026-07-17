"""Placeholder Task: exercises every seam (contexts, features, candidate-building,
simulator-backed reward) with the least logic possible, so the training base can be verified
correct before a real objective is layered on. Bias-only features mean the policy can only ever
learn one global "does adding slime anywhere tend to work" rate - that's expected, not a bug.

Meant to be replaced by rl_ml/tasks/slime_extension.py (see the plan's "Future task" section for
the real objective this Task protocol was built for).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import BLOCK_SLIME, make_state
from genetic_ml.mutation import _FACING_OFFSETS

Candidate = dict[str, Any]


@dataclass(frozen=True)
class DummyContext:
    machine: Candidate
    position: tuple[int, int, int]


def _candidate_positions(machine: Candidate) -> list[tuple[int, int, int]]:
    occupied = {(b["x"], b["y"], b["z"]) for b in machine["blocks"]}
    positions: set[tuple[int, int, int]] = set()
    for block in machine["blocks"]:
        for dx, dy, dz in _FACING_OFFSETS:
            pos = (block["x"] + dx, block["y"] + dy, block["z"] + dz)
            if pos not in occupied:
                positions.add(pos)
    return sorted(positions)


class DummyTask:
    def sample_contexts(
        self, base_machines: list[Candidate], rng: random.Random, count: int
    ) -> list[DummyContext]:
        contexts = []
        for _ in range(count):
            machine = rng.choice(base_machines)
            positions = _candidate_positions(machine)
            if not positions:
                continue
            contexts.append(DummyContext(machine=machine, position=rng.choice(positions)))
        return contexts

    def features(self, context: DummyContext) -> list[float]:
        return [1.0]  # bias only - deliberately no structural feature engineering

    def build_candidate(self, context: DummyContext, action: bool, candidate_id: int) -> Candidate | None:
        if not action:
            return None
        x, y, z = context.position
        blocks = [dict(block) for block in context.machine["blocks"]]
        blocks.append({"x": x, "y": y, "z": z, "state": make_state(BLOCK_SLIME)})
        return {"id": candidate_id, "trigger": dict(context.machine["trigger"]), "blocks": blocks}

    def reward_of(self, action: bool, result: dict[str, Any] | None) -> float:
        if not action or result is None:
            return 0.0
        return 1.0 if result.get("working") is True else -1.0
