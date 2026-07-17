"""Placeholder Task: exercises every seam (contexts, features, candidate-building,
simulator-backed reward) with the least logic possible, so the training base can be verified
correct before a real objective is layered on. Bias-only features mean the policy can only ever
learn one global "does adding slime anywhere tend to work" rate - that's expected, not a bug.

Superseded by rl_ml/tasks/block_attachment.py as main_rl.py's default, but kept as a second Task
implementation - e.g. for the "does swapping the task actually leave train_loop.py/policy.py
untouched" check in the plan's verification section.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import BLOCK_SLIME, make_state

from rl_ml.positions import candidate_positions as _candidate_positions

Candidate = dict[str, Any]


@dataclass(frozen=True)
class DummyContext:
    machine: Candidate
    position: tuple[int, int, int]


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
