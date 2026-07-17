"""Real training objective: learn where a block from a small candidate palette (slime today, more
types later) can be attached to a working flying machine without breaking it.

How it works, in plain terms:
    For each seed machine, this looks at empty spots right next to existing blocks and asks
    "should I glue a block here?" A spot to consider is picked at random - nothing smart happens
    there, it's just "here's something to try." To decide yes/no, it looks at what's immediately
    next to that spot (a piston facing this way? an existing slime block? a redstone block? an
    observer?) and turns that into a handful of yes/no flags. Each flag has a learned weight (how
    much it should push toward "yes, attach here"); the flags that are true get summed into a
    probability, and the policy (rl_ml/policy.py) samples an actual yes/no from that. If it tries a
    spot and the machine still works afterward, the flags that were true there get nudged to be
    more convincing next time; if the machine breaks, they get nudged down. Across many tries on
    many machines, the weights settle into a general rule ("yes near a piston's push face, no near
    a redstone block") rather than a memorized list of coordinates - because the weights belong to
    the flags, not to any one machine or position.

    Caveat: this is a flat linear model, so the same flag weights apply to every block type in
    candidate_block_ids - it can only learn "this flag matters," not "this flag matters for slime
    but not for redstone." That only becomes a real limitation once more block types are added.

One shared, parameterized Task instead of one file per block type - every "attach block type X
here, still working?" question is the same decision shape (pick a position, pick a block type,
test one addition, get {0,+1,-1}), so a single class covers today's slime-only goal
(candidate_block_ids=[BLOCK_SLIME]) and later multi-block-type goals (extend the list) without
touching train_loop.py or policy.py - see the plan's "Next task" section for the full rationale.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
    block_id,
    block_meta,
    make_state,
)

from rl_ml.positions import candidate_positions, neighbor_states

Candidate = dict[str, Any]

_PISTON_BLOCK_IDS = {BLOCK_PISTON, BLOCK_STICKY_PISTON}


@dataclass(frozen=True)
class BlockAttachmentContext:
    machine: Candidate
    position: tuple[int, int, int]
    block_id: int


def _is_push_face(direction_index: int, facing: int) -> bool:
    # _FACING_OFFSETS pairs opposite directions as (0,1), (2,3), (4,5) - direction_index ^ 1 is the
    # opposite direction, i.e. the one pointing back from a neighbor toward the candidate position.
    # A piston's front face points at that position exactly when its facing equals that opposite.
    return facing == (direction_index ^ 1)


class BlockAttachmentTask:
    def __init__(self, candidate_block_ids: list[int] | None = None) -> None:
        self.candidate_block_ids = candidate_block_ids if candidate_block_ids is not None else [BLOCK_SLIME]

    def sample_contexts(
        self, base_machines: list[Candidate], rng: random.Random, count: int
    ) -> list[BlockAttachmentContext]:
        contexts = []
        for _ in range(count):
            machine = rng.choice(base_machines)
            positions = candidate_positions(machine)
            if not positions:
                continue
            position = rng.choice(positions)
            block = rng.choice(self.candidate_block_ids)
            contexts.append(BlockAttachmentContext(machine=machine, position=position, block_id=block))
        return contexts

    def features(self, context: BlockAttachmentContext) -> list[float]:
        push_face = 0.0
        other_piston_face = 0.0
        near_slime = 0.0
        near_redstone = 0.0
        near_observer = 0.0

        for direction_index, state in neighbor_states(context.machine, context.position):
            bid = block_id(state)
            if bid in _PISTON_BLOCK_IDS:
                facing = block_meta(state) & 0b111
                if _is_push_face(direction_index, facing):
                    push_face = 1.0
                else:
                    other_piston_face = 1.0
            elif bid == BLOCK_SLIME:
                near_slime = 1.0
            elif bid == BLOCK_REDSTONE_BLOCK:
                near_redstone = 1.0
            elif bid == BLOCK_OBSERVER:
                near_observer = 1.0

        block_type_one_hot = [
            1.0 if candidate == context.block_id else 0.0 for candidate in self.candidate_block_ids
        ]
        return [1.0, push_face, other_piston_face, near_slime, near_redstone, near_observer, *block_type_one_hot]

    def build_candidate(self, context: BlockAttachmentContext, action: bool, candidate_id: int) -> Candidate | None:
        if not action:
            return None
        x, y, z = context.position
        blocks = [dict(block) for block in context.machine["blocks"]]
        blocks.append({"x": x, "y": y, "z": z, "state": make_state(context.block_id)})
        return {"id": candidate_id, "trigger": dict(context.machine["trigger"]), "blocks": blocks}

    def reward_of(self, action: bool, result: dict[str, Any] | None) -> float:
        if not action or result is None:
            return 0.0
        return 1.0 if result.get("working") is True else -1.0
