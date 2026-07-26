"""Generator 1 (doc §3, the highest-value piece): forward-chain construction.

Never scatters blocks and hopes a chain emerges - each link is placed by reading ground truth
from the previous simulate() call: a root piston is guaranteed to fire (statically powered, a
movable block in front, space to push into), then an observer is attached to whichever cell just
changed, and a new piston is placed on the observer's output side so its own pulse re-fires the
chain one link deeper. Each link costs exactly one simulator call.

Attachment policy (doc §3's open design question) is resolved per-contraption by sampling which
prior changed cell the next link attaches to: `most_recent` -> long thin chains, `random_earlier`
-> branching trees, `multi_cell` -> the next link watches two prior cells at once (approximated
here as attaching two independent single-cell links off the same recent cell, since a single
observer only ever watches one front cell) -> converging graphs.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

import generator  # noqa: F401
from genetic_ml.blocks import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_STONE,
    FACING_DOWN,
    FACING_UP,
    make_state,
)
from geometry import FACING_OFFSETS, occupancy
from runner import SimRunError, simulate
from transformer_gym.simlog_reader import iter_block_events, read_block_index, read_footer, unpack_pos

Candidate = dict[str, Any]
Pos = tuple[int, int, int]

ATTACH_POLICIES = ("most_recent", "random_earlier", "multi_cell")


def _root_candidate() -> tuple[Candidate, Pos]:
    """Piston at origin facing UP, redstone block on its south face (a non-front side), a movable
    stone block directly above (front) to push, and empty space above that to push into. Starts
    retracted even though statically powered - Simulator::trigger's cold-kick path fires it on
    the first tick, producing a real PistonMoveExecuted/BlockPushed pair to attach the next link
    to (a pre-extended piston would already be "settled" and fire nothing)."""
    piston = {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, FACING_UP)}
    redstone = {"x": 0, "y": 0, "z": 1, "state": make_state(BLOCK_REDSTONE_BLOCK)}
    pushed = {"x": 0, "y": 1, "z": 0, "state": make_state(BLOCK_STONE)}
    blocks = [piston, redstone, pushed]
    return {"id": 0, "trigger": {"x": 0, "y": 0, "z": 0}, "blocks": blocks}, (0, 0, 0)


def _changed_cells(log: bytes) -> list[tuple[Pos, int]]:
    """(final position, last-event tick) for every block whose position moved during the run."""
    footer = read_footer(log)
    out = []
    for entry in read_block_index(log, footer):
        if entry.currentKey == entry.originalKey:
            continue
        last_tick = max((ev.activationTick for ev in iter_block_events(log, entry)), default=0)
        out.append((unpack_pos(entry.currentKey), last_tick))
    return out


def _attach_observer_and_piston(
    blocks: list[dict], attach_cell: Pos, rng: random.Random
) -> list[dict] | None:
    """Places an observer watching attach_cell, and a fresh unpowered piston on the observer's
    output side (so the observer's own pulse is what fires it next round) with a movable block
    ahead and empty space beyond. Returns the new block list, or None if no free orientation
    exists (crowded attach point - caller should try a different attach_cell).

    Observer geometry (verified against simulator.cpp, not assumed): the SENSED cell is
    offset(pos, facing) - the same direction as the observer's own meta, not the opposite - per
    observedNeighborChanged's `watched = offset(pos, facing)` check. The pulse OUTPUT (what
    notifyObserverFront pokes when it fires) is offset(pos, opposite(facing)), the far side. So
    to watch attach_cell C with meta `f`: observer sits at C - offset(f) (one cell behind C,
    away from it), and its output/driven piston sits one further cell behind that."""
    occ = occupancy(blocks)
    watch_facings = list(range(6))
    rng.shuffle(watch_facings)
    for watch_facing in watch_facings:
        odx, ody, odz = FACING_OFFSETS[watch_facing]
        observer_pos = (attach_cell[0] - odx, attach_cell[1] - ody, attach_cell[2] - odz)
        if observer_pos in occ:
            continue

        # Direction from the new piston to the observer is `watch_facing` (see docstring), so
        # the piston's own front must not be that direction (a piston is never powered via its
        # own front face).
        piston_facings = [f for f in range(6) if f != watch_facing]
        rng.shuffle(piston_facings)
        for piston_facing in piston_facings:
            piston_pos = (observer_pos[0] - odx, observer_pos[1] - ody, observer_pos[2] - odz)
            if piston_pos in occ or piston_pos == attach_cell:
                continue
            fdx, fdy, fdz = FACING_OFFSETS[piston_facing]
            ahead_pos = (piston_pos[0] + fdx, piston_pos[1] + fdy, piston_pos[2] + fdz)
            beyond_pos = (ahead_pos[0] + fdx, ahead_pos[1] + fdy, ahead_pos[2] + fdz)
            if ahead_pos in occ or beyond_pos in occ or ahead_pos == observer_pos or beyond_pos == observer_pos:
                continue

            new_blocks = list(blocks)
            new_blocks.append({"x": observer_pos[0], "y": observer_pos[1], "z": observer_pos[2],
                                "state": make_state(BLOCK_OBSERVER, watch_facing)})
            new_blocks.append({"x": piston_pos[0], "y": piston_pos[1], "z": piston_pos[2],
                                "state": make_state(BLOCK_PISTON, piston_facing)})
            new_blocks.append({"x": ahead_pos[0], "y": ahead_pos[1], "z": ahead_pos[2],
                                "state": make_state(BLOCK_STONE)})
            return new_blocks
    return None


def build_chain(rng: random.Random, target_depth: int, policy: str) -> Candidate | None:
    candidate, root_cell = _root_candidate()
    recent_cells: list[Pos] = [root_cell]

    for _ in range(target_depth):
        try:
            log = simulate(candidate)
        except SimRunError:
            return None
        changed = _changed_cells(log)
        if not changed:
            return None  # chain died (jammed, blocked, out of space) - not worth keeping

        if policy == "most_recent":
            attach_cell = max(changed, key=lambda c: c[1])[0]
        elif policy == "random_earlier":
            attach_cell = rng.choice(changed)[0]
        else:  # multi_cell: attach off two distinct recent cells if available (else falls back)
            pool = changed if len(changed) < 2 else rng.sample(changed, 2)
            attach_cell = pool[0][0]

        new_blocks = _attach_observer_and_piston(candidate["blocks"], attach_cell, rng)
        if new_blocks is None:
            return None
        candidate = {"id": candidate["id"], "trigger": candidate["trigger"], "blocks": new_blocks}

        if policy == "multi_cell" and len(changed) >= 2:
            second = _attach_observer_and_piston(candidate["blocks"], pool[1][0], rng)
            if second is not None:
                candidate["blocks"] = second

    return candidate


def generate(rng: random.Random) -> Iterator[Candidate]:
    while True:
        depth = rng.randint(1, 8)
        policy = rng.choice(ATTACH_POLICIES)
        result = build_chain(rng, depth, policy)
        if result is not None:
            yield result
