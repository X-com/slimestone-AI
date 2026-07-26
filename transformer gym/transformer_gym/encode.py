"""Turns one decoded .simlog into fixed-shape tensors: node features, typed relation edges,
and training targets. See transformer gym/README-less plan (session history) for the full
rationale; short version: hand the model precomputed relations instead of making it learn to
count/scan, and separate the STATIC picture (what the model sees) from the DYNAMIC outcome
(what happened) so targets can't leak into inputs.

Skipped in this first pass (ship the small thing, add when a real training run needs it):
  - y_cause (pointer-over-tokens head) - needs its own attention-pointer training loop.
  - per-tick event timing - y_event_grid collapses time into "did this kind ever happen to
    this block", not a [T x kinds] grid. Coarser, but a much smaller/simpler target to start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .simlog_reader import (
    KIND_NAMES,
    iter_block_events,
    read_block_index,
    read_component_members,
    read_components,
    read_footer,
    read_initial_state,
    read_static_push_members,
    read_static_push_preview,
    read_summary,
    read_would_power,
    unpack_pos,
)

N_KINDS = len(KIND_NAMES)  # 16
FACING_OFFSET = {0: (0, -1, 0), 1: (0, 1, 0), 2: (0, 0, -1), 3: (0, 0, 1), 4: (-1, 0, 0), 5: (1, 0, 0)}
BLOCK_OBSERVER = 218
BLOCK_VOCAB = 256  # block type ids are a single byte in practice
NO_FACING = 6      # facing field is 0xFF ("none") for non-directional blocks

RELATION_TYPES = ("sticks_to", "observes", "would_power_direct", "would_power_qc", "same_push_group")


class TooManyNodesError(ValueError):
    """Raised when a fixture's node count would make the dense [N,N] relation tensor too big."""


@dataclass
class Sample:
    name: str
    # node features
    block_type: torch.Tensor      # [N] long
    facing: torch.Tensor          # [N] long, 0-5 or NO_FACING
    flags: torch.Tensor           # [N, 3] float: extended, powered, open
    movability: torch.Tensor      # [N] long
    stickiness: torch.Tensor      # [N] long
    is_trigger: torch.Tensor      # [N] float
    is_air: torch.Tensor          # [N] float (interface-air synthetic tokens)
    rel_pos: torch.Tensor         # [N, 3] float, position minus bbox-min, /8 (matches relation bucketing scale)
    # relations: one [N, N] float adjacency per type in RELATION_TYPES
    relations: torch.Tensor       # [len(RELATION_TYPES), N, N]
    # targets
    y_moves: torch.Tensor         # [N] float in {0,1}: did this block end up displaced
    y_stays_attached: torch.Tensor  # [N] float in {0,1}: preserved offset within its t=0 component
    y_event_grid: torch.Tensor    # [N, N_KINDS] float in {0,1}
    y_net_shift: torch.Tensor     # [3] float
    y_valid_cycle: torch.Tensor   # [] float in {0,1}
    y_termination: torch.Tensor   # [] long


def _facing_or_none(f: int) -> int:
    return f if 0 <= f <= 5 else NO_FACING


def encode(log_path: Path, max_nodes: int = 300) -> Sample:
    data = log_path.read_bytes()
    footer = read_footer(data)
    initial = read_initial_state(data, footer)
    index = read_block_index(data, footer)
    components = read_components(data, footer)
    comp_members = read_component_members(data, footer)
    would_power = read_would_power(data, footer)
    static_groups = read_static_push_preview(data, footer)
    static_members = read_static_push_members(data, footer)
    summary = read_summary(data, footer)

    solid = {(s.x, s.y, s.z): s for s in initial}
    bbox_min = summary.bboxMin

    # Interface-air tokens (gap #3): empty cells adjacent to a solid block, within the run's bbox.
    air_cells: set[tuple[int, int, int]] = set()
    for s in initial:
        for dx, dy, dz in FACING_OFFSET.values():
            cell = (s.x + dx, s.y + dy, s.z + dz)
            if cell not in solid:
                air_cells.add(cell)

    nodes = list(initial) + [None] * len(air_cells)  # None marks a synthetic air node
    air_positions = sorted(air_cells)
    n = len(initial) + len(air_positions)
    if n > max_nodes:
        raise TooManyNodesError(f"{log_path.stem}: {n} nodes > max_nodes={max_nodes}")

    key_to_idx: dict[int, int] = {s.stableKey: i for i, s in enumerate(initial)}
    pos_to_idx: dict[tuple[int, int, int], int] = {(s.x, s.y, s.z): i for i, s in enumerate(initial)}
    for j, pos in enumerate(air_positions):
        pos_to_idx[pos] = len(initial) + j

    block_type = torch.zeros(n, dtype=torch.long)
    facing = torch.full((n,), NO_FACING, dtype=torch.long)
    flags = torch.zeros(n, 3)
    movability = torch.zeros(n, dtype=torch.long)
    stickiness = torch.zeros(n, dtype=torch.long)
    is_trigger = torch.zeros(n)
    is_air = torch.zeros(n)
    rel_pos = torch.zeros(n, 3)

    for i, s in enumerate(initial):
        block_type[i] = min(s.blockTypeId, BLOCK_VOCAB - 1)
        facing[i] = _facing_or_none(s.facing)
        flags[i, 0] = float(s.stateFlags & 1)
        flags[i, 1] = float((s.stateFlags >> 1) & 1)
        flags[i, 2] = float((s.stateFlags >> 2) & 1)
        movability[i] = s.movabilityClass
        stickiness[i] = s.stickinessClass
        is_trigger[i] = float(s.isTrigger)
        rel_pos[i] = torch.tensor([s.x - bbox_min[0], s.y - bbox_min[1], s.z - bbox_min[2]]) / 8.0
    for j, (x, y, z) in enumerate(air_positions):
        idx = len(initial) + j
        is_air[idx] = 1.0
        rel_pos[idx] = torch.tensor([x - bbox_min[0], y - bbox_min[1], z - bbox_min[2]]) / 8.0

    relations = torch.zeros(len(RELATION_TYPES), n, n)
    rel_idx = {name: i for i, name in enumerate(RELATION_TYPES)}

    # sticks_to: all pairs within the same t=0 sticky component.
    for c in components:
        members = [key_to_idx[comp_members[c.memberOffset + i]] for i in range(c.memberCount)]
        r = rel_idx["sticks_to"]
        for a in members:
            for b in members:
                if a != b:
                    relations[r, a, b] = 1.0

    # observes: observer -> the block it senses (opposite of its facing side), purely geometric.
    for s in initial:
        if s.blockTypeId != BLOCK_OBSERVER or s.facing > 5:
            continue
        dx, dy, dz = FACING_OFFSET[s.facing]
        front = (s.x - dx, s.y - dy, s.z - dz)  # opposite(facing)
        j = pos_to_idx.get(front)
        if j is not None:
            relations[rel_idx["observes"], key_to_idx[s.stableKey], j] = 1.0

    # would_power: static power relation (direct vs. QC get separate relation channels so the
    # model isn't forced to conflate "adjacent redstone" with "QC through the block above").
    for wp in would_power:
        src, dst = key_to_idx.get(wp.sourceKey), key_to_idx.get(wp.pistonKey)
        if src is None or dst is None:
            continue
        r = rel_idx["would_power_qc"] if wp.viaQC else rel_idx["would_power_direct"]
        relations[r, src, dst] = 1.0

    # same_push_group: static preview only (never the dynamic runtime push-groups - those are
    # part of the OUTCOME we're trying to predict, not something known at t=0).
    r = rel_idx["same_push_group"]
    for g in static_groups:
        members = [key_to_idx[static_members[g.memberOffset + i]] for i in range(g.memberCount)]
        for a in members:
            for b in members:
                if a != b:
                    relations[r, a, b] = 1.0

    # --- targets ---
    y_moves = torch.zeros(n)
    y_event_grid = torch.zeros(n, N_KINDS)
    for entry in index:
        i = key_to_idx.get(entry.originalKey)
        if i is None:
            continue
        y_moves[i] = float(entry.currentKey != entry.originalKey)
        for ev in iter_block_events(data, entry):
            if 0 <= ev.kind < N_KINDS:
                y_event_grid[i, ev.kind] = 1.0

    final_pos = {e.originalKey: unpack_pos(e.currentKey) for e in index}
    y_stays_attached = torch.ones(n)  # blocks with no component record default to "trivially attached"
    for c in components:
        members = [comp_members[c.memberOffset + i] for i in range(c.memberCount)]
        if len(members) < 2:
            continue
        orig0 = next(s for s in initial if s.stableKey == members[0])
        final0 = final_pos.get(members[0], (orig0.x, orig0.y, orig0.z))
        for key in members:
            orig = next(s for s in initial if s.stableKey == key)
            final = final_pos.get(key, (orig.x, orig.y, orig.z))
            orig_off = (orig.x - orig0.x, orig.y - orig0.y, orig.z - orig0.z)
            final_off = (final[0] - final0[0], final[1] - final0[1], final[2] - final0[2])
            y_stays_attached[key_to_idx[key]] = float(orig_off == final_off)

    y_net_shift = torch.tensor(summary.netShift, dtype=torch.float32)
    y_valid_cycle = torch.tensor(float(summary.validCycle))
    y_termination = torch.tensor(summary.terminationReason, dtype=torch.long)

    return Sample(
        name=log_path.stem, block_type=block_type, facing=facing, flags=flags,
        movability=movability, stickiness=stickiness, is_trigger=is_trigger, is_air=is_air,
        rel_pos=rel_pos, relations=relations, y_moves=y_moves, y_stays_attached=y_stays_attached,
        y_event_grid=y_event_grid, y_net_shift=y_net_shift, y_valid_cycle=y_valid_cycle,
        y_termination=y_termination,
    )
