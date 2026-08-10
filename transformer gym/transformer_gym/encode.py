"""Turns one decoded .simlog into fixed-shape tensors: node features, typed relation edges,
and training targets. See transformer gym/README-less plan (session history) for the full
rationale; short version: hand the model precomputed relations instead of making it learn to
count/scan, and separate the STATIC picture (what the model sees) from the DYNAMIC outcome
(what happened) so targets can't leak into inputs.

Relations are a sparse edge list (`rel_edges: [3, E]`), not a dense [N,N] matrix - measured edge
counts are 549-535,733x smaller than the dense form (edge count actually FALLS as N grows: big
fixtures are mostly disconnected structure), and the dense form is what made 21 of 55 fixtures
un-encodable at all (some needing tens of GB for one relation tensor). `build_graph()` is the
reusable half of this - transformer_gym/state.py's incremental per-tick state builder calls it too,
so the node/relation construction logic exists in exactly one place.

Skipped in this first pass (ship the small thing, add when a real training run needs it):
  - y_cause (pointer-over-tokens head) - needs its own attention-pointer training loop.
  - per-tick event timing - y_event_grid collapses time into "did this kind ever happen to
    this block", not a [T x kinds] grid. Coarser, but a much smaller/simpler target to start.
"""
from __future__ import annotations

from dataclasses import dataclass
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

# "self" is a synthetic relation, not a physics relation: a self-loop on every node so
# edge-restricted attention (model.py) always has at least one key to attend to, even for a node
# with no sticks_to/observes/would_power/same_push_group edge at all.
RELATION_TYPES = ("sticks_to", "observes", "would_power_direct", "would_power_qc",
                  "same_push_group", "self")


class TooManyNodesError(ValueError):
    """Raised when a fixture's node count exceeds the safety valve (see dataset.MAX_NODES) -
    should not trigger under normal operation now that relations are sparse; kept as a guard
    against pathological inputs, not a real capacity ceiling."""


@dataclass
class Graph:
    """The static, t=0 half of a Sample - node identity/features plus the sparse relation edges.
    Shared by encode() (whole-run aggregate targets) and transformer_gym/state.py's
    initial_state() (the first SimState of a per-tick recurrent rollout), so this construction
    logic lives in exactly one place."""
    n: int
    block_type: torch.Tensor      # [N] long
    facing: torch.Tensor          # [N] long, 0-5 or NO_FACING
    flags: torch.Tensor           # [N, 3] float: extended, powered, open
    movability: torch.Tensor      # [N] long
    stickiness: torch.Tensor      # [N] long
    is_trigger: torch.Tensor      # [N] float
    is_air: torch.Tensor          # [N] float (interface-air synthetic tokens)
    rel_pos: torch.Tensor         # [N, 3] float, position minus bbox-min, /8 (matches relation bucketing scale)
    rel_edges: torch.Tensor       # [3, E] long: (relation_type_idx, src, dst)
    key_to_idx: dict[int, int]    # stableKey -> node index, solid blocks only (air has no key)
    initial: list                 # the InitialBlockState list, for target-building reuse


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
    rel_edges: torch.Tensor       # [3, E] long: (relation_type_idx, src, dst)
    # targets
    y_moves: torch.Tensor         # [N] float in {0,1}: did this block end up displaced
    y_stays_attached: torch.Tensor  # [N] float in {0,1}: preserved offset within its t=0 component
    y_event_grid: torch.Tensor    # [N, N_KINDS] float in {0,1}
    y_net_shift: torch.Tensor     # [3] float
    y_valid_cycle: torch.Tensor   # [] float in {0,1}
    y_termination: torch.Tensor   # [] long


def _facing_or_none(f: int) -> int:
    return f if 0 <= f <= 5 else NO_FACING


def build_graph(data: bytes, footer: dict, max_nodes: int) -> Graph:
    """The static node/relation half of a Sample - see the Graph docstring for why this is
    factored out. Reads only t=0 sections; no event stream access here."""
    initial = read_initial_state(data, footer)
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

    air_positions = sorted(air_cells)
    n = len(initial) + len(air_positions)
    if n > max_nodes:
        raise TooManyNodesError(f"{n} nodes > max_nodes={max_nodes}")

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

    rel_idx = {name: i for i, name in enumerate(RELATION_TYPES)}
    edge_type: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []

    def add_edge(rtype: str, a: int, b: int) -> None:
        edge_type.append(rel_idx[rtype])
        edge_src.append(a)
        edge_dst.append(b)

    # sticks_to: all pairs within the same t=0 sticky component.
    for c in components:
        members = [key_to_idx[comp_members[c.memberOffset + i]] for i in range(c.memberCount)]
        for a in members:
            for b in members:
                if a != b:
                    add_edge("sticks_to", a, b)

    # observes: observer -> the block it senses (opposite of its facing side), purely geometric.
    for s in initial:
        if s.blockTypeId != BLOCK_OBSERVER or s.facing > 5:
            continue
        dx, dy, dz = FACING_OFFSET[s.facing]
        front = (s.x - dx, s.y - dy, s.z - dz)  # opposite(facing)
        j = pos_to_idx.get(front)
        if j is not None:
            add_edge("observes", key_to_idx[s.stableKey], j)

    # would_power: static power relation (direct vs. QC get separate relation channels so the
    # model isn't forced to conflate "adjacent redstone" with "QC through the block above").
    for wp in would_power:
        src, dst = key_to_idx.get(wp.sourceKey), key_to_idx.get(wp.pistonKey)
        if src is None or dst is None:
            continue
        add_edge("would_power_qc" if wp.viaQC else "would_power_direct", src, dst)

    # same_push_group: static preview only (never the dynamic runtime push-groups - those are
    # part of the OUTCOME we're trying to predict, not something known at t=0).
    for g in static_groups:
        members = [key_to_idx[static_members[g.memberOffset + i]] for i in range(g.memberCount)]
        for a in members:
            for b in members:
                if a != b:
                    add_edge("same_push_group", a, b)

    # Guarantees every node (including ones with no physics relation at all) has >=1 incoming
    # edge, so edge-restricted attention (model.py) never has to attend over an empty key set.
    for i in range(n):
        add_edge("self", i, i)

    rel_edges = torch.stack([
        torch.tensor(edge_type, dtype=torch.long),
        torch.tensor(edge_src, dtype=torch.long),
        torch.tensor(edge_dst, dtype=torch.long),
    ])

    return Graph(
        n=n, block_type=block_type, facing=facing, flags=flags, movability=movability,
        stickiness=stickiness, is_trigger=is_trigger, is_air=is_air, rel_pos=rel_pos,
        rel_edges=rel_edges, key_to_idx=key_to_idx, initial=initial,
    )


def encode(log_path: Path, max_nodes: int = 300) -> Sample:
    data = log_path.read_bytes()
    footer = read_footer(data)
    graph = build_graph(data, footer, max_nodes=max_nodes)
    index = read_block_index(data, footer)
    summary = read_summary(data, footer)

    n = graph.n
    key_to_idx = graph.key_to_idx

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
    components = read_components(data, footer)
    comp_members = read_component_members(data, footer)
    for c in components:
        members = [comp_members[c.memberOffset + i] for i in range(c.memberCount)]
        if len(members) < 2:
            continue
        orig0 = next(s for s in graph.initial if s.stableKey == members[0])
        final0 = final_pos.get(members[0], (orig0.x, orig0.y, orig0.z))
        for key in members:
            orig = next(s for s in graph.initial if s.stableKey == key)
            final = final_pos.get(key, (orig.x, orig.y, orig.z))
            orig_off = (orig.x - orig0.x, orig.y - orig0.y, orig.z - orig0.z)
            final_off = (final[0] - final0[0], final[1] - final0[1], final[2] - final0[2])
            y_stays_attached[key_to_idx[key]] = float(orig_off == final_off)

    y_net_shift = torch.tensor(summary.netShift, dtype=torch.float32)
    y_valid_cycle = torch.tensor(float(summary.validCycle))
    y_termination = torch.tensor(summary.terminationReason, dtype=torch.long)

    return Sample(
        name=log_path.stem, block_type=graph.block_type, facing=graph.facing, flags=graph.flags,
        movability=graph.movability, stickiness=graph.stickiness, is_trigger=graph.is_trigger,
        is_air=graph.is_air, rel_pos=graph.rel_pos, rel_edges=graph.rel_edges,
        y_moves=y_moves, y_stays_attached=y_stays_attached, y_event_grid=y_event_grid,
        y_net_shift=y_net_shift, y_valid_cycle=y_valid_cycle, y_termination=y_termination,
    )
