"""The game definition - md/ALPHAZERO.md Part 2.

Five things and nothing else, because MCTS calls nothing else:

    state       (base machine's record, set of placements)
    actions     (cell, palette entry), plus a reserved masked stop
    transition  write the placement into the set - deterministic, free, no simulator call
    terminal    len(placements) == k
    reward      build, hash, simulate if new, validCycle as a float

The reward is deliberately a float everywhere and never a bool: it becomes a graded 0-1 value
later by means not yet decided, and keeping the type right now makes that change free.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable

from rlgym.blocks import FACING_OFFSETS, PALETTE, PALETTE_SLOTS, PaletteEntry

Cell = tuple[int, int, int]
Candidate = dict[str, Any]

# Face-step neighbours: one step changes exactly one coordinate by one. Deliberately NOT the
# 26-neighbour Chebyshev box - a diagonal-only cell touches nothing, so it can never be dragged
# or pushed, and including it would add actions that are always wrong.
_FACE_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

DEFAULT_SHELL_RADIUS = 1
DEFAULT_K = 1


def blocks_to_map(blocks: Iterable[dict[str, int]]) -> dict[Cell, int]:
    return {(b["x"], b["y"], b["z"]): b["state"] for b in blocks}


def map_to_blocks(cells: dict[Cell, int]) -> list[dict[str, int]]:
    """Sorted, so the block list is a function of the cell map alone - two machines that are
    the same set of cells always serialise identically, keeping encoding and hashing stable."""
    return [
        {"x": x, "y": y, "z": z, "state": state}
        for (x, y, z), state in sorted(cells.items())
    ]


def candidate_cells(cells: dict[Cell, int], radius: int = DEFAULT_SHELL_RADIUS) -> list[Cell]:
    """Every cell within radius face-steps of a machine block, sorted.

    Measured on the tick-0 layout: a placement edits the starting configuration, so where the
    machine later sweeps does not affect legality. That the swept volume matters at all is
    information for the model, not a rule of the game.
    """
    if radius < 0:
        raise ValueError("shell radius must be >= 0")
    frontier = set(cells)
    reached = set(frontier)
    for _ in range(radius):
        nxt = set()
        for (x, y, z) in frontier:
            for dx, dy, dz in _FACE_STEPS:
                cell = (x + dx, y + dy, z + dz)
                if cell not in reached:
                    reached.add(cell)
                    nxt.add(cell)
        frontier = nxt
    return sorted(reached)


@dataclass(frozen=True)
class Placement:
    """One action: set this cell to this palette entry."""

    cell: Cell
    slot: int

    @property
    def entry(self) -> PaletteEntry:
        entry = PALETTE[self.slot]
        if entry is None:
            raise ValueError(f"palette slot {self.slot} is reserved and has no entry")
        return entry

    def written_cells(self) -> tuple[Cell, ...]:
        """The cells this action writes - two for an extended piston, one for anything else."""
        entry = self.entry
        if entry.head is None:
            return (self.cell,)
        dx, dy, dz = FACING_OFFSETS[entry.facing]
        x, y, z = self.cell
        return (self.cell, (x + dx, y + dy, z + dz))


def apply_placement(cells: dict[Cell, int], placement: Placement) -> dict[Cell, int]:
    """A new cell map with the placement written in. Air removes the cell rather than storing
    state 0, so the block list always describes only what is actually present."""
    out = dict(cells)
    entry = placement.entry
    targets = placement.written_cells()
    if entry.is_air:
        out.pop(targets[0], None)
        return out
    out[targets[0]] = entry.state
    if entry.head is not None:
        out[targets[1]] = entry.head_state
    return out


def is_no_op(cells: dict[Cell, int], placement: Placement) -> bool:
    """True when the placement leaves the machine unchanged.

    These must be masked, or the model learns they always succeed - which is true, and
    useless (DECISIONS.md point 12).
    """
    return apply_placement(cells, placement) == cells


def legal_mask(
    cells: dict[Cell, int],
    cell_list: list[Cell],
    written: frozenset[Cell] = frozenset(),
) -> list[bool]:
    """One flag per action, in action order: cell_list x PALETTE_SLOTS, then the stop slot.

    Masked: reserved palette slots; stop; no-ops; and any action touching a cell already
    written this episode - BOTH cells for an extended piston. That last rule is what makes the
    set of placements order-independent, which kills point 23's factorial duplicate explosion
    structurally instead of catching it with a hash afterwards.
    """
    mask = [False] * (len(cell_list) * PALETTE_SLOTS + 1)
    for cell_index, cell in enumerate(cell_list):
        base = cell_index * PALETTE_SLOTS
        for slot in range(PALETTE_SLOTS):
            if PALETTE[slot] is None:
                continue
            placement = Placement(cell, slot)
            if written and any(c in written for c in placement.written_cells()):
                continue
            if is_no_op(cells, placement):
                continue
            mask[base + slot] = True
    # mask[-1] is stop: reserved and always masked while k is a fixed constant. The slot exists
    # so that unmasking it later is not a change to the policy head's shape.
    return mask


def action_index(cell_index: int, slot: int) -> int:
    return cell_index * PALETTE_SLOTS + slot


def action_count(cell_list: list[Cell]) -> int:
    return len(cell_list) * PALETTE_SLOTS + 1


def decode_action(index: int, cell_list: list[Cell]) -> Placement:
    if index == len(cell_list) * PALETTE_SLOTS:
        raise ValueError("index is the stop action, which has no placement")
    return Placement(cell_list[index // PALETTE_SLOTS], index % PALETTE_SLOTS)


# --- candidate serialisation -------------------------------------------------------------
#
# Copied from genetic_ml/compact_format.py rather than imported (self-containment, point 26).
# The layout is the wire protocol the C++ side reads in readCandidateCompact (json_stream.h),
# so it is a contract, not an implementation detail:
#
#     int32 id | int32 trigger_x, trigger_y, trigger_z | uint32 block_count
#     block_count x { int32 x, int32 y, int32 z | uint32 state }

_HEADER = struct.Struct("<iiiiI")
_BLOCK = struct.Struct("<iiiI")


def encode_candidate(candidate: Candidate) -> bytes:
    trigger = candidate["trigger"]
    blocks = candidate["blocks"]
    out = bytearray(_HEADER.size + _BLOCK.size * len(blocks))
    _HEADER.pack_into(
        out, 0, candidate["id"], trigger["x"], trigger["y"], trigger["z"], len(blocks)
    )
    offset = _HEADER.size
    for block in blocks:
        _BLOCK.pack_into(out, offset, block["x"], block["y"], block["z"], block["state"])
        offset += _BLOCK.size
    return bytes(out)


def decode_candidate(stream: BinaryIO) -> Candidate | None:
    header = stream.read(_HEADER.size)
    if not header:
        return None
    if len(header) < _HEADER.size:
        raise EOFError("truncated compact-format header")
    cid, tx, ty, tz, block_count = _HEADER.unpack(header)
    blocks: list[dict[str, int]] = []
    for _ in range(block_count):
        raw = stream.read(_BLOCK.size)
        if len(raw) < _BLOCK.size:
            raise EOFError("truncated compact-format block record")
        x, y, z, state = _BLOCK.unpack(raw)
        blocks.append({"x": x, "y": y, "z": z, "state": state})
    return {"id": cid, "trigger": {"x": tx, "y": ty, "z": tz}, "blocks": blocks}


def canonical_hash(candidate: Candidate) -> str:
    """Structure hash, independent of block order, id and absolute position.

    Byte-identical to genetic_ml/population.py's version, so hashes computed here agree with
    every hash already recorded elsewhere in the project. Deliberately NOT rotation-invariant:
    rotating a machine changes which block is the anchor (simulator.cpp:2052 orders by y, then
    z, then x) and changes neighbour update order, so rotations genuinely can behave
    differently and are different machines (point 23).
    """
    blocks = candidate["blocks"]
    if blocks:
        min_x = min(block["x"] for block in blocks)
        min_y = min(block["y"] for block in blocks)
        min_z = min(block["z"] for block in blocks)
    else:
        min_x = min_y = min_z = 0

    normalized_blocks = sorted(
        (block["x"] - min_x, block["y"] - min_y, block["z"] - min_z, block["state"])
        for block in blocks
    )
    trigger = candidate["trigger"]
    normalized_trigger = (
        trigger["x"] - min_x,
        trigger["y"] - min_y,
        trigger["z"] - min_z,
    )
    payload = json.dumps([normalized_trigger, normalized_blocks], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- the state ---------------------------------------------------------------------------


@dataclass
class Machine:
    """A base machine: its blocks, its trigger, and the candidate cells derived from them."""

    cells: dict[Cell, int]
    trigger: Cell
    radius: int = DEFAULT_SHELL_RADIUS
    cell_list: list[Cell] = field(init=False)

    def __post_init__(self) -> None:
        self.cell_list = candidate_cells(self.cells, self.radius)

    @classmethod
    def from_candidate(
        cls, candidate: Candidate, radius: int = DEFAULT_SHELL_RADIUS
    ) -> "Machine":
        trigger = candidate["trigger"]
        return cls(
            cells=blocks_to_map(candidate["blocks"]),
            trigger=(trigger["x"], trigger["y"], trigger["z"]),
            radius=radius,
        )

    def to_candidate(self, cells: dict[Cell, int] | None = None, cid: int = 0) -> Candidate:
        return {
            "id": cid,
            "trigger": {"x": self.trigger[0], "y": self.trigger[1], "z": self.trigger[2]},
            "blocks": map_to_blocks(self.cells if cells is None else cells),
        }

    @property
    def action_count(self) -> int:
        return action_count(self.cell_list)


@dataclass
class GameState:
    """(base machine, set of placements). The base is fixed for the whole episode."""

    machine: Machine
    placements: tuple[Placement, ...] = ()
    k: int = DEFAULT_K

    @property
    def is_terminal(self) -> bool:
        return len(self.placements) >= self.k

    @property
    def written(self) -> frozenset[Cell]:
        out: set[Cell] = set()
        for placement in self.placements:
            out.update(placement.written_cells())
        return frozenset(out)

    def current_cells(self) -> dict[Cell, int]:
        cells = self.machine.cells
        for placement in self.placements:
            cells = apply_placement(cells, placement)
        return cells

    def legal_mask(self) -> list[bool]:
        if self.is_terminal:
            return [False] * self.machine.action_count
        return legal_mask(self.current_cells(), self.machine.cell_list, self.written)

    def step(self, placement: Placement) -> "GameState":
        """Deterministic, free, no simulator call - the whole point of the note mechanism."""
        if self.is_terminal:
            raise ValueError("cannot place on a terminal state")
        return GameState(self.machine, self.placements + (placement,), self.k)

    def to_candidate(self, cid: int = 0) -> Candidate:
        return self.machine.to_candidate(self.current_cells(), cid)
