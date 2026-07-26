"""The jigsaw-piece library for gen_puzzle.py: a Tile is a small local-coordinate block group
plus typed sockets (see geometry.Socket) marking where it can be driven from ("in") and where it
produces an effect other tiles can attach to ("out"), plus which yaw rotations are legal to place
it at.

Every non-root tile is a self-contained "observer senses an external marker cell -> internally
powers its own piston -> piston pushes a block" unit. This is deliberate: a piston reads STATIC
power (redstone-block adjacency), which is a fundamentally different geometric relationship than
an observer SENSING a changed cell (verified the hard way - see gen_forward.py's docstring on
observer geometry). Rather than model two different socket-connection types, every tile bundles
its own observer+piston pair, so the *only* connection any tile ever needs is "sense an external
cell" - one proven formula, reused everywhere, including gen_forward.py's exact geometry:
sensed cell S -> observer sits at S - offset(watch_facing) -> its powered piston sits one more
cell further back (S - 2*offset(watch_facing)).

Two library sources, per the approved plan: a small hand-authored set below (explicit sockets,
guaranteed-clean joins) and `extract_tile`, which detects sockets automatically from a simulated
crop (grows the library from real redstone, no hand-authoring) - see the plan's "library-bounded
variety" corner: extraction is what keeps the hand-authored menu from being the whole story.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import generator  # noqa: F401
from genetic_ml.blocks import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STONE,
    FACING_EAST,
    FACING_NORTH,
    block_id,
    make_state,
)
from geometry import FACING_OFFSETS, Socket
from transformer_gym.simlog_reader import iter_block_events, read_block_index, read_footer, unpack_pos

Block = dict[str, Any]
Pos = tuple[int, int, int]

ALL_YAW = (0, 1, 2, 3)

_WATCH_FACING = FACING_NORTH   # fixed local sensing direction every hand-authored tile uses
_PISTON_FACING = FACING_EAST   # fixed local push direction (!= _WATCH_FACING, checked below)
assert _PISTON_FACING != _WATCH_FACING


def _observer_piston_positions(sense_marker: Pos) -> tuple[Pos, Pos, Pos, Pos]:
    """Given the local sense-marker cell S, returns (observer_pos, piston_pos, ahead_pos,
    landing_pos) using gen_forward.py's proven geometry: observer sits at S - offset(watch), its
    powered piston sits one cell further back, pushing whatever's directly ahead of it out by
    one more cell."""
    wdx, wdy, wdz = FACING_OFFSETS[_WATCH_FACING]
    observer_pos = (sense_marker[0] - wdx, sense_marker[1] - wdy, sense_marker[2] - wdz)
    piston_pos = (observer_pos[0] - wdx, observer_pos[1] - wdy, observer_pos[2] - wdz)
    pdx, pdy, pdz = FACING_OFFSETS[_PISTON_FACING]
    ahead_pos = (piston_pos[0] + pdx, piston_pos[1] + pdy, piston_pos[2] + pdz)
    landing_pos = (ahead_pos[0] + pdx, ahead_pos[1] + pdy, ahead_pos[2] + pdz)
    return observer_pos, piston_pos, ahead_pos, landing_pos


@dataclass
class Tile:
    name: str
    blocks: list[Block]
    sockets: list[Socket]
    allowed_orientations: tuple[int, ...] = ALL_YAW


def _root_piston() -> Tile:
    """Statically self-powered piston (redstone on a non-front face) pushing a stone block -
    identical setup to gen_forward.py's root, reused here as the one tile type that can start an
    assembly without any upstream driver. No `in` socket; one `out` socket at the landing cell."""
    blocks = [
        {"x": 0, "y": 0, "z": 0, "state": make_state(BLOCK_PISTON, _PISTON_FACING)},
        {"x": -1, "y": 0, "z": 0, "state": make_state(BLOCK_REDSTONE_BLOCK)},  # non-front face
        {"x": 1, "y": 0, "z": 0, "state": make_state(BLOCK_STONE)},
    ]
    sockets = [{"pos": (2, 0, 0), "facing": _PISTON_FACING, "kind": "out"}]
    return Tile("root_piston", blocks, sockets)


def _observer_piston_link(name: str, passenger_offset: Pos | None = None) -> Tile:
    """Sense marker S=(0,0,0) as the `in` socket; the pushed block's final rest cell as `out`.
    `passenger_offset`, if given, adds a slime passenger beside the pushed cell (dragged along
    when pushed) - the group-size surprise from gen_pushgroup.py, reusable as an assembly piece."""
    S = (0, 0, 0)
    observer_pos, piston_pos, ahead_pos, landing_pos = _observer_piston_positions(S)
    pushed_id = BLOCK_SLIME if passenger_offset is not None else BLOCK_STONE

    blocks = [
        {"x": observer_pos[0], "y": observer_pos[1], "z": observer_pos[2],
         "state": make_state(BLOCK_OBSERVER, _WATCH_FACING)},
        {"x": piston_pos[0], "y": piston_pos[1], "z": piston_pos[2],
         "state": make_state(BLOCK_PISTON, _PISTON_FACING)},
        {"x": ahead_pos[0], "y": ahead_pos[1], "z": ahead_pos[2], "state": make_state(pushed_id)},
    ]
    if passenger_offset is not None:
        px, py, pz = ahead_pos[0] + passenger_offset[0], ahead_pos[1] + passenger_offset[1], ahead_pos[2] + passenger_offset[2]
        blocks.append({"x": px, "y": py, "z": pz, "state": make_state(BLOCK_STONE)})

    sockets = [
        {"pos": S, "facing": _WATCH_FACING, "kind": "in"},
        {"pos": landing_pos, "facing": _PISTON_FACING, "kind": "out"},
    ]
    return Tile(name, blocks, sockets)


PRESET_TILES: tuple[Tile, ...] = (
    _root_piston(),
    _observer_piston_link("observer_piston_link"),
    _observer_piston_link("observer_piston_slime_link", passenger_offset=(0, 1, 0)),
)


def extract_tile(crop_blocks: list[Block], log: bytes, name: str = "extracted") -> Tile | None:
    """Auto-extracts sockets from a simulated crop: a block whose position moved becomes an
    `out` socket at its final position; an observer becomes an `in` socket at its *sensed* cell
    (pos + offset(meta) - the same direction as the observer's own facing meta, verified against
    simulator.cpp's observedNeighborChanged, not offset(opposite(meta))). Returns None if the
    crop produced neither (nothing useful to reuse as a tile)."""
    footer = read_footer(log)
    sockets: list[Socket] = []

    for entry in read_block_index(log, footer):
        if entry.currentKey == entry.originalKey:
            continue
        events = list(iter_block_events(log, entry))
        if not events:
            continue
        last = max(events, key=lambda e: e.activationTick)
        if last.kind != 2:  # BlockPushed
            continue
        dx = last.toX - last.fromX
        dy = last.toY - last.fromY
        dz = last.toZ - last.fromZ
        for facing, offset in enumerate(FACING_OFFSETS):
            if offset == (dx, dy, dz):
                sockets.append({"pos": unpack_pos(entry.currentKey), "facing": facing, "kind": "out"})
                break

    for b in crop_blocks:
        if block_id(b["state"]) != BLOCK_OBSERVER:
            continue
        facing = b["state"] >> 8 & 0b111
        dx, dy, dz = FACING_OFFSETS[facing]
        sensed = (b["x"] + dx, b["y"] + dy, b["z"] + dz)
        sockets.append({"pos": sensed, "facing": facing, "kind": "in"})

    if not sockets:
        return None
    return Tile(name, [dict(b) for b in crop_blocks], sockets)
