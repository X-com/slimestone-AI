"""Position/rotation helpers shared by every gen_*.py: cropping a window out of a large machine,
rebasing to local coords, picking a trigger, checking occupancy, and rotating blocks/facings for
puzzle-tile assembly (gen_puzzle.py).

Facing offsets match facing.h / blocks.py's FACING_DOWN..FACING_EAST (0-5) order: down, up,
north, south, west, east. Duplicated here as a small tuple (same convention transformer_gym/
encode.py already uses) rather than importing mutation.py's underscore-prefixed private copy.
"""
from __future__ import annotations

import random
from typing import Any

import generator  # noqa: F401  (sys.path shim)
from genetic_ml.blocks import BLOCK_OBSERVER, BLOCK_PISTON, BLOCK_STICKY_PISTON, block_id, block_meta

Block = dict[str, Any]
Pos = tuple[int, int, int]

FACING_OFFSETS: tuple[Pos, ...] = (
    (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0),
)
FACING_OPPOSITE = (1, 0, 3, 2, 5, 4)
_PISTON_IDS = {BLOCK_PISTON, BLOCK_STICKY_PISTON}


def block_pos(b: Block) -> Pos:
    return (b["x"], b["y"], b["z"])


def bbox(blocks: list[Block]) -> tuple[Pos, Pos]:
    xs, ys, zs = (b["x"] for b in blocks), (b["y"] for b in blocks), (b["z"] for b in blocks)
    xs, ys, zs = list(xs), list(ys), list(zs)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def translate(blocks: list[Block], offset: Pos) -> list[Block]:
    dx, dy, dz = offset
    return [{"x": b["x"] + dx, "y": b["y"] + dy, "z": b["z"] + dz, "state": b["state"]} for b in blocks]


def occupancy(blocks: list[Block]) -> dict[Pos, int]:
    return {block_pos(b): b["state"] for b in blocks}


def crop_window(all_blocks: list[Block], window_min: Pos, size: Pos) -> list[Block]:
    """Blocks whose position falls in [window_min, window_min+size), rebased so window_min -> 0."""
    wx, wy, wz = window_min
    sx, sy, sz = size
    cropped = [
        b for b in all_blocks
        if wx <= b["x"] < wx + sx and wy <= b["y"] < wy + sy and wz <= b["z"] < wz + sz
    ]
    return translate(cropped, (-wx, -wy, -wz))


def pick_trigger(blocks: list[Block], rng: random.Random) -> Pos | None:
    """A trigger must sit on a piston or observer (Simulator::trigger enforces this) - pick one
    at random from what's actually in `blocks`; None if there's nothing triggerable."""
    candidates = [block_pos(b) for b in blocks if block_id(b["state"]) in _PISTON_IDS | {BLOCK_OBSERVER}]
    if not candidates:
        return None
    return rng.choice(candidates)


def rotate_yaw_pos(pos: Pos, turns: int) -> Pos:
    """Rotates a position turns*90 deg around the Y axis (down/up unaffected). turns is taken
    mod 4; each turn maps (dx,dz) -> (-dz,dx), which is exactly the permutation that sends
    facing NORTH->EAST->SOUTH->WEST->NORTH (verified against facing.h's offsets)."""
    x, y, z = pos
    for _ in range(turns % 4):
        x, z = -z, x
    return (x, y, z)


_YAW_CYCLE = {2: 5, 5: 3, 3: 4, 4: 2}  # NORTH->EAST->SOUTH->WEST->NORTH; DOWN/UP fixed


def rotate_facing(facing: int, turns: int) -> int:
    for _ in range(turns % 4):
        facing = _YAW_CYCLE.get(facing, facing)
    return facing


def rotate_state(state: int, turns: int) -> int:
    """Rotates a block's facing meta (low 3 bits) by `turns`, leaving any higher bits (extended,
    sticky, powered, ...) untouched - mirrors blocks.make_state's bit layout."""
    bid = block_id(state)
    meta = block_meta(state)
    facing = meta & 0b111
    other_bits = meta & ~0b111
    return bid | ((rotate_facing(facing, turns) | other_bits) << 8)


def rotate_block(b: Block, turns: int) -> Block:
    x, y, z = rotate_yaw_pos(block_pos(b), turns)
    return {"x": x, "y": y, "z": z, "state": rotate_state(b["state"], turns)}


# A socket is a plain dict {"pos": (x,y,z), "facing": int|None, "kind": "in"|"out"}. `pos` is a
# marker cell in the tile's own local coordinates: for an "in" socket, the cell that must land
# exactly on the driving tile's "out" socket cell (the sense-marker a tile's own observer is
# built to watch, at a fixed offset baked into the tile at authoring time - see tiles.py); for an
# "out" socket, the cell whose change is this tile's effect (a pushed block's final rest cell).
# `facing` is informational only (not used in the snap equation) - a hint about which local
# direction the socket's internal mechanism faces, useful for authoring/debugging.
# Kept as a dict rather than a class so geometry.py (rotation/snap math) never needs to import
# tiles.py's Tile type.
Socket = dict[str, Any]


def snap_tile(
    tile_blocks: list[Block], tile_sockets: list[Socket], in_index: int,
    target_pos: Pos, turns: int, occupied: set[Pos],
) -> tuple[list[Block], list[Socket]] | None:
    """Rotates `tile_blocks`/`tile_sockets` by `turns` and translates them so the tile's
    sockets[in_index].pos lands exactly on `target_pos` (an assembly out-socket's cell). Any
    rotation can satisfy this via translation alone (3 free parameters) - rotation is chosen by
    the caller purely to vary shape / dodge collisions, not because it's physically required.
    Returns None if any rotated+placed block collides with `occupied`."""
    in_socket = tile_sockets[in_index]
    rq = rotate_yaw_pos(in_socket["pos"], turns)
    offset = tuple(target_pos[i] - rq[i] for i in range(3))

    new_blocks = translate([rotate_block(b, turns) for b in tile_blocks], offset)
    if any(block_pos(b) in occupied for b in new_blocks):
        return None

    new_sockets: list[Socket] = []
    for s in tile_sockets:
        rp = rotate_yaw_pos(s["pos"], turns)
        rp = (rp[0] + offset[0], rp[1] + offset[1], rp[2] + offset[2])
        facing = rotate_facing(s["facing"], turns) if s.get("facing") is not None else None
        new_sockets.append({"pos": rp, "facing": facing, "kind": s["kind"]})
    return new_blocks, new_sockets
