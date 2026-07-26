"""Generator 6 (user's jigsaw/socket idea): assemble preset/extracted Tiles (tiles.py) by
snapping an unused `in` socket of a new tile onto an unused `out` socket of the growing assembly
(geometry.snap_tile handles the rotate+translate+collision-check). Builds causal depth by
construction like gen_forward.py, but without a simulator call per link - one simulate() at the
end covers however many tiles got attached.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

import generator  # noqa: F401
from geometry import block_pos, occupancy, snap_tile
from tiles import PRESET_TILES, Tile

Candidate = dict[str, Any]

MAX_PIECES = 6
SNAP_RETRIES_PER_TILE = 6


def _root(library: list[Tile]) -> tuple[list[dict], list[dict], tuple[int, int, int]]:
    """The one tile type with no `in` socket (a statically self-powered root) starts the
    assembly; its trigger is its own piston's position (the tile's first block)."""
    tile = next(t for t in library if not any(s["kind"] == "in" for s in t.sockets))
    blocks = [dict(b) for b in tile.blocks]
    sockets = [dict(s) for s in tile.sockets]
    trigger_pos = block_pos(tile.blocks[0])
    return blocks, sockets, trigger_pos


def _assemble(library: list[Tile], rng: random.Random) -> Candidate | None:
    blocks, open_sockets, trigger_pos = _root(library)
    piece_count = rng.randint(1, MAX_PIECES)
    drivable = [t for t in library if any(s["kind"] == "in" for s in t.sockets)]
    if not drivable:
        return None

    for _ in range(piece_count):
        out_sockets = [s for s in open_sockets if s["kind"] == "out"]
        if not out_sockets:
            break
        target = rng.choice(out_sockets)

        tile = rng.choice(drivable)
        in_indices = [i for i, s in enumerate(tile.sockets) if s["kind"] == "in"]
        in_index = rng.choice(in_indices)

        occ = occupancy(blocks)
        placed = None
        for _ in range(SNAP_RETRIES_PER_TILE):
            turns = rng.choice(tile.allowed_orientations)
            placed = snap_tile(tile.blocks, tile.sockets, in_index, target["pos"], turns, occ)
            if placed is not None:
                break
        if placed is None:
            continue

        new_blocks, new_sockets = placed
        blocks = blocks + new_blocks
        open_sockets = [s for s in open_sockets if s is not target]
        open_sockets += [s for i, s in enumerate(new_sockets) if i != in_index]

    return {
        "id": 0,
        "trigger": {"x": trigger_pos[0], "y": trigger_pos[1], "z": trigger_pos[2]},
        "blocks": blocks,
    }


def generate(rng: random.Random, extra_tiles: list[Tile] | None = None) -> Iterator[Candidate]:
    library = list(PRESET_TILES) + list(extra_tiles or ())
    while True:
        result = _assemble(library, rng)
        if result is not None:
            yield result
