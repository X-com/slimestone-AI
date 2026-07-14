from __future__ import annotations

from dataclasses import dataclass


# state = blockId | (meta << 8), mirrors block_registry.cpp in the C++ simulator.
# Facing meta is the low 3 bits (0-5): down, up, north, south, west, east.
FACING_DOWN, FACING_UP, FACING_NORTH, FACING_SOUTH, FACING_WEST, FACING_EAST = range(6)

BLOCK_AIR = 0
BLOCK_STONE = 1
BLOCK_GLASS = 20
BLOCK_DETECTOR_RAIL = 28
BLOCK_STICKY_PISTON = 29
BLOCK_PISTON = 33
BLOCK_PISTON_HEAD = 34
BLOCK_OBSIDIAN = 49
BLOCK_FENCE_GATE = 107
BLOCK_REDSTONE_LAMP = 123
BLOCK_REDSTONE_BLOCK = 152
BLOCK_SLIME = 165
BLOCK_OBSERVER = 218

PISTON_EXTENDED_BIT = 3
OBSERVER_POWERED_BIT = 3


def make_state(block_id: int, meta: int = 0) -> int:
    return block_id | (meta << 8)


def block_id(state: int) -> int:
    return state & 0xFF


def block_meta(state: int) -> int:
    return state >> 8


@dataclass(frozen=True)
class BlockKind:
    """One block type the simulator gives special physics to, and the meta values
    that are legal to place it with. Mutation only ever emits states built from this
    table, so a mutated candidate can never contain a block type the simulator would
    silently treat as an inert solid."""

    name: str
    block_id: int
    facings: tuple[int, ...] = ()  # empty means "no facing, meta is always 0"

    def random_state(self, rng) -> int:
        meta = rng.choice(self.facings) if self.facings else 0
        return make_state(self.block_id, meta)


# The full mutation palette. Deliberately excludes piston head/extension (36, 34) -
# those are simulator-managed piston-arm blocks produced at runtime, not something a
# candidate should be hand-placed with. Excludes lit redstone lamp (124) - that state
# is simulator-derived from block 123 + power, not a placeable starting state.
MUTATION_PALETTE: tuple[BlockKind, ...] = (
    BlockKind("air", BLOCK_AIR),
    BlockKind("stone", BLOCK_STONE),
    BlockKind("glass", BLOCK_GLASS),
    #BlockKind("obsidian", BLOCK_OBSIDIAN),
    BlockKind("slime", BLOCK_SLIME),
    BlockKind("redstone_block", BLOCK_REDSTONE_BLOCK),
    #BlockKind("redstone_lamp", BLOCK_REDSTONE_LAMP),
    #BlockKind("detector_rail", BLOCK_DETECTOR_RAIL),
    BlockKind("piston", BLOCK_PISTON, facings=(0, 1, 2, 3, 4, 5)),
    BlockKind("sticky_piston", BLOCK_STICKY_PISTON, facings=(0, 1, 2, 3, 4, 5)),
    BlockKind("observer", BLOCK_OBSERVER, facings=(0, 1, 2, 3, 4, 5)),
    #BlockKind("fence_gate", BLOCK_FENCE_GATE, facings=(0, 1, 2, 3)),
)

# Kinds worth inserting when growing a machine (air is handled separately by the
# "remove block" operator, not by insertion).
INSERTABLE_KINDS: tuple[BlockKind, ...] = tuple(
    kind for kind in MUTATION_PALETTE if kind.name != "air"
)

# Kinds with a facing that mutation is allowed to re-roll.
FACING_KINDS: tuple[BlockKind, ...] = tuple(kind for kind in MUTATION_PALETTE if kind.facings)

_BY_BLOCK_ID: dict[int, BlockKind] = {kind.block_id: kind for kind in MUTATION_PALETTE}


def kind_for_state(state: int) -> BlockKind | None:
    return _BY_BLOCK_ID.get(block_id(state))


def is_palette_state(state: int) -> bool:
    return block_id(state) in _BY_BLOCK_ID
