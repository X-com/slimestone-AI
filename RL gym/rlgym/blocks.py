"""Block ids, meta conventions and the placement palette.

Every constant here mirrors cpp simulator/src/block_registry.h. The packing convention
state = block_id | (meta << 8) is the one used by the fixture JSON and by the compact wire
format both sides of the stdin protocol agree on.

The palette is md/ALPHAZERO.md Part 2: 39 live entries in 48 slots. The 9 spare slots are
masked and exist so a new block type can be added later without changing the policy head's
shape - see the reserved-but-frozen table in that document.
"""
from __future__ import annotations

from dataclasses import dataclass

BLOCK_AIR = 0
BLOCK_STONE = 1
BLOCK_GLASS = 20
BLOCK_STICKY_PISTON = 29
BLOCK_PISTON = 33
BLOCK_PISTON_HEAD = 34
BLOCK_TRAPDOOR = 96
BLOCK_FENCE_GATE = 107
BLOCK_REDSTONE_LAMP = 123
BLOCK_REDSTONE_BLOCK = 152
BLOCK_SLIME = 165
BLOCK_OBSERVER = 218
BLOCK_GLAZED_TERRACOTTA = 235

# Meta bit 3 on a piston/sticky piston means "extended" (simulator.cpp uses metaBit(state, 3)
# everywhere for this). An extended piston is immovable and owns a head in the cell it faces.
EXTENDED_BIT = 8
# A piston head's meta is its facing, plus bit 3 when the base is sticky - see
# simulator.cpp:1653 setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), facing).
STICKY_HEAD_BIT = 8

# Facing index -> offset, matching encode.py's FACING_OFFSET and vanilla EnumFacing ordinals.
FACING_OFFSETS = ((0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
FACING_NAMES = ("down", "up", "north", "south", "west", "east")
N_FACINGS = 6

PALETTE_SLOTS = 48


def make_state(block_id: int, meta: int = 0) -> int:
    return block_id | (meta << 8)


def block_id_of(state: int) -> int:
    return state & 0xFF


def meta_of(state: int) -> int:
    return state >> 8


@dataclass(frozen=True)
class PaletteEntry:
    """One placeable thing. Most write a single cell; extended pistons write two.

    head is None for everything except extended pistons, where it is the (block_id, meta) that
    must land in the cell at base + FACING_OFFSETS[facing]. That pairing is not optional: a lone
    head is destroyed as an orphan (simulator.cpp:718) and replacing an extended piston's head
    kills the base (simulator.cpp:2197), so the two cells are written atomically as one action.
    """

    name: str
    block_id: int
    meta: int
    facing: int | None = None
    head: tuple[int, int] | None = None

    @property
    def state(self) -> int:
        return make_state(self.block_id, self.meta)

    @property
    def head_state(self) -> int | None:
        if self.head is None:
            return None
        return make_state(self.head[0], self.head[1])

    @property
    def is_air(self) -> bool:
        return self.block_id == BLOCK_AIR

    @property
    def writes_two_cells(self) -> bool:
        return self.head is not None


def _build_palette() -> list[PaletteEntry | None]:
    entries: list[PaletteEntry | None] = []

    for base_id, label in ((BLOCK_PISTON, "piston"), (BLOCK_STICKY_PISTON, "sticky_piston")):
        sticky = base_id == BLOCK_STICKY_PISTON
        for extended in (False, True):
            for facing in range(N_FACINGS):
                meta = facing | (EXTENDED_BIT if extended else 0)
                head = None
                if extended:
                    head = (BLOCK_PISTON_HEAD, facing | (STICKY_HEAD_BIT if sticky else 0))
                suffix = "extended" if extended else "retracted"
                entries.append(
                    PaletteEntry(
                        name=f"{label}_{suffix}_{FACING_NAMES[facing]}",
                        block_id=base_id,
                        meta=meta,
                        facing=facing,
                        head=head,
                    )
                )

    for facing in range(N_FACINGS):
        entries.append(
            PaletteEntry(
                name=f"observer_{FACING_NAMES[facing]}",
                block_id=BLOCK_OBSERVER,
                meta=facing,
                facing=facing,
            )
        )

    # Non-directional. Fence gate and trapdoor are pinned to meta 0 - their facings are not
    # modelled (md/ALPHAZERO.md Part 2). Stone stands in for every plain solid cube in the
    # corpus: sandstone, gold, quartz, prismarine, diamond, coal and stained clay are all
    # functionally identical to it, ~1,100 uses across a dozen ids.
    for block_id, name in (
        (BLOCK_SLIME, "slime"),
        (BLOCK_STONE, "stone"),
        (BLOCK_GLASS, "glass"),
        (BLOCK_REDSTONE_BLOCK, "redstone_block"),
        (BLOCK_GLAZED_TERRACOTTA, "glazed_terracotta"),
        (BLOCK_REDSTONE_LAMP, "redstone_lamp"),
        (BLOCK_FENCE_GATE, "fence_gate"),
        (BLOCK_TRAPDOOR, "trapdoor"),
    ):
        entries.append(PaletteEntry(name=name, block_id=block_id, meta=0))

    entries.append(PaletteEntry(name="air", block_id=BLOCK_AIR, meta=0))

    if len(entries) > PALETTE_SLOTS:
        raise AssertionError(f"palette overflows its {PALETTE_SLOTS} slots: {len(entries)}")
    entries.extend([None] * (PALETTE_SLOTS - len(entries)))
    return entries


# Index -> entry, or None for a reserved slot. Index IS the action's palette component, so this
# ordering is part of the saved model's meaning: appending is safe, reordering is not.
PALETTE: tuple[PaletteEntry | None, ...] = tuple(_build_palette())
LIVE_SLOTS = tuple(i for i, entry in enumerate(PALETTE) if entry is not None)
RESERVED_SLOTS = tuple(i for i, entry in enumerate(PALETTE) if entry is None)


# --- BlockData lookups, mirroring cpp simulator/src/block_registry.cpp ---------------------
#
# DECISIONS.md point 1 calls these the "worked out from block type by table lookup" features:
# supplied from a preset table rather than learned, because the simulator already knows them and
# making the model rediscover them from examples wastes data it does not have.

PUSH_NORMAL = 0
PUSH_BLOCK = 1
PUSH_DESTROY = 2
PUSH_ONLY = 3

_NORMAL_CUBES = frozenset({
    1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 19, 21, 22, 24, 35, 41, 42, 43, 45, 47, 48, 56,
    57, 58, 61, 62, 73, 74, 80, 82, 87, 88, 97, 98, 99, 100, 110, 112, 121, 123, 124, 125, 129,
    133, 137, 159, 162, 165, 166, 168, 170, 172, 173, 174, 179, 181, 201, 202, 204, 206, 210,
    211, 213, 214, 215, 216, 251, 252, 255,
})
_DESTROY = frozenset({
    6, 8, 9, 10, 11, 18, 26, 30, 31, 32, 37, 38, 39, 40, 50, 51, 55, 64, 65, 70, 71, 72, 75, 76,
    78, 81, 83, 86, 91, 92, 93, 94, 103, 104, 105, 106, 115, 122, 127, 131, 132, 140, 143, 147,
    148, 175, 193, 194, 195, 196, 197, 199, 200, 207, 217, 219, 220, 221, 222, 223, 224, 225,
    226, 227, 228, 229, 230, 231, 232, 233, 234,
})
_PUSH_BLOCKED = frozenset({7, 90, 119, 145, 166, 209})
_GLAZED = frozenset(range(235, 251))
# hardness -1 in the registry, plus obsidian, which piston.cpp:18 hardcodes exactly as vanilla
# BlockPistonBase.canPush:384 does. Extended pistons are immovable too, but that is a state
# property rather than a block-id one and is handled from the meta bit.
_IMMOVABLE = frozenset({7, 49, 90, 119, 120, 137, 166, 209, 210, 211, 255})
_PROVIDES_POWER = frozenset({BLOCK_REDSTONE_BLOCK, BLOCK_OBSERVER})
_PISTON_FAMILY = frozenset({BLOCK_PISTON, BLOCK_STICKY_PISTON, BLOCK_PISTON_HEAD, 36})


def push_reaction(block: int) -> int:
    if block in _GLAZED:
        return PUSH_ONLY
    if block in _PISTON_FAMILY or block in _PUSH_BLOCKED:
        return PUSH_BLOCK
    if block in _DESTROY:
        return PUSH_DESTROY
    return PUSH_NORMAL


def is_normal_cube(block: int) -> bool:
    """A full opaque cube that does NOT itself provide power - the registry excludes power
    sources here deliberately, so their own weak output is not shadowed."""
    return (block in _NORMAL_CUBES or block in _GLAZED) and block not in _PROVIDES_POWER


def is_full_block(block: int) -> bool:
    """Physical support: can a rail sit on this. Unlike is_normal_cube this includes the
    observer, which things do sit on."""
    return block in _NORMAL_CUBES or block in _GLAZED or block == BLOCK_OBSERVER


def can_provide_power(block: int) -> bool:
    return block in _PROVIDES_POWER


def is_immovable(block: int, meta: int = 0) -> bool:
    """An extended piston is immovable while extended; every other case is by block id."""
    if block in _PISTON_FAMILY and (meta & EXTENDED_BIT):
        return True
    return block in _IMMOVABLE or block in _PUSH_BLOCKED


def is_sticky(block: int) -> bool:
    return block == BLOCK_SLIME
