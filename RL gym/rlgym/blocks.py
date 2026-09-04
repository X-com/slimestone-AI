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
