"""The palette - md/ALPHAZERO.md Part 2.

Expected values are written out by hand from block_registry.h and the vanilla meta conventions,
not computed from rlgym.blocks, so an error in the palette builder cannot validate itself.
"""
from __future__ import annotations

import pytest

from rlgym.blocks import (
    BLOCK_AIR,
    BLOCK_PISTON,
    BLOCK_PISTON_HEAD,
    BLOCK_STICKY_PISTON,
    EXTENDED_BIT,
    FACING_OFFSETS,
    LIVE_SLOTS,
    PALETTE,
    PALETTE_SLOTS,
    RESERVED_SLOTS,
    block_id_of,
    make_state,
    meta_of,
)

# Immovable in this simulator: PushReaction::Block in block_registry.cpp (bedrock, portal, end
# portal, anvil, barrier, end gateway) plus obsidian, which is hardcoded in piston.cpp:18 and
# mirrors BlockPistonBase.canPush:384. None of these may ever be placeable - md/ALPHAZERO.md
# excludes them from the palette. Obsidian stays valid for the DEFERRED.md probe, which reads
# the model rather than placing blocks.
IMMOVABLE_IDS = {7, 49, 90, 119, 145, 166, 209}


def test_39_live_entries_in_48_slots():
    assert PALETTE_SLOTS == 48
    assert len(LIVE_SLOTS) == 39
    assert len(RESERVED_SLOTS) == 9
    assert len(PALETTE) == 48


def test_reserved_slots_are_at_the_end():
    """Spare slots must not be interleaved: index IS the action's palette component, so a live
    entry appearing after a reserved one would change meaning if the gap were ever filled."""
    assert RESERVED_SLOTS == tuple(range(39, 48))


def test_no_immovable_block_is_placeable():
    for slot in LIVE_SLOTS:
        assert PALETTE[slot].block_id not in IMMOVABLE_IDS


def test_piston_head_is_never_directly_placeable():
    """A lone head is destroyed as an orphan (simulator.cpp:718). It only ever appears as the
    second cell of an extended-piston action."""
    for slot in LIVE_SLOTS:
        assert PALETTE[slot].block_id != BLOCK_PISTON_HEAD


@pytest.mark.parametrize("base_id", [BLOCK_PISTON, BLOCK_STICKY_PISTON])
@pytest.mark.parametrize("facing", range(6))
def test_retracted_piston_has_extended_bit_clear(base_id, facing):
    entry = next(
        PALETTE[s]
        for s in LIVE_SLOTS
        if PALETTE[s].block_id == base_id
        and PALETTE[s].facing == facing
        and PALETTE[s].head is None
    )
    assert meta_of(entry.state) == facing
    assert not meta_of(entry.state) & EXTENDED_BIT
    assert entry.writes_two_cells is False


@pytest.mark.parametrize("base_id", [BLOCK_PISTON, BLOCK_STICKY_PISTON])
@pytest.mark.parametrize("facing", range(6))
def test_extended_piston_pairs_with_its_head(base_id, facing):
    """base meta = facing | 8; head meta = facing | (8 if sticky else 0).

    Verified against the corpus: (sticky, meta 11) appears 2,597 times and (head, meta 11)
    exactly 2,597 times; (piston, meta 10) 692 times and (head, meta 2) 692 times.
    """
    sticky = base_id == BLOCK_STICKY_PISTON
    entry = next(
        PALETTE[s]
        for s in LIVE_SLOTS
        if PALETTE[s].block_id == base_id
        and PALETTE[s].facing == facing
        and PALETTE[s].head is not None
    )
    assert entry.state == make_state(base_id, facing | EXTENDED_BIT)
    assert entry.head_state == make_state(BLOCK_PISTON_HEAD, facing | (8 if sticky else 0))
    assert entry.writes_two_cells is True


def test_extended_piston_corpus_pairs_exactly():
    """The two pairings actually counted in the fixtures, spelled out as literals."""
    sticky_11 = next(
        PALETTE[s]
        for s in LIVE_SLOTS
        if PALETTE[s].state == make_state(BLOCK_STICKY_PISTON, 11)
    )
    assert sticky_11.head_state == make_state(BLOCK_PISTON_HEAD, 11)

    piston_10 = next(
        PALETTE[s] for s in LIVE_SLOTS if PALETTE[s].state == make_state(BLOCK_PISTON, 10)
    )
    assert piston_10.head_state == make_state(BLOCK_PISTON_HEAD, 2)


def test_air_entry_exists_once_and_is_air():
    airs = [PALETTE[s] for s in LIVE_SLOTS if PALETTE[s].is_air]
    assert len(airs) == 1
    assert airs[0].block_id == BLOCK_AIR
    assert airs[0].state == 0


def test_state_packing_is_id_or_meta_shifted_eight():
    """The convention the fixture JSON uses: state 2589 is sticky piston (29) with meta 10."""
    assert make_state(29, 10) == 2589
    assert block_id_of(2589) == 29
    assert meta_of(2589) == 10
    for slot in LIVE_SLOTS:
        entry = PALETTE[slot]
        assert block_id_of(entry.state) == entry.block_id
        assert meta_of(entry.state) == entry.meta


def test_facing_offsets_match_vanilla_ordinals():
    """down, up, north, south, west, east - the order encode.py's FACING_OFFSET uses."""
    assert FACING_OFFSETS == (
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (-1, 0, 0),
        (1, 0, 0),
    )


def test_every_live_entry_has_a_unique_name_and_state():
    names = [PALETTE[s].name for s in LIVE_SLOTS]
    states = [(PALETTE[s].state, PALETTE[s].head_state) for s in LIVE_SLOTS]
    assert len(set(names)) == len(names)
    assert len(set(states)) == len(states)


def test_directional_entries_are_exactly_thirty():
    """5 kinds x 6 facings: piston and sticky piston in both retracted and extended, plus the
    observer. This is the 5x6 the palette is built around."""
    directional = [s for s in LIVE_SLOTS if PALETTE[s].facing is not None]
    assert len(directional) == 30
