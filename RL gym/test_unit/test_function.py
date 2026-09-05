"""Redundant-block detection - md/VERIFICATION.md, for what replaced the cargo metric.

The definition being tested:

> A block is functional if it is part of a group of blocks that leads to a piston pushing other
> blocks, **and** removing it disturbs the operations of the pistons. Disturbed means a piston
> fires on a different tick, or changes order relative to other pistons firing in the same tick.

Plus the half that measurement forced into it: **the machine must still work.** Those are not the
same condition, and the fixture proves it - see `test_signature_alone_is_not_enough`.

Signature tests build their records by hand rather than simulating, so a failure points at the
comparison and cannot be a symptom of the decoder.
"""
from __future__ import annotations

import dataclasses

import pytest

from rlgym.blocks import PALETTE
from rlgym.function import (
    added_cells,
    block_cells,
    changed_push_groups,
    piston_signature,
    trim,
    unchanged,
)
from rlgym.game import Placement, apply_placement
from rlgym.record import PISTON_MOVE_EXECUTED, SEF_EXTEND, SEF_SUCCESS

SLIME = 30
GLASS = 32
EXTENDED_PISTON = 6


# --- the signature, on hand-built records ---------------------------------------------


def _event(tick: int, subtick: int, pos=(0, 0, 0), kind: int = PISTON_MOVE_EXECUTED):
    from rlgym.record import Event

    return Event(
        block_key=0, actor_key=0, target_key=0, global_seq=subtick,
        activation_tick=tick, scheduled_tick=tick, executed_tick=tick,
        activation_subtick=subtick, scheduled_subtick=subtick, executed_subtick=subtick,
        push_group_id=0, from_pos=pos, to_pos=pos, kind=kind, direction=5,
        flags=SEF_EXTEND | SEF_SUCCESS, attempted_amount=0, actual_amount=0,
        failure_reason=0, reserved0=0, reserved1=0, reserved2=0,
    )


def _record_with(events):
    class Fake:
        def events_in_order(self):
            return sorted(events, key=lambda e: e.activation_subtick)

    return Fake()


def test_the_signature_is_sensitive_to_the_tick_a_piston_fires_on():
    """The first half of the definition, stated as literally as it can be."""
    early = piston_signature(_record_with([_event(tick=4, subtick=10)]))
    late = piston_signature(_record_with([_event(tick=5, subtick=10)]))
    assert early != late
    assert early[0][0] == 4 and late[0][0] == 5


def test_the_signature_is_sensitive_to_order_within_one_tick():
    """The second half, and the one a set comparison would silently pass.

    Two pistons firing on the SAME tick, with their subticks swapped. Every tick is identical,
    every position is identical, only the order differs - and that is the definition of changed
    behaviour, so the signatures must differ.
    """
    a, b = (0, 0, 0), (5, 0, 0)
    forwards = piston_signature(
        _record_with([_event(3, 10, a), _event(3, 11, b)])
    )
    backwards = piston_signature(
        _record_with([_event(3, 10, b), _event(3, 11, a)])
    )
    assert forwards != backwards
    assert [op[1] for op in forwards] == [0, 1], "rank within tick is not being assigned"
    assert forwards[0][2] == a and backwards[0][2] == b


def test_non_piston_events_are_ignored():
    """The signature is about pistons. A block-state write on a different tick must not enter."""
    from rlgym.record import BLOCK_STATE_CHANGED

    only = piston_signature(_record_with([_event(2, 5)]))
    noisy = piston_signature(
        _record_with([_event(2, 5), _event(9, 99, kind=BLOCK_STATE_CHANGED)])
    )
    assert only == noisy


# --- block identity -------------------------------------------------------------------


def test_an_extended_piston_is_one_block_of_two_cells():
    """A lone head is destroyed as an orphan, so body and head are removed together or not at
    all. Getting this wrong would test removals that are not legal machine edits."""
    entry = PALETTE[EXTENDED_PISTON]
    assert entry is not None and entry.head is not None
    body = (0, 0, 0)
    placement = Placement(body, EXTENDED_PISTON)
    cells = apply_placement({}, placement)
    unit = block_cells(cells, body)
    assert unit == frozenset(placement.written_cells())
    assert len(unit) == 2
    # ...and asking about the head must return the same pair, not the head alone.
    head = placement.written_cells()[1]
    assert block_cells(cells, head) == unit


def test_an_ordinary_block_is_one_cell():
    cells = apply_placement({}, Placement((1, 2, 3), SLIME))
    assert block_cells(cells, (1, 2, 3)) == frozenset({(1, 2, 3)})


def test_an_empty_cell_has_no_block():
    assert block_cells({}, (0, 0, 0)) == frozenset()


# --- against the simulator --------------------------------------------------------------


@pytest.fixture(scope="module")
def engine_base(request):
    from rlgym.simlog import record_for

    machine = request.getfixturevalue("engine")
    return machine, record_for(machine.to_candidate(cid=0))


@pytest.mark.slow
def test_signature_alone_is_not_enough(engine_base):
    """The measurement that changed the definition.

    Removing (0,0,1) from `simple_observer_engine` leaves the piston signature **byte-identical**
    while `validCycle` flips True -> False and the period collapses from 10 to 2. A redundancy
    test on pistons alone would delete a load-bearing block and admit a broken machine.
    """
    from rlgym.simlog import record_for

    machine, base = engine_base
    signature = piston_signature(base)
    without = {c: s for c, s in machine.cells.items() if c != (0, 0, 1)}
    record = record_for(machine.to_candidate(without, cid=0))

    assert piston_signature(record) == signature, "the premise of this test has changed"
    assert not record.summary.valid_cycle
    assert not unchanged(record, signature), "validCycle is not being required"


@pytest.mark.slow
def test_a_refused_machine_is_never_called_redundant():
    """No record means the simulator refused it outright - maximal disturbance, so the block is
    load-bearing. `None` must not fall through as 'unchanged'."""
    assert not unchanged(None, ())


@pytest.mark.slow
def test_a_block_clear_of_the_mechanism_is_stripped(engine_base):
    machine, base = engine_base
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    cells = apply_placement(machine.cells, Placement(cell, GLASS))
    result = trim(machine, base, cells, added_cells(machine.cells, cells))
    assert cell in result.removed
    assert not result.added_is_load_bearing
    assert cell not in result.cells


@pytest.mark.slow
def test_trimming_never_returns_an_unverified_machine(engine_base):
    """Whatever comes back must itself work and keep the pistons on the same schedule - which is
    the property the greedy fallback exists to preserve when a joint removal breaks."""
    from rlgym.simlog import record_for

    machine, base = engine_base
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    cells = apply_placement(machine.cells, Placement(cell, SLIME))
    result = trim(machine, base, cells, added_cells(machine.cells, cells))

    before = piston_signature(record_for(machine.to_candidate(cells, cid=0)))
    after = record_for(machine.to_candidate(result.cells, cid=0))
    assert after.summary.valid_cycle
    assert piston_signature(after) == before


@pytest.mark.slow
def test_an_unmodified_machine_has_no_changed_push_groups(engine_base):
    """The diff must be empty against itself, or every discovery would report every group as
    modified and the candidate set would be meaningless."""
    _, base = engine_base
    assert changed_push_groups(base, base) == []


@pytest.mark.slow
def test_added_cells_finds_the_placement(engine_base):
    machine, _ = engine_base
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    cells = apply_placement(machine.cells, Placement(cell, SLIME))
    assert added_cells(machine.cells, cells) == {cell}


@pytest.mark.slow
def test_removing_a_block_is_not_reported_as_added(engine_base):
    """Air removes a cell rather than storing state 0, so the candidate is a strict subset and
    nothing was added."""
    machine, _ = engine_base
    air = next(i for i, e in enumerate(PALETTE) if e is not None and e.is_air)
    occupied = next(iter(machine.cells))
    cells = apply_placement(machine.cells, Placement(occupied, air))
    assert added_cells(machine.cells, cells) == set()
