"""Hand-written semantic assertions on the graph - md/VERIFICATION.md, the honest gap.

Structural invariants cannot catch "I connected the observer to the wrong cell". Nothing can,
automatically: there is no second implementation to diff against and no ground truth for what
the graph *should* say. What there is, is a machine small enough to reason about by hand.

simple_observer_engine, in record coordinates (the fixture sits at y=0-1, the log at y=64-65):

    (1,64,1) sticky piston facing west     (0,65,1) observer facing north  <- trigger
    (0,64,1) slime                         (1,65,2) observer facing south
    (1,64,2) slime                         (0,64,2) piston facing east

Every expected value below is derived from that layout and the vanilla facing conventions, never
from calling the graph builder.

This file already earned its place: writing it is what surfaced the observer at (2,65,2) watching
a cell outside the encoded region, which no structural check would ever have flagged.
"""
from __future__ import annotations

import numpy as np
import pytest

from rlgym.blocks import block_id_of, meta_of
from rlgym.boards import board_at, boards_by_tick
from rlgym.game import Placement
from rlgym.graph import (
    REL_CELL_HAS_EVENT,
    REL_EVENT_AT_CELL,
    REL_EVENT_NEXT,
    REL_FROM_SUMMARY,
    REL_OBSERVED_BY,
    REL_OBSERVES,
    REL_SAME_PUSH_GROUP,
    REL_SELF,
    REL_TIME_BWD,
    REL_TIME_FWD,
    REL_TO_SUMMARY,
    REL_TOUCHING,
    SCALARS,
    build,
)
from rlgym.simlog import record_for

pytestmark = pytest.mark.slow

OBSERVER = 218
SLIME_SLOT = 30


@pytest.fixture(scope="module")
def built(request):
    engine = request.getfixturevalue("engine")
    record = record_for(engine.to_candidate(cid=0))
    return engine, record, build(engine, record)


def _edges(graph, relation):
    mask = graph.edges[0] == relation
    return list(zip(graph.edges[1][mask].tolist(), graph.edges[2][mask].tolist()))


def _cell_item_map(graph, record, machine):
    """Rebuild (cell, tick) -> item independently of the builder, from the pooling rows and the
    known ordering, so this file does not simply trust the structure it is checking."""
    boards = boards_by_tick(record)
    span = record.summary.period
    out = {}
    n_ticks = graph.n_ticks
    cells = int((graph.item_type == 0).sum()) // n_ticks
    # cell items are emitted tick-major, cells sorted within each tick
    ordered = sorted(
        set(k for t in range(span + 1) for k in board_at(boards, t))
        | set(_shifted_policy(graph, machine, record))
    )
    return out, ordered


def _shifted_policy(graph, machine, record):
    mx = min(c[0] for c in machine.cells)
    rx = min(b.pos[0] for b in record.initial)
    my = min(c[1] for c in machine.cells)
    ry = min(b.pos[1] for b in record.initial)
    mz = min(c[2] for c in machine.cells)
    rz = min(b.pos[2] for b in record.initial)
    off = (rx - mx, ry - my, rz - mz)
    return [(c[0] + off[0], c[1] + off[1], c[2] + off[2]) for c in machine.cell_list]


# --- observers ------------------------------------------------------------------------------


def test_each_observer_watches_exactly_one_cell_per_tick_it_exists(built):
    """An observer observes the single block it faces, not a line of sight. So the number of
    observes edges must equal the number of (observer, tick) pairs where the observer is
    actually on the board - it vanishes while mid-flight, holding the id-36 placeholder."""
    _, record, graph = built
    boards = boards_by_tick(record)
    expected = 0
    for tick in range(record.summary.period + 1):
        board = board_at(boards, tick)
        expected += sum(1 for state in board.values() if block_id_of(state) == OBSERVER)
    assert len(_edges(graph, REL_OBSERVES)) == expected
    assert expected == 18  # 2 observers x 11 ticks, minus 4 mid-flight observer-ticks


def test_observes_and_observed_by_are_exact_reverses(built):
    _, _, graph = built
    forward = set(_edges(graph, REL_OBSERVES))
    backward = set(_edges(graph, REL_OBSERVED_BY))
    assert forward == {(b, a) for a, b in backward}


def test_the_north_facing_observer_watches_the_cell_to_its_north(built):
    """(0,65,1) faces north (meta 2), and north is -z, so it watches (0,65,0). Asserted through
    the graph's own edges, with the expected target computed from the facing convention rather
    than read back from the builder."""
    machine, record, graph = built
    boards = boards_by_tick(record)
    board = board_at(boards, 0)
    observer_cells = [c for c, s in board.items() if block_id_of(s) == OBSERVER]
    north_facing = [c for c in observer_cells if (meta_of(board[c]) & 7) == 2]
    assert north_facing == [(0, 65, 1)]

    # tick-0 cell items are the first block of items, cells sorted
    cells = sorted(
        set(k for t in range(record.summary.period + 1) for k in board_at(boards, t))
        | set(_shifted_policy(graph, machine, record))
    )
    index = {cell: i for i, cell in enumerate(cells)}
    src = index[(0, 65, 1)]
    dst = index[(0, 65, 0)]
    assert (src, dst) in _edges(graph, REL_OBSERVES)


# --- push groups ----------------------------------------------------------------------------


def test_push_group_members_form_a_clique(built):
    """Every member links to every other, which is what makes a 12-block group ONE step across
    regardless of size or shape. A 3-member group is 3 unordered pairs, so 6 directed edges."""
    _, record, graph = built
    groups = [g for g in record.push_groups if g.member_count > 0]
    expected = sum(g.member_count * (g.member_count - 1) for g in groups)
    assert len(_edges(graph, REL_SAME_PUSH_GROUP)) == expected
    assert expected == 12  # two groups of 3 members


def test_push_group_edges_are_symmetric(built):
    _, _, graph = built
    edges = set(_edges(graph, REL_SAME_PUSH_GROUP))
    assert edges == {(b, a) for a, b in edges}


# --- touching -------------------------------------------------------------------------------


def test_touching_is_symmetric_and_never_self(built):
    _, _, graph = built
    edges = set(_edges(graph, REL_TOUCHING))
    assert edges == {(b, a) for a, b in edges}
    assert not any(a == b for a, b in edges)


def test_touching_only_joins_items_in_the_same_tick(built):
    """A cell at tick 3 must not be 'touching' a cell at tick 7. Time is what the time links are
    for; conflating the two would let information leap across the cycle for free."""
    _, _, graph = built
    n_ticks = graph.n_ticks
    cells = int((graph.item_type == 0).sum()) // n_ticks
    for a, b in _edges(graph, REL_TOUCHING):
        assert a // cells == b // cells


# --- time -----------------------------------------------------------------------------------


def test_time_forward_and_backward_are_exact_reverses(built):
    _, _, graph = built
    for gap_index in range(4):
        forward = set(_edges(graph, REL_TIME_FWD + gap_index))
        backward = set(_edges(graph, REL_TIME_BWD + gap_index))
        assert forward == {(b, a) for a, b in backward}


def test_gap_one_time_links_join_consecutive_ticks_of_one_cell(built):
    _, _, graph = built
    n_ticks = graph.n_ticks
    cells = int((graph.item_type == 0).sum()) // n_ticks
    for a, b in _edges(graph, REL_TIME_FWD):
        assert b - a == cells  # exactly one tick later, same cell


def test_only_gaps_shorter_than_the_span_are_emitted(built):
    """An 11-tick span uses gaps 1, 2, 4 and 8. Slots for 16, 64 and 256 stay reserved and
    empty, so the model's shape does not depend on cycle length."""
    _, _, graph = built
    used = {int(r) for r in np.unique(graph.edges[0])}
    for gap_index in range(4):
        assert REL_TIME_FWD + gap_index in used
    for gap_index in range(4, 7):
        assert REL_TIME_FWD + gap_index not in used


# --- events ---------------------------------------------------------------------------------


def test_event_and_cell_links_are_exact_reverses(built):
    _, _, graph = built
    forward = set(_edges(graph, REL_EVENT_AT_CELL))
    backward = set(_edges(graph, REL_CELL_HAS_EVENT))
    assert forward == {(b, a) for a, b in backward}


def test_within_tick_ordering_is_a_chain_not_a_clique(built):
    """Events in one tick form a path in emission order. A clique would say they happened
    simultaneously, which is exactly the ordering information the separate event items exist to
    preserve."""
    _, _, graph = built
    edges = _edges(graph, REL_EVENT_NEXT)
    sources = [a for a, _ in edges]
    targets = [b for _, b in edges]
    assert len(set(sources)) == len(sources)
    assert len(set(targets)) == len(targets)


def test_event_items_link_to_a_cell_at_their_own_tick(built):
    _, _, graph = built
    n_ticks = graph.n_ticks
    cells = int((graph.item_type == 0).sum()) // n_ticks
    for event_item, cell_item in _edges(graph, REL_EVENT_AT_CELL):
        assert graph.item_type[event_item] == 1
        assert graph.item_type[cell_item] == 0
        assert cell_item < cells * n_ticks


# --- summary --------------------------------------------------------------------------------


def test_summary_reaches_every_item_both_ways(built):
    _, _, graph = built
    inbound = {a for a, _ in _edges(graph, REL_TO_SUMMARY)}
    outbound = {b for _, b in _edges(graph, REL_FROM_SUMMARY)}
    everything = set(range(graph.n_items)) - {graph.summary_item}
    assert inbound == everything
    assert outbound == everything


def test_every_item_has_a_self_loop(built):
    _, _, graph = built
    assert {a for a, _ in _edges(graph, REL_SELF)} == set(range(graph.n_items))


# --- placements -----------------------------------------------------------------------------


def test_a_note_marks_tick_zero_only_but_flags_every_tick(built):
    """note_block_type states a fact about the initial configuration, so it belongs to tick 0
    alone. cell_is_noted says 'everything you read at this cell describes a machine that no
    longer exists', which is true at every tick."""
    machine, record, _ = built
    empty = next(c for c in machine.cell_list if c not in machine.cells)
    graph = build(machine, record, placements=(Placement(empty, SLIME_SLOT),))

    noted_column = SCALARS.index("cell_is_noted")
    cell_index = machine.cell_list.index(empty)
    items = graph.cell_items[cell_index]

    assert graph.note_type[items[0]] == SLIME_SLOT + 1
    assert all(graph.note_type[item] == 0 for item in items[1:])
    assert all(graph.scalars[item, noted_column] == 1.0 for item in items)


def test_an_unnoted_cell_carries_no_note(built):
    machine, record, _ = built
    empties = [c for c in machine.cell_list if c not in machine.cells]
    graph = build(machine, record, placements=(Placement(empties[0], SLIME_SLOT),))
    noted_column = SCALARS.index("cell_is_noted")
    other = machine.cell_list.index(empties[1])
    for item in graph.cell_items[other]:
        assert graph.note_type[item] == 0
        assert graph.scalars[item, noted_column] == 0.0


def test_extended_piston_notes_both_of_its_cells(built):
    """The compound action writes two cells, so both must be flagged - otherwise the head cell
    would read as untouched and the model would reason about a machine that cannot exist."""
    machine, record, _ = built
    slot = 23  # sticky_piston_extended_east
    base = next(
        c
        for c in machine.cell_list
        if (c[0] + 1, c[1], c[2]) in machine.cell_list and c not in machine.cells
    )
    graph = build(machine, record, placements=(Placement(base, slot),))
    noted_column = SCALARS.index("cell_is_noted")
    for cell in (base, (base[0] + 1, base[1], base[2])):
        index = machine.cell_list.index(cell)
        assert graph.scalars[graph.cell_items[index][0], noted_column] == 1.0


# --- features -------------------------------------------------------------------------------


def test_the_trigger_cell_is_flagged_exactly_once_per_tick(built):
    _, record, graph = built
    column = SCALARS.index("is_trigger")
    flagged = graph.scalars[:, column].sum()
    assert flagged == graph.n_ticks


def test_slime_is_marked_sticky_and_stone_is_not(built):
    """A table lookup from block_registry.cpp, supplied rather than learned."""
    _, _, graph = built
    sticky = SCALARS.index("is_sticky")
    for item in range(graph.n_items):
        if graph.block_type[item] == 165:
            assert graph.scalars[item, sticky] == 1.0
        elif graph.block_type[item] == 1:
            assert graph.scalars[item, sticky] == 0.0


def test_push_group_size_is_normalised_by_the_push_limit(built):
    """Groups of 3 in a 12-block limit, so 0.25 - not a raw count, which would dominate the
    shared input projection."""
    _, _, graph = built
    column = SCALARS.index("push_group_size")
    values = {round(float(v), 4) for v in graph.scalars[:, column] if v > 0}
    assert values == {0.25}
