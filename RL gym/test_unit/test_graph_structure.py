"""Structural invariants for the graph - md/VERIFICATION.md.

These catch indices out of range, wrong counts and shape drift. They cannot catch a plausible
but wrong edge; test_graph_semantics.py exists for that.
"""
from __future__ import annotations

import numpy as np
import pytest

from rlgym.game import Machine, Placement
from rlgym.graph import (
    FEATURE_VERSION,
    ITEM_CELL,
    ITEM_EVENT,
    ITEM_SUMMARY,
    N_RELATION_SLOTS,
    N_SCALARS,
    SCALARS,
    TIME_GAPS,
    GraphTooLarge,
    build,
)
from rlgym.record import N_KINDS
from rlgym.simlog import record_for

pytestmark = pytest.mark.slow

SLIME_SLOT = 30


@pytest.fixture(scope="module")
def built(request):
    engine = request.getfixturevalue("engine")
    record = record_for(engine.to_candidate(cid=0))
    return engine, record, build(engine, record)


def test_exactly_one_summary_item(built):
    _, _, graph = built
    assert int((graph.item_type == ITEM_SUMMARY).sum()) == 1
    assert graph.item_type[graph.summary_item] == ITEM_SUMMARY


def test_item_counts_are_consistent(built):
    _, record, graph = built
    cells = int((graph.item_type == ITEM_CELL).sum())
    events = int((graph.item_type == ITEM_EVENT).sum())
    assert cells % graph.n_ticks == 0
    assert cells + events + 1 == graph.n_items
    assert events <= record.footer["eventCount"]


def test_every_feature_array_is_the_right_length(built):
    _, _, graph = built
    for array in (
        graph.item_type,
        graph.block_type,
        graph.facing,
        graph.note_type,
        graph.note_facing,
        graph.event_kind,
        graph.failure_reason,
    ):
        assert array.shape == (graph.n_items,)
    assert graph.scalars.shape == (graph.n_items, N_SCALARS)


def test_every_edge_index_is_in_range(built):
    _, _, graph = built
    relations, src, dst = graph.edges
    assert int(relations.min()) >= 0
    assert int(relations.max()) < N_RELATION_SLOTS
    for side in (src, dst):
        assert int(side.min()) >= 0
        assert int(side.max()) < graph.n_items


def test_every_item_has_at_least_one_incoming_edge(built):
    """Edge-restricted attention needs a key set per item. The self-loop guarantees it even for
    an item with no physics relation at all - without it those items would produce NaN."""
    _, _, graph = built
    incoming = np.bincount(graph.edges[2], minlength=graph.n_items)
    assert int(incoming.min()) >= 1


def test_categorical_features_are_within_their_vocabularies(built):
    _, _, graph = built
    assert int(graph.block_type.min()) >= 0 and int(graph.block_type.max()) <= 255
    assert int(graph.facing.min()) >= 0 and int(graph.facing.max()) <= 6
    assert int(graph.event_kind.max()) <= N_KINDS
    assert int(graph.note_type.max()) <= 48


def test_scalars_are_finite_and_bounded(built):
    """Unbounded features would dominate the shared input projection and are almost always a
    normalisation mistake rather than real signal."""
    _, _, graph = built
    assert np.isfinite(graph.scalars).all()
    assert float(np.abs(graph.scalars).max()) <= 10.0


def test_non_cell_items_carry_no_block_features(built):
    _, _, graph = built
    other = graph.item_type != ITEM_CELL
    assert int(graph.block_type[other].max(initial=0)) == 0
    assert (graph.facing[other] == 6).all()


def test_only_event_items_have_an_event_kind(built):
    _, _, graph = built
    not_event = graph.item_type != ITEM_EVENT
    assert (graph.event_kind[not_event] == N_KINDS).all()
    assert (graph.event_kind[graph.item_type == ITEM_EVENT] < N_KINDS).all()


def test_pool_rows_match_the_action_space(built):
    """One pooling row per POLICY cell, in action order - the swept-volume extras are visible to
    the model but are never placement targets."""
    machine, _, graph = built
    assert len(graph.cell_items) == len(machine.cell_list)
    assert graph.policy_cells == machine.cell_list
    for row in graph.cell_items:
        assert len(row) == graph.n_ticks
        assert all(graph.item_type[item] == ITEM_CELL for item in row)


def test_graph_cells_are_a_superset_of_policy_cells(built):
    """The machine flies out of its own tick-0 shell, so the graph must cover more cells than
    the action space does."""
    machine, _, graph = built
    cells = int((graph.item_type == ITEM_CELL).sum()) // graph.n_ticks
    assert cells >= len(machine.cell_list)


def test_feature_version_is_stamped(built):
    """Point 1's safeguard: a saved model records what its columns meant, so a stale checkpoint
    fails loudly instead of training on shifted columns."""
    _, _, graph = built
    assert graph.feature_version == FEATURE_VERSION
    assert len(set(SCALARS)) == len(SCALARS)


def test_time_gaps_are_powers_that_fit_the_reserved_slots(built):
    assert TIME_GAPS == (1, 2, 4, 8, 16, 64, 256)
    assert len(TIME_GAPS) == 7


def test_tick_cap_refuses_rather_than_truncates(built):
    machine, record, _ = built
    with pytest.raises(GraphTooLarge, match="cap"):
        build(machine, record, tick_cap=2)


def test_placements_do_not_change_the_shape(built):
    """A note changes features, never structure - so the same machine with and without a
    placement produces graphs the model can treat identically."""
    machine, record, graph = built
    empty = next(c for c in machine.cell_list if c not in machine.cells)
    noted = build(machine, record, placements=(Placement(empty, SLIME_SLOT),))
    assert noted.n_items == graph.n_items
    assert noted.n_edges == graph.n_edges
    assert len(noted.cell_items) == len(graph.cell_items)


def test_record_from_a_different_machine_is_refused(built):
    """A record silently belonging to another machine would poison every feature. Checked by
    block multiset, not by trusting the caller."""
    import json
    from pathlib import Path

    machine, _, _ = built
    other_path = (
        Path(__file__).resolve().parents[2]
        / "flying machines"
        / "json"
        / "simple_machine2.json"
    )
    other = Machine.from_candidate(json.loads(other_path.read_text(encoding="utf-8")))
    other_record = record_for(other.to_candidate(cid=0))
    with pytest.raises(ValueError, match="different block multisets"):
        build(machine, other_record)
