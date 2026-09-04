"""The exhaustive labeller and the cargo split - md/ALPHAZERO.md Part 8 (Milestone 1).

The measured result on simple_observer_engine at R=1 is pinned here as literals. These are not
predictions - they were produced by running the labeller - but pinning them means a change in
the palette, the shell rule, the encoder or the simulator shows up as a failing test rather
than as a quietly different training set.
"""
from __future__ import annotations

import pytest

from rlgym.game import GameState, Machine
from rlgym.labeller import enumerate_k1, label_k1, load_fixture


def test_enumerate_k1_matches_the_legal_mask(engine):
    actions = enumerate_k1(engine)
    mask = GameState(engine, k=1).legal_mask()
    assert len(actions) == sum(mask)
    assert [index for index, _ in actions] == [i for i, legal in enumerate(mask) if legal]


def test_enumerate_k1_never_yields_a_reserved_slot(engine):
    for _, placement in enumerate_k1(engine):
        assert placement.entry is not None


def test_load_fixture_by_name_and_by_path(engine_candidate):
    assert load_fixture("simple_observer_engine")["blocks"] == engine_candidate["blocks"]


@pytest.mark.slow
def test_milestone_one_numbers(engine_candidate):
    """The Milestone 1 measurement, pinned.

    The headline is deliberately two numbers, not one. 31.78% of single-block modifications
    leave the machine working - but 313 of those 314 share the base machine's exact period AND
    net shift, so essentially every success is a block riding along. The non-cargo rate is
    0.10%: one modification in 988.

    Both branches of the plan's decision table fire at once, which is the finding: the raw rate
    says the task is easy, the non-cargo rate says it is nearly impossible, and the gap between
    them is the trivial-extension problem arriving through a binary reward exactly as OPEN.md
    predicted.
    """
    result = label_k1("simple_observer_engine", radius=1, workers=12)

    assert result.diagnostics["base_blocks"] == 6
    assert result.diagnostics["candidate_cells"] == 26
    assert result.action_slots == 26 * 48 + 1
    assert result.legal_actions == 988

    assert result.diagnostics["base_period"] == 10
    assert result.diagnostics["base_shift"] == (1, 0, 0)

    assert result.working == 314
    assert result.maybe_cargo == 313
    assert result.non_cargo == 1
    assert 0.31 < result.success_rate < 0.33
    assert result.non_cargo_rate < 0.002

    # working and validCycle are separate fields; DESIGN.md chose validCycle. On this machine
    # they never disagree, which is worth knowing before either is relied on alone.
    assert result.diagnostics["working_validcycle_disagreements"] == 0
    # Refusals are legitimate reward-0 outcomes, not crashes: 21 modifications destroy the
    # trigger, either by overwriting it or by landing an extended piston head on it.
    assert sum(result.diagnostics["rejected_by_simulator"].values()) == 21
    assert not result.diagnostics["simulate_failures"]


@pytest.mark.slow
def test_removing_any_block_breaks_this_machine(engine_candidate):
    """Every one of the six blocks is load-bearing - air succeeds 0 times out of 6. That is
    what makes this a sound base machine: there is nothing already dead in it to trim."""
    result = label_k1("simple_observer_engine", radius=1, workers=12)
    removals = [label for label in result.labels if label.entry == "air"]
    assert len(removals) == 6
    assert all(label.reward == 0.0 for label in removals)


@pytest.mark.slow
def test_the_single_non_cargo_discovery_reverses_direction(engine_candidate):
    """The one genuinely non-trivial single-block modification: an extended sticky piston at
    (-1,0,2) facing east flips the machine from flying +x to flying -x, at the same period."""
    result = label_k1("simple_observer_engine", radius=1, workers=12)
    discoveries = [
        label for label in result.labels if label.reward > 0.0 and not label.maybe_cargo
    ]
    assert len(discoveries) == 1
    found = discoveries[0]
    assert found.cell == (-1, 0, 2)
    assert found.entry == "sticky_piston_extended_east"
    assert found.shift == (-1, 0, 0)
    assert found.period == 10


@pytest.mark.slow
def test_success_is_driven_by_cell_not_by_block(engine_candidate):
    """A finding that shapes what the model has to learn: 11 of the 26 candidate cells accept
    almost any block, and the remaining 15 accept almost none. Position dominates identity,
    which is why the graph must carry the geometric facts of point 3."""
    result = label_k1("simple_observer_engine", radius=1, workers=12)
    per_cell: dict[tuple[int, int, int], list[float]] = {}
    for label in result.labels:
        per_cell.setdefault(label.cell, []).append(label.reward)
    rates = sorted((sum(v) / len(v) for v in per_cell.values()), reverse=True)
    permissive = [rate for rate in rates if rate > 0.5]
    hostile = [rate for rate in rates if rate < 0.05]
    assert len(permissive) == 11
    assert len(hostile) == 15
    assert len(permissive) + len(hostile) == len(per_cell)


@pytest.mark.slow
def test_base_machine_must_itself_be_valid():
    """Labelling against a broken reference would make every cargo comparison meaningless, so
    it is refused loudly rather than measured."""
    with pytest.raises(ValueError, match="valid cycle"):
        label_k1("test_flyer_2_doesnt_loop", radius=1, workers=2)
