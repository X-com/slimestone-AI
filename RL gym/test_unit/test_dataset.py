"""The note patch and the Stage 0 targets - md/VERIFICATION.md.

The claim under test is the one the whole training budget rests on:

> **`apply_notes(graph, p)` is identical to `build(machine, record, placements=p)`.**

`BENCHMARKS.md` measures a graph build at 157 ms, so 207,935 training examples would be nine
hours of rebuilding. Patching instead makes it 31 builds and microseconds each - but only if the
patch really is the same thing. If it drifts, training silently uses a slightly different input
from the one every structural test checked, and nothing else in the suite would notice.
"""
from __future__ import annotations

import numpy as np
import pytest

from rlgym.config import Config
from rlgym.dataset import machine_data, reward_of
from rlgym.game import Placement
from rlgym.graph import apply_notes, build
from rlgym.labeller import Label, LabelSet
from rlgym.simlog import record_for

pytestmark = pytest.mark.slow

SLIME = 30
EXTENDED_PISTON = 6  # writes two cells: the body and its head


@pytest.fixture(scope="module")
def built(request):
    machine = request.getfixturevalue("engine")
    record = record_for(machine.to_candidate(cid=0))
    return machine, record, build(machine, record)


def _same(a, b) -> None:
    assert np.array_equal(a.note_type, b.note_type)
    assert np.array_equal(a.note_facing, b.note_facing)
    assert np.allclose(a.scalars, b.scalars)
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.block_type, b.block_type)


def test_the_note_patch_equals_a_rebuild(built):
    machine, record, graph = built
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    placement = (Placement(cell, SLIME),)
    _same(build(machine, record, placements=placement), apply_notes(graph, placement))


def test_the_patch_marks_both_cells_of_an_extended_piston(built):
    """Both cells are noted (Part 3). A patch that marked only the body would describe a machine
    with a floating piston head, which is not a machine."""
    machine, record, graph = built
    cell = next(
        c
        for c in machine.cell_list
        if c not in machine.cells
        and all(w not in machine.cells for w in Placement(c, EXTENDED_PISTON).written_cells())
    )
    placement = (Placement(cell, EXTENDED_PISTON),)
    patched = apply_notes(graph, placement)
    _same(build(machine, record, placements=placement), patched)
    noted = int((patched.note_type > 0).sum())
    assert noted == 2, f"expected the body and the head to be noted, got {noted}"


def test_patching_does_not_mutate_the_original(built):
    """The cache holds one graph per machine and hands it out repeatedly. An in-place patch
    would leave every later candidate carrying the previous one's note."""
    machine, _, graph = built
    before = graph.note_type.copy()
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    apply_notes(graph, (Placement(cell, SLIME),))
    assert np.array_equal(graph.note_type, before)
    assert int(graph.note_type.max()) == 0


def test_no_placement_is_a_no_op(built):
    _, _, graph = built
    assert apply_notes(graph, ()) is graph


def test_the_flag_is_on_every_tick_but_the_type_only_at_tick_zero(built):
    """Part 3's split. Without the per-tick flag a noted cell still reads "slime, moving east,
    push group of 5" at tick 5 - a description of a machine that will not exist."""
    from rlgym.graph import NOTE_SCALAR

    machine, _, graph = built
    cell = next(c for c in machine.cell_list if c not in machine.cells)
    patched = apply_notes(graph, (Placement(cell, SLIME),))
    items = graph.cell_index[cell]
    assert patched.note_type[items[0]] == SLIME + 1
    assert all(patched.note_type[item] == 0 for item in items[1:])
    assert all(patched.scalars[item, NOTE_SCALAR] == 1.0 for item in items)


# --- the Stage 0 targets ---------------------------------------------------------------


def _labelset(rewards, cargo=()):
    labels = [
        Label(
            action=action,
            cell=(0, 0, 0),
            slot=0,
            entry="stone",
            reward=reward,
            period=4,
            shift=(0, 0, 1),
            working=reward > 0,
            maybe_cargo=action in cargo,
        )
        for action, reward in rewards.items()
    ]
    return LabelSet(
        machine="x",
        radius=1,
        k=1,
        action_slots=0,
        legal_actions=len(labels),
        simulated=len(labels),
        duplicates=0,
        working=sum(1 for label in labels if label.reward > 0),
        labels=labels,
    )


def test_the_policy_target_is_uniform_over_working_moves(engine):
    """`pi` from brute force is the *exact* optimal policy - dense over every action, which no
    amount of search can produce."""
    labelset = _labelset({0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0})
    data = machine_data("simple_observer_engine", _relabel(labelset, engine))
    target = data.policy_target
    assert target.sum() == pytest.approx(1.0)
    assert target[0] == pytest.approx(0.5)
    assert target[2] == pytest.approx(0.5)
    assert target[1] == 0.0


def test_reward_cargo_moves_pi_and_z_together():
    """They are two views of one reward function. Letting them disagree would be a subtle bug
    that shows up as a value head calibrated against a policy trained on something else."""
    label = Label(0, (0, 0, 0), 0, "stone", 1.0, 4, (0, 0, 1), True, maybe_cargo=True)
    assert reward_of(label, 1.0) == 1.0
    assert reward_of(label, 0.1) == pytest.approx(0.1)
    non_cargo = Label(1, (0, 0, 0), 0, "stone", 1.0, 8, (0, 0, 2), True, maybe_cargo=False)
    assert reward_of(non_cargo, 0.1) == 1.0


def test_a_machine_where_nothing_works_has_no_policy_target(engine):
    """Part 5's edge case: skip the policy term rather than training toward a uniform target
    that claims every move is equally good when in fact none is."""
    labelset = _labelset({0: 0.0, 1: 0.0})
    data = machine_data("simple_observer_engine", _relabel(labelset, engine))
    assert not data.has_policy_target
    assert data.policy_target.sum() == 0.0


def _relabel(labelset, machine):
    labelset.action_slots = machine.action_count
    return labelset


def test_the_config_refuses_an_unknown_key():
    """A typo in a config is otherwise a silent no-op that looks exactly like the knob having
    no effect - which is the one conclusion the whole measurement plan must not reach falsely."""
    with pytest.raises(ValueError, match="unknown config key"):
        Config().merged({"reward_cargo": 0.1})  # it lives under "train", not at the top
    assert Config().merged({"train": {"reward_cargo": 0.1}}).train.reward_cargo == 0.1
