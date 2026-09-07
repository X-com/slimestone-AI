"""The stop action and the reward that prices it - configs/stop_probe.json.

Three switchable parts, one test group each:

    search.allow_stop        stop becomes reachable, so k is a ceiling not a length
    search.k                 already a config field; only the ceiling behaviour is new
    loop.functional_reward   a working machine is priced by what survives trimming

**Every default must reproduce today's behaviour exactly**, and that is what most of this file
asserts. A knob whose "off" position is not the old behaviour cannot be used to measure
anything, because every comparison against it would be confounded.
"""
from __future__ import annotations

import numpy as np
import pytest

from rlgym.config import Config, LoopConfig, SearchConfig
from rlgym.function import Trim, graded_reward, surviving_fraction
from rlgym.game import GameState, Placement, is_stop, legal_mask, stop_index

SLIME = 30


# --- part 1: the stop action -----------------------------------------------------------


def test_stop_is_masked_by_default(engine):
    """The default has to be the old behaviour: stop unreachable, k an exact edit length."""
    mask = GameState(engine, k=2).legal_mask()
    assert mask[stop_index(engine.cell_list)] is False
    assert not SearchConfig().allow_stop


def test_stop_is_legal_when_allowed(engine):
    mask = GameState(engine, k=2, allow_stop=True).legal_mask()
    assert mask[stop_index(engine.cell_list)] is True


def test_stop_does_not_move_when_the_machine_changes(engine):
    """The stop slot is last, so its index depends on the cell count - and a mask built for one
    machine applied to another would silently mask a real placement instead."""
    assert stop_index(engine.cell_list) == len(engine.cell_list) * 48
    assert len(legal_mask(engine.cells, engine.cell_list)) == stop_index(engine.cell_list) + 1


def test_stopping_ends_the_episode_below_k(engine):
    """The whole point: k is a ceiling the model may stop short of."""
    state = GameState(engine, k=8, allow_stop=True)
    cell = next(c for c in engine.cell_list if c not in engine.cells)
    state = state.step(Placement(cell, SLIME))
    assert not state.is_terminal, "one placement of eight is not terminal"

    stopped = state.stop()
    assert stopped.is_terminal
    assert len(stopped.placements) == 1, "stopping must not invent or drop a placement"
    assert stopped.current_cells() == state.current_cells()


def test_advance_routes_stop_and_placements_alike(engine):
    """`advance` is the only thing that knows one action index is not a placement. If it did
    not exist, every one of the four call sites would need its own guard."""
    state = GameState(engine, k=8, allow_stop=True)
    cell_index = next(
        i for i, c in enumerate(engine.cell_list) if c not in engine.cells
    )
    placed = state.advance(cell_index * 48 + SLIME)
    assert len(placed.placements) == 1 and not placed.stopped

    ended = state.advance(stop_index(engine.cell_list))
    assert ended.stopped and not ended.placements


def test_stopping_at_depth_zero_is_allowed(engine):
    """Deliberately NOT banned - it is the degenerate policy the reward is supposed to price,
    and hiding it behind a rule would make the probe unable to observe its own failure."""
    state = GameState(engine, k=8, allow_stop=True)
    assert state.legal_mask()[stop_index(engine.cell_list)] is True
    assert state.stop().current_cells() == engine.cells


def test_a_terminal_state_cannot_be_stopped(engine):
    with pytest.raises(ValueError):
        GameState(engine, k=1, allow_stop=True).step(
            Placement(engine.cell_list[0], SLIME)
        ).stop()


# --- part 3: the graded reward ---------------------------------------------------------


def _trim(removed):
    return Trim(cells={}, removed=list(removed), added_is_load_bearing=not removed,
                signature_length=4, simulator_calls=0)


def test_weight_zero_is_exactly_the_old_binary_reward():
    """The off position. Anything working scores 1.0 whatever it is made of."""
    for removed, added in ((set(), {(0, 0, 0)}), ({(0, 0, 0)}, {(0, 0, 0)}), (set(), set())):
        assert graded_reward(_trim(removed), set(added), 0.0) == 1.0


def test_a_fully_redundant_addition_scores_the_floor():
    """Both added blocks stripped, so nothing was found. At w=1 that is worth zero."""
    added = {(0, 0, 0), (1, 0, 0)}
    assert graded_reward(_trim(added), added, 1.0) == 0.0
    assert graded_reward(_trim(added), added, 0.5) == 0.5


def test_a_fully_load_bearing_addition_still_scores_one():
    added = {(0, 0, 0), (1, 0, 0)}
    assert graded_reward(_trim(set()), added, 1.0) == 1.0


def test_the_reward_is_graded_in_between():
    added = {(0, 0, 0), (1, 0, 0)}
    assert graded_reward(_trim({(0, 0, 0)}), added, 1.0) == pytest.approx(0.5)


def test_stopping_at_depth_zero_is_priced_at_the_floor():
    """The measurement this whole change exists for. An episode that stopped immediately added
    nothing, so it survives nothing, so at w=1 the free 1.0 it used to collect becomes 0.0."""
    assert surviving_fraction(_trim(set()), set()) == 0.0
    assert graded_reward(_trim(set()), set(), 1.0) == 0.0
    assert graded_reward(_trim(set()), set(), 0.0) == 1.0, "off must stay off"


def test_weight_above_one_is_clamped():
    """A config typo of 10.0 must not produce negative rewards, which would rank a working
    machine below a broken one and invert the whole search."""
    added = {(0, 0, 0)}
    assert graded_reward(_trim(added), added, 10.0) == 0.0


# --- the config surface ----------------------------------------------------------------


def test_every_default_reproduces_todays_behaviour():
    assert SearchConfig().allow_stop is False
    assert LoopConfig().functional_reward == 0.0
    assert SearchConfig().k == 2


def test_the_probe_config_turns_all_three_on():
    config = Config.load("configs/stop_probe.json")
    assert config.search.allow_stop is True
    assert config.search.k == 8
    assert config.loop.functional_reward == 1.0


# --- the round row ---------------------------------------------------------------------


def _totals(**overrides):
    from rlgym.loop import SOURCES, TALLIED

    totals = {source: dict.fromkeys(TALLIED, 0) for source in SOURCES}
    for source, values in overrides.items():
        totals[source].update(values)
    return totals


def test_the_round_row_reports_stopping():
    """`stopped` and `mean_depth` are the only evidence of what the stop action did, so they
    have to survive the per-source summing rather than being computed per episode and lost."""
    from rlgym.loop import Loop

    head = Loop._headline(
        _totals(
            top={"calls": 100, "load_bearing": 2, "stopped": 3, "depth": 6, "episodes": 5},
            sampled={"calls": 100, "load_bearing": 1, "stopped": 1, "depth": 9, "episodes": 3},
        )
    )
    assert head["stopped"] == 4
    assert head["mean_depth"] == round(15 / 8, 2)  # the row rounds; 8 episodes, 15 blocks


def test_grading_calls_are_not_charged_to_the_search():
    """The headline is discoveries per simulator call. Folding the price of grading into
    `calls` would make a graded run look worse at finding things when all that changed is what
    it paid to know."""
    from rlgym.loop import Loop

    head = Loop._headline(
        _totals(top={"calls": 1000, "load_bearing": 5, "grading_calls": 600, "episodes": 1})
    )
    assert head["model"] == 5.0, "grading calls leaked into the denominator"
    assert head["grading_calls"] == 600


def test_a_default_round_row_still_has_the_old_fields():
    from rlgym.loop import Loop

    head = Loop._headline(_totals(top={"calls": 200, "load_bearing": 1, "episodes": 1}))
    assert head["model"] == 5.0
    assert head["stopped"] == 0 and head["mean_depth"] == 0.0
