"""Projecting a record into an animation - rlgym/animation.py.

These run the real simulator on `simple_observer_engine`, because the two bugs this file exists
to prevent are both invisible in a synthetic record: they are things the simulator really emits
that a plausible reading of the format mishandles.

The expected numbers below are stated from the machine itself - 2 pistons, a 10-tick cycle -
rather than copied from a run, so a regression fails here instead of quietly re-baselining.
"""
from __future__ import annotations

from collections import Counter

import pytest

from rlgym.animation import animate
from rlgym.blocks import BLOCK_PISTON, BLOCK_PISTON_HEAD, BLOCK_STICKY_PISTON

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine_animation(engine_candidate) -> dict:
    return animate(engine_candidate)


def _shape(blocks, trigger):
    """Positions relative to the machine's own minimum corner, plus where the trigger sits in
    that frame. The simulator builds the machine at its own base height, so absolute coordinates
    legitimately differ from the fixture's; the arrangement must not."""
    low = tuple(min(b[axis] for b in blocks) for axis in "xyz")
    cells = {tuple(b[axis] - low[i] for i, axis in enumerate("xyz")) for b in blocks}
    return cells, tuple(trigger[axis] - low[i] for i, axis in enumerate("xyz"))


def test_it_describes_the_machine_it_was_given(engine_animation, engine_candidate):
    """Same arrangement, same trigger. The viewer draws the animation from THIS record's own
    blocks - not from the compact one it already holds - so the two never have to agree on
    coordinates, but a projection that lost or moved a block would show a different machine
    than the one that was found."""
    assert len(engine_animation["blocks"]) == len(engine_candidate["blocks"])
    assert _shape(engine_animation["blocks"], engine_animation["trigger"]) == _shape(
        engine_candidate["blocks"], engine_candidate["trigger"]
    )
    assert engine_animation["terminationTick"] > 0


def test_every_piston_gets_a_timeline_starting_at_its_real_t0(engine_animation):
    """A head is not a block in the initial state, so extension cannot be tracked as movement -
    every piston needs its own timeline, and it has to start at tick 0 or a piston that begins
    extended renders retracted until its first event."""
    pistons = [
        i
        for i, b in enumerate(engine_animation["blocks"])
        if b["state"] & 0xFF in (BLOCK_PISTON, BLOCK_STICKY_PISTON)
    ]
    assert len(pistons) == 2
    assert {e["blockIndex"] for e in engine_animation["extensions"]} == set(pistons)
    for ext in engine_animation["extensions"]:
        assert ext["steps"][0]["tick"] == 0 and ext["steps"][0]["order"] == 0


def test_a_retraction_is_logged_once(engine_animation):
    """The simulator emits a queue, a dedup-suppressed requeue AND an execution for one retract.
    Taking all three gave 8 pistonRetract events for 2 pistons in a 10-tick cycle, and an
    extension timeline with three keyframes where the machine has two states.

    A cycling machine ends where it started, so extends and retracts must balance exactly."""
    kinds = Counter(e["kind"] for e in engine_animation["events"])
    assert kinds["pistonExtend"] == kinds["pistonRetract"] == 2

    for ext in engine_animation["extensions"]:
        states = [s["extended"] for s in ext["steps"]]
        assert states == [False, True, False], f"piston {ext['blockIndex']} toggled oddly"
        moments = [(s["tick"], s["order"]) for s in ext["steps"]]
        assert moments == sorted(moments), "keyframes must be in recorded order"


def test_a_retracting_head_is_not_a_destroyed_block(engine_animation):
    """BlockDestroyed fires for the head a retracting piston removes, keyed by POSITION - which
    resolves to whichever block started there. Reporting it made the viewer delete the piston
    itself at tick 8, halfway through a cycle it completes fine.

    Nothing in this machine is ever destroyed: it is a working flyer, and every block it has at
    tick 0 it still has at the end."""
    assert [e for e in engine_animation["events"] if e["kind"] == "blockDestroyed"] == []
    assert all(b["state"] & 0xFF != BLOCK_PISTON_HEAD for b in engine_animation["blocks"])


def test_a_pushed_block_carries_its_own_arrival(engine_animation):
    """A push is not instant - the destination holds a placeholder for ~2 ticks - and the viewer
    renders the block semi-transparent between departure and arrival. A constant arrival would
    look right on the common case and wrong on a push a short pulse cancels early."""
    assert engine_animation["moves"], "a flying machine moves its blocks"
    for move in engine_animation["moves"]:
        for step in move["steps"]:
            assert "group" not in step, "the pairing key is internal, not wire data"
            assert (step["arriveTick"], step["arriveOrder"]) >= (step["tick"], step["order"])


def test_events_are_in_recorded_order(engine_animation):
    """The /generator stepper walks this list one entry at a time; out of order, it would show
    effects before their causes."""
    moments = [(e["tick"], e["order"]) for e in engine_animation["events"]]
    assert moments == sorted(moments)
