"""Projecting a .simlog into the shape the visualizer animates.

The Live Training page shows a discovered machine as a still life. Everything needed to make it
move is already produced and thrown away: the C++ simulator writes a full per-event record
(`simlog.records_for`), and `flyer-web-visualizer/src/lib/animatedScene.ts` already plays moves,
piston extension, powered timelines, push-flight transparency and destroyed blocks. The only
missing piece was the projection between them, which is this file.

**Every keyframe here is a real event's own executed tick and position.** Nothing is
interpolated, inferred from block positions, or re-derived by a second simulation - what plays
is what the simulator logged. That is the whole reason this is worth having: an animation that
agreed with a re-simulation but not with the record would be a very convincing lie.

Ported from `util tools/stream_to_visualizer.py:build_animation_record_from_bytes` rather than
imported, per RL gym's self-contained rule. The port is smaller than the original because
`record.py` reads every event up front, so grouping by `block_key` replaces the original's
per-block index walk.

Usage:
    py -c "from rlgym.animation import animate; print(animate(candidate)['terminationTick'])"
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from rlgym.blocks import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_PISTON_HEAD,
    BLOCK_STICKY_PISTON,
    make_state,
)
from rlgym.game import Candidate
from rlgym.record import (
    BLOCK_DESTROYED,
    BLOCK_POWERED_CHANGED,
    BLOCK_PUSHED,
    BLOCK_SETTLED,
    OBSERVER_FIRED,
    PISTON_MOVE_EXECUTED,
    PISTON_QUEUED,
    SCHEDULED_TICK_DROPPED,
    SEF_EXTEND,
    SEF_OBSERVER_ON,
    SEF_POWERED_ON,
    SEF_SELF_ARM,
    SEF_SUCCESS,
    Event,
    Record,
)
from rlgym.sim import SimConfig
from rlgym.simlog import records_for

_PISTONS = frozenset({BLOCK_PISTON, BLOCK_STICKY_PISTON})
# A lit lamp is a different block id, not a meta bit, so its "on" state is read off the id.
BLOCK_LIT_REDSTONE_LAMP = 124
# state_flags bit 1 (powered) or bit 2 (open) - rails, fence gates and trapdoors.
_ON_FLAGS = 0b110


def animation_record(record: Record) -> dict[str, Any]:
    """One decoded record -> the JSON `animatedScene.ts` plays. Pure; no simulator involved."""
    initial = record.initial
    index_of = {block.stable_key: i for i, block in enumerate(initial)}

    by_key: dict[int, list[Event]] = defaultdict(list)
    for event in record.events:
        by_key[event.block_key].append(event)

    blocks = [
        {
            "x": block.pos[0],
            "y": block.pos[1],
            "z": block.pos[2],
            # Facing only, deliberately: a piston's extended bit belongs to the extension
            # timeline below, which owns that state over time. `block.raw_state` is the upgrade
            # path if a palette with rail shapes ever needs its meta preserved here.
            "state": make_state(
                block.block_type_id, block.facing if 0 <= block.facing <= 5 else 0
            ),
        }
        for block in initial
    ]
    trigger_block = next((b for b in initial if b.is_trigger), None)
    trigger = (
        {"x": trigger_block.pos[0], "y": trigger_block.pos[1], "z": trigger_block.pos[2]}
        if trigger_block is not None
        else {"x": 0, "y": 0, "z": 0}
    )

    events: list[dict[str, Any]] = []
    last_tick = 0

    def note(event: Event, kind: str, i: int) -> None:
        nonlocal last_tick
        last_tick = max(last_tick, event.executed_tick)
        events.append(
            {
                "tick": event.executed_tick,
                "order": event.executed_subtick,
                "kind": kind,
                "blockIndex": i,
            }
        )

    # --- blocks relocating, and when they land ------------------------------------------
    moves: list[dict[str, Any]] = []
    for key, own in by_key.items():
        i = index_of.get(key)
        if i is None:
            continue
        # A push and its arrival are two events sharing a pushGroupId: BlockPushed when the block
        # leaves, BlockSettled when it lands - usually 2 ticks later, but sooner when a short
        # pulse cancels the move. Pairing them lets each keyframe carry its own real arrival
        # instead of a constant, which is what draws the in-flight transparency correctly.
        settled: dict[int, Event] = {}
        steps: list[dict[str, Any]] = []
        for event in own:
            if event.kind == BLOCK_SETTLED:
                settled[event.push_group_id] = event
                note(event, "blockSettled", i)
            elif event.kind == BLOCK_PUSHED:
                # The simulator logs a BlockPushed for a piston's own head extending and
                # animating back too, filed under the acting piston. Those do not relocate it -
                # rendering them would teleport the piston onto its own head.
                if event.flags & SEF_SELF_ARM:
                    continue
                steps.append(
                    {
                        "tick": event.executed_tick,
                        "order": event.executed_subtick,
                        "x": event.to_pos[0],
                        "y": event.to_pos[1],
                        "z": event.to_pos[2],
                        "group": event.push_group_id,
                    }
                )
                note(event, "blockPushed", i)
        for step in steps:
            arrival = settled.get(step.pop("group"))
            # No matching settle - destroyed in flight, or an older log - degrades to "arrived
            # instantly", so the block simply never renders as in-flight.
            step["arriveTick"] = arrival.executed_tick if arrival else step["tick"]
            step["arriveOrder"] = arrival.executed_subtick if arrival else step["order"]
        if steps:
            moves.append({"blockIndex": i, "steps": steps})

    # --- piston extension ----------------------------------------------------------------
    # A head is not a block in the initial state - it exists only while extended - so it cannot
    # be tracked as a moved block. Each piston instead gets its own timeline from its true t=0
    # state (state_flags bit 0) forward.
    extensions: list[dict[str, Any]] = []
    for i, block in enumerate(initial):
        if block.block_type_id not in _PISTONS:
            continue
        own = by_key.get(block.stable_key, ())
        steps = [{"tick": 0, "order": 0, "extended": bool(block.state_flags & 1)}]
        # PistonQueued(retract) is a FALLBACK, not a second source. It exists because a retract
        # that never reaches doPistonMove emits no PistonMoveExecuted, leaving the head extended
        # forever. When the executed event does arrive, taking both logs the same retraction
        # three times over - a queue, a dedup-suppressed requeue, and the execution - which is
        # what this machine actually produces: 8 "pistonRetract" events for 2 pistons in 10 ticks.
        executes_retract = any(
            e.kind == PISTON_MOVE_EXECUTED and e.flags & SEF_SUCCESS and not e.flags & SEF_EXTEND
            for e in own
        )
        for event in own:
            if event.kind == PISTON_QUEUED:
                # A queued EXTEND can still be blocked, so it is never a state change on its own;
                # only its successful execution is.
                if not event.flags & SEF_EXTEND and not executes_retract:
                    steps.append(
                        {
                            "tick": event.executed_tick,
                            "order": event.executed_subtick,
                            "extended": False,
                        }
                    )
                    note(event, "pistonRetract", i)
                continue
            if event.kind != PISTON_MOVE_EXECUTED:
                continue
            if event.flags & SEF_SUCCESS:
                steps.append(
                    {
                        "tick": event.executed_tick,
                        "order": event.executed_subtick,
                        "extended": bool(event.flags & SEF_EXTEND),
                    }
                )
                note(event, "pistonExtend" if event.flags & SEF_EXTEND else "pistonRetract", i)
            else:
                # A blocked push moves nothing, which is exactly why it is worth showing: the
                # stepper's job is to let you watch a piston try and fail, not do nothing.
                note(event, "pistonBlocked", i)
        extensions.append({"blockIndex": i, "steps": steps})

    # --- observers -----------------------------------------------------------------------
    # SEF_OBSERVER_ON separates the ON pulse from the OFF transition two ticks later; both are
    # the same event kind, so without the flag the lit window has to be guessed.
    for i, block in enumerate(initial):
        if block.block_type_id != BLOCK_OBSERVER:
            continue
        for event in by_key.get(block.stable_key, ()):
            if event.kind == OBSERVER_FIRED:
                note(event, "observerFired" if event.flags & SEF_OBSERVER_ON else "observerOff", i)

    # --- powered / open state, and destruction ---------------------------------------------
    # Not pre-filtered by block type: the subject of these is whatever occupies the position, so
    # the same generic walk as the moves above is the honest shape.
    powered: list[dict[str, Any]] = []
    for key, own in by_key.items():
        i = index_of.get(key)
        if i is None:
            continue
        block = initial[i]
        on = block.block_type_id == BLOCK_LIT_REDSTONE_LAMP or bool(block.state_flags & _ON_FLAGS)
        steps = [{"tick": 0, "order": 0, "on": on}]
        changed = False
        for event in own:
            if event.kind == BLOCK_DESTROYED:
                # reserved0 is the destroyed block's own id (simulator.cpp:logBlockDestroyed).
                # A piston head being removed - retracted, or orphaned - is not a block leaving
                # the machine, and the extension timeline already draws it. Reporting it would
                # make the viewer delete whatever block happens to sit at that position, which
                # on this fixture is the piston itself: it vanished at tick 8.
                if event.reserved0 == BLOCK_PISTON_HEAD:
                    continue
                note(event, "blockDestroyed", i)
            elif event.kind == BLOCK_POWERED_CHANGED:
                changed = True
                now_on = bool(event.flags & SEF_POWERED_ON)
                steps.append(
                    {"tick": event.executed_tick, "order": event.executed_subtick, "on": now_on}
                )
                note(event, "poweredOn" if now_on else "poweredOff", i)
            elif event.kind == SCHEDULED_TICK_DROPPED:
                # Diagnostic only - the scheduling collision it marks is intended behaviour.
                # Surfaced so an otherwise silent drop is visible in the trace.
                note(event, "scheduledTickDropped", i)
        if changed:
            powered.append({"blockIndex": i, "steps": steps})

    events.sort(key=lambda e: (e["tick"], e["order"]))
    return {
        "trigger": trigger,
        "blocks": blocks,
        "moves": moves,
        "extensions": extensions,
        "powered": powered,
        "events": events,
        "terminationTick": last_tick,
    }


def animate(candidate: Candidate, config: SimConfig | None = None) -> dict[str, Any]:
    """Simulate one candidate with logging on and project the record.

    Re-simulating rather than keeping every discovery's record is the point: a machine is fully
    described by its blocks, so this is reproducible from what the viewer already holds, and the
    training loop pays nothing for animations nobody asks to watch.
    """
    with records_for([candidate], config) as records:
        record = records.get(candidate["id"])
        if record is None:
            raise ValueError(f"the simulator produced no record for candidate {candidate['id']}")
        return animation_record(record)
