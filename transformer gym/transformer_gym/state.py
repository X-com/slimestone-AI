"""Incremental "sufficient statistic" state builder - the SimState an autoregressive model would
condition each tick's prediction on, and the state-update rule (apply_tick) that folds one tick's
events into the next SimState. See the tick-chunking plan: this is valid because the RESUME-
completeness check (util tools/check_log_completeness.py) already proved these four components are
a complete summary of everything before a tick boundary - nothing earlier can affect what happens
next except through them.

Node identity (block_type/facing/flags/movability/stickiness/is_trigger/is_air/rel_pos/rel_edges)
is the frozen t=0 snapshot from encode.py's build_graph(), one node per ORIGINAL block - these do
NOT evolve per tick in this pass. A block's raw type/facing/meta genuinely can change at runtime
(piston extension, rail reshaping, lamp lighting), but that live truth lives in `board` below, not
in the node tensors - `board` is the actual ground truth "what does this cell hold right now",
built the same way util tools/check_log_completeness.py's REPLAY check builds it, and is what
util tools/check_state_builder.py verifies against the independently replayed board.

The four dynamic components, and which event kinds move them (see sim_event_log.h for the field
semantics each of these relies on - all of it SDL10 or earlier, all of it already verified this
session by util tools/check_log_completeness.py):
  board                - BlockStateChanged (kind 22): targetKey=old, extraWord=new. Authoritative.
  in_flight             - BlockPushed (kind 2) opens an entry (extraWord=payload, SEF_SELF_ARM
                          entries included - a piston's own head/retract arm is a genuine
                          World::movingBuckets entry); BlockSettled/MovingBlockDropped (18/19)
                          closes it, keyed by (pushGroupId, destination).
  pending_block_events  - PistonQueued (kind 0) opens an entry UNLESS SEF_QUEUE_DEDUPED is set (in
                          which case nothing actually entered World::blockEvents);
                          PistonMoveExecuted (kind 1) closes it. checkForMove() and
                          sendQueuedBlockEvents() both run inside the same tickWorld() call, so a
                          SUCCESSFULLY resolved entry always closes same-tick - but a queued move
                          can also be silently REJECTED (fireBlockEvent/pistonEventReceived
                          returning false with no event emitted; this is the log's known
                          "block-event rejection is unlogged" gap), and nothing closes those. So
                          this field is a measured OVER-approximation of the true live queue, not
                          an exact one - confirmed by check_state_builder.py: movableRCA ends with
                          5 entries that were queued once and never resolved by any later
                          PistonMoveExecuted. Treat "nonzero" as "queued, outcome unlogged", not as
                          proof the move is still pending.
  pending_scheduled     - ScheduledTickCreated (kind 20) opens an entry (due tick = scheduledTick);
                          consumption is NOT separately logged but is fully derivable - due<=now
                          always leaves the queue (see ScheduledTickCreated's doc) - so apply_tick
                          purges due entries at the START of a tick, mirroring tickWorld()'s own
                          order (tickScheduledUpdates runs before the rest of the tick's events).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .encode import Graph, build_graph

BLOCK_STATE_CHANGED = 22
BLOCK_PUSHED = 2
BLOCK_SETTLED = 18
MOVING_BLOCK_DROPPED = 19
PISTON_QUEUED = 0
PISTON_MOVE_EXECUTED = 1
SCHEDULED_TICK_CREATED = 20

SEF_EXTEND = 1 << 0
SEF_QUEUE_DEDUPED = 1 << 7

Pos = tuple[int, int, int]


@dataclass(frozen=True)
class InFlight:
    payload: int          # the raw state word that will land - BlockPushed.reserved2/extraWord
    source: Pos
    created_tick: int


@dataclass
class SimState:
    tick: int
    # --- static node identity, one per original block (t=0 snapshot; see module docstring) ---
    block_type: object
    facing: object
    flags: object
    movability: object
    stickiness: object
    is_trigger: object
    is_air: object
    rel_pos: object
    rel_edges: object                 # [3, E] - see encode.py Graph/build_graph
    # --- the four-component sufficient statistic (all dynamic, evolve via apply_tick) ---
    board: dict[Pos, int] = field(default_factory=dict)
    in_flight: dict[tuple[int, Pos], InFlight] = field(default_factory=dict)
    pending_block_events: set[tuple[int, bool]] = field(default_factory=set)
    pending_scheduled: dict[tuple[Pos, int], int] = field(default_factory=dict)


def _graph_fields(graph: Graph) -> dict:
    return dict(
        block_type=graph.block_type, facing=graph.facing, flags=graph.flags,
        movability=graph.movability, stickiness=graph.stickiness, is_trigger=graph.is_trigger,
        is_air=graph.is_air, rel_pos=graph.rel_pos, rel_edges=graph.rel_edges,
    )


def initial_state(data: bytes, footer: dict, max_nodes: int = 300) -> SimState:
    """SimState at tick 0: node identity from build_graph(), board seeded from InitialBlockState,
    all queues/flights empty (nothing has happened yet)."""
    graph = build_graph(data, footer, max_nodes=max_nodes)
    board = {(s.x, s.y, s.z): s.rawState for s in graph.initial}
    return SimState(tick=0, board=board, **_graph_fields(graph))


def apply_tick(state: SimState, tick: int, events: list) -> SimState:
    """Folds one tick's events (already in globalSeq order - see util tools/simlog_ticks.py) into
    a new SimState. `events` must all share `activationTick == tick`. `tick` may be later than
    `state.tick + 1` if intervening ticks produced no events at all - due-scheduled-tick purging
    below still uses `tick` directly, so gaps are handled correctly either way."""
    board = dict(state.board)
    in_flight = dict(state.in_flight)
    pending_block_events = set(state.pending_block_events)
    # Consumption of a due scheduled tick isn't separately logged (see module docstring) - purge
    # first, mirroring tickWorld()'s order (tickScheduledUpdates before the rest of the tick).
    pending_scheduled = {k: v for k, v in state.pending_scheduled.items() if v > tick}

    for ev in events:
        if ev.activationTick != tick:
            raise ValueError(f"event at tick {ev.activationTick} passed to apply_tick(tick={tick})")

        if ev.kind == BLOCK_STATE_CHANGED:
            pos = (ev.fromX, ev.fromY, ev.fromZ)
            if ev.extraWord == 0:
                board.pop(pos, None)
            else:
                board[pos] = ev.extraWord

        elif ev.kind == BLOCK_PUSHED:
            dest = (ev.toX, ev.toY, ev.toZ)
            in_flight[(ev.pushGroupId, dest)] = InFlight(
                payload=ev.extraWord, source=(ev.fromX, ev.fromY, ev.fromZ), created_tick=tick,
            )

        elif ev.kind in (BLOCK_SETTLED, MOVING_BLOCK_DROPPED):
            dest = (ev.toX, ev.toY, ev.toZ)
            in_flight.pop((ev.pushGroupId, dest), None)

        elif ev.kind == PISTON_QUEUED:
            if ev.flags & SEF_QUEUE_DEDUPED:
                continue  # nothing actually entered World::blockEvents
            pending_block_events.add((ev.blockKey, bool(ev.flags & SEF_EXTEND)))

        elif ev.kind == PISTON_MOVE_EXECUTED:
            pending_block_events.discard((ev.blockKey, bool(ev.flags & SEF_EXTEND)))

        elif ev.kind == SCHEDULED_TICK_CREATED:
            pos = (ev.fromX, ev.fromY, ev.fromZ)
            pending_scheduled[(pos, ev.neighborSourceBlockId)] = ev.scheduledTick

    return replace(
        state, tick=tick, board=board, in_flight=in_flight,
        pending_block_events=pending_block_events, pending_scheduled=pending_scheduled,
    )
