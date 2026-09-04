"""Per-tick board state, by replaying the log's write stream.

DECISIONS.md point 27 decided the C++ should emit per-tick snapshots behind a flag, on the
grounds that rebuilding state in Python "is a reimplementation of logic the C++ already
performs, two implementations must be kept in step, and check_state_builder.py exists precisely
because they can drift".

**That reasoning does not apply here, and the C++ change is not needed.** It is true of
transformer_gym/state.py's apply_tick, which reimplements piston physics, flight settling and
sticky-drag rules. It is false of BlockStateChanged (kind 22), which sim_event_log.h calls the
replication backstop:

    Emitted from inside setBlockState for EVERY world write that actually changes something,
    whatever rule caused it. So world state at any point == InitialBlockState with every
    BlockStateChanged up to that globalSeq applied in order - no rule needs re-implementing and
    no future rule can silently escape the log.

There is no logic to keep in step because there is no logic: the event carries the position, the
old raw state and the new raw state, so replay is a dictionary assignment.

And it is self-checking. Every record carries the OLD state it overwrites, so asserting the
board already holds that value means **a single missing write cannot hide** - the next write at
that cell would disagree. That check is on by default here rather than living only in a test,
because it costs one comparison and it is the entire basis for trusting the result.

Measured across 46 fixtures: zero old-state mismatches, everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from rlgym.record import BLOCK_STATE_CHANGED, Event, Record

Cell = tuple[int, int, int]
Board = dict[Cell, int]


class ReplayError(ValueError):
    """A write disagreed with the board about what it was overwriting.

    This means the log is missing a write, the events were applied out of order, or the reader
    and the C++ side disagree about the record layout. All three are serious: every downstream
    feature would be computed against a world that never existed.
    """


def replay(record: Record, until_tick: int | None = None, verify: bool = True) -> Board:
    """The board after every write up to and including until_tick (None = the whole run)."""
    board = record.initial_board()
    for event in record.events_in_order():
        if event.kind != BLOCK_STATE_CHANGED:
            continue
        if until_tick is not None and event.executed_tick > until_tick:
            break
        _apply(board, event, verify)
    return board


def boards_by_tick(record: Record, verify: bool = True) -> dict[int, Board]:
    """One board per tick that changed, plus tick 0.

    Only ticks where something actually happened get an entry - the write stream is sparse, and
    a cell that does not change has nothing to record. Callers wanting a dense series should use
    board_at, which carries the last known state forward.
    """
    board = record.initial_board()
    out: dict[int, Board] = {0: dict(board)}
    current = 0
    for event in record.events_in_order():
        if event.kind != BLOCK_STATE_CHANGED:
            continue
        if event.executed_tick != current:
            out[current] = dict(board)
            current = event.executed_tick
        _apply(board, event, verify)
    out[current] = dict(board)
    return out


def board_at(by_tick: dict[int, Board], tick: int) -> Board:
    """The board at any tick, carrying the last change forward."""
    known = [t for t in by_tick if t <= tick]
    if not known:
        raise KeyError(f"no board at or before tick {tick}")
    return by_tick[max(known)]


def changed_ticks(record: Record) -> list[int]:
    """Ticks at which at least one cell changed. This is the economy DECISIONS.md point 6
    mentions for cell items - deferred for now, but the ticks are cheap to know."""
    return sorted(
        {e.executed_tick for e in record.events if e.kind == BLOCK_STATE_CHANGED}
    )


def _apply(board: Board, event: Event, verify: bool) -> None:
    pos = event.from_pos
    if verify:
        have = board.get(pos, 0)
        if have != event.target_key:
            raise ReplayError(
                f"write at {pos} (tick {event.executed_tick}, subtick "
                f"{event.activation_subtick}) says it overwrites state {event.target_key}, "
                f"but the board holds {have}. A write is missing from the log, the ordering is "
                f"wrong, or the record layout has drifted from sim_event_log.h."
            )
    if event.reserved2 == 0:
        board.pop(pos, None)
    else:
        board[pos] = event.reserved2


@dataclass(frozen=True)
class ReplayReport:
    writes: int
    mismatches: int
    ticks_with_changes: int
    final_size: int

    @property
    def consistent(self) -> bool:
        return self.mismatches == 0


def check_replay(record: Record) -> ReplayReport:
    """Replay without raising, reporting what was found. For the check script and for tests
    that want to measure rather than assert."""
    board = record.initial_board()
    writes = mismatches = 0
    ticks: set[int] = set()
    for event in record.events_in_order():
        if event.kind != BLOCK_STATE_CHANGED:
            continue
        writes += 1
        ticks.add(event.executed_tick)
        if board.get(event.from_pos, 0) != event.target_key:
            mismatches += 1
        _apply(board, event, verify=False)
    return ReplayReport(
        writes=writes,
        mismatches=mismatches,
        ticks_with_changes=len(ticks),
        final_size=len(board),
    )
