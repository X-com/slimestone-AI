"""Per-tick board reconstruction - md/VERIFICATION.md, the third weighted test.

The claim under test is the one that removes the C++ snapshot change from the critical path:
replaying BlockStateChanged reproduces world state exactly, with no physics reimplemented and
no second implementation to keep in step.

Two independent confirmations, neither derived from the other:

  SELF-CHECK  every write carries the OLD state it overwrites, so asserting the board already
              holds it means a single missing write cannot hide - the next write at that cell
              would disagree. This is check_log_completeness.py's REPLAY argument, reused.
  ENDPOINT    on a machine whose run is exactly one period, the replayed board must equal the
              initial board translated by the summary's netShift. That compares the event
              stream against the RunSummary, which is computed by different code.
"""
from __future__ import annotations

import pytest

from rlgym.boards import (
    ReplayError,
    board_at,
    boards_by_tick,
    changed_ticks,
    check_replay,
    replay,
)
from rlgym.record import BLOCK_STATE_CHANGED, Record
from rlgym.simlog import record_for, records_for

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine_record(request):
    engine = request.getfixturevalue("engine")
    return record_for(engine.to_candidate(cid=0))


def test_replay_is_self_consistent(engine_record):
    report = check_replay(engine_record)
    assert report.mismatches == 0
    assert report.writes == 34
    assert report.consistent


def test_replay_raises_on_an_inconsistency(engine_record):
    """The self-check must actually fire. Corrupting one event's old-state field has to be
    caught, or the check is decoration."""
    events = list(engine_record.events)
    index = next(i for i, e in enumerate(events) if e.kind == BLOCK_STATE_CHANGED)
    import dataclasses

    broken = dataclasses.replace(
        engine_record,
        events=[
            e if i != index else dataclasses.replace(e, target_key=e.target_key + 1)
            for i, e in enumerate(events)
        ],
    )
    with pytest.raises(ReplayError, match="overwrites state"):
        replay(broken)


def test_endpoint_matches_the_summary(engine_record):
    """simple_observer_engine runs exactly one period (totalTicks == period == 10), so the
    board at the end of the log must be the initial board shifted by netShift."""
    assert engine_record.summary.total_ticks == engine_record.summary.period
    final = replay(engine_record)
    dx, dy, dz = engine_record.summary.net_shift
    expected = {
        (x + dx, y + dy, z + dz): state
        for (x, y, z), state in engine_record.initial_board().items()
    }
    assert final == expected


def test_board_at_tick_zero_is_the_initial_state(engine_record):
    by_tick = boards_by_tick(engine_record)
    assert by_tick[0] == engine_record.initial_board()


def test_boards_by_tick_agrees_with_replay_until(engine_record):
    """Two routes to the same board - the incremental series and a fresh replay with a cutoff."""
    by_tick = boards_by_tick(engine_record)
    for tick in changed_ticks(engine_record):
        assert board_at(by_tick, tick) == replay(engine_record, until_tick=tick)


def test_board_carries_forward_between_changes(engine_record):
    """The write stream is sparse: most ticks change nothing, and board_at must return the last
    known state rather than raising or returning empty."""
    by_tick = boards_by_tick(engine_record)
    ticks = changed_ticks(engine_record)
    between = ticks[0] + 1
    if between not in by_tick:
        assert board_at(by_tick, between) == by_tick[ticks[0]]


def test_changed_ticks_are_sparse(engine_record):
    """34 writes land on 5 distinct ticks out of a 10-tick cycle. That sparsity is what the
    deferred cell-item economy would exploit."""
    assert changed_ticks(engine_record) == [2, 4, 6, 8, 10]


def test_replay_holds_across_many_fixtures(engine):
    """The claim is not about one machine. Every small fixture that produces a record must
    replay with zero mismatches - measured at 46 of 46 when this was written."""
    import json
    from pathlib import Path

    from rlgym.game import Machine

    fixture_dir = Path(__file__).resolve().parents[2] / "flying machines" / "json"
    names = sorted(p.stem for p in fixture_dir.glob("*.json"))
    candidates = []
    for index, name in enumerate(names):
        data = json.loads((fixture_dir / f"{name}.json").read_text(encoding="utf-8"))
        if len(data["blocks"]) > 120:  # keep the test quick; the sweep script covers the rest
            continue
        candidates.append(Machine.from_candidate(data).to_candidate(cid=index))

    assert len(candidates) >= 10
    with records_for(candidates) as records:
        assert records
        for cid, record in records.items():
            report = check_replay(record)
            assert report.mismatches == 0, f"candidate {cid} had {report.mismatches} mismatches"
