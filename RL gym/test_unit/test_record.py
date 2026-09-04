"""Decoding a .simlog - md/VERIFICATION.md.

Struct sizes and field meanings are asserted against literals taken from sim_event_log.h, which
has static_asserts for the same numbers on the C++ side. If the two ever disagree, one of them
is wrong and this says so instead of silently misreading every record.
"""
from __future__ import annotations

import pytest

from rlgym.record import (
    BLOCK_STATE_CHANGED,
    FORMAT_VERSION,
    MAGIC,
    N_KINDS,
    Record,
    SimLogError,
    _EVENT,
    _INITIAL,
    pack_pos,
    read_footer,
    unpack_pos,
)
from rlgym.simlog import record_for

pytestmark = pytest.mark.slow


def test_record_sizes_match_the_cpp_static_asserts():
    """sim_event_log.h: static_assert(sizeof(SimEvent) == 96) and
    static_assert(sizeof(BlockIndexEntry) == 32)."""
    assert _EVENT.size == 96
    assert _INITIAL.size == 32


def test_position_packing_round_trips():
    """packed_pos.h: three signed 21-bit fields in one uint64."""
    for pos in [(0, 0, 0), (1, 64, 2), (-7, 65, 300), (-1048576, 0, 1048575)]:
        assert unpack_pos(pack_pos(*pos)) == pos


def test_bad_magic_is_refused():
    with pytest.raises(SimLogError, match="magic"):
        read_footer(b"\x00" * 400)


def test_footer_reports_the_expected_version(engine):
    record = record_for(engine.to_candidate(cid=0))
    assert record.footer["magic"] == MAGIC
    assert record.footer["formatVersion"] == FORMAT_VERSION


def test_base_machine_record_matches_what_design_recorded(engine):
    """DESIGN.md measured simple_observer_engine at 6 blocks and 100 events. Pinned so a change
    in the simulator's logging shows up here rather than downstream."""
    record = record_for(engine.to_candidate(cid=0))
    assert record.footer["initialCount"] == 6
    assert record.footer["eventCount"] == 100


def test_summary_agrees_with_the_stdout_verdict(engine):
    """The record footer and the JSON verdict are two separate code paths reporting the same
    run. They must agree, or one of them is lying about the machine."""
    from rlgym.sim import SimulatorProcess

    record = record_for(engine.to_candidate(cid=0))
    with SimulatorProcess() as proc:
        verdict = proc.simulate(engine.to_candidate(cid=0))

    assert bool(record.summary.valid_cycle) == verdict["validCycle"]
    assert record.summary.period == verdict["period"]
    assert record.summary.net_shift == (
        verdict["finalShift"]["x"],
        verdict["finalShift"]["y"],
        verdict["finalShift"]["z"],
    )
    assert record.summary.total_ticks == verdict["ticks"]
    assert record.summary.block_count == len(engine.cells)


def test_initial_state_matches_the_machine_we_sent(engine):
    """Positions are translated by the simulator (the fixture sits at y=0-1, the log at y=64-65),
    so identity is checked by the relative layout and the state words, not absolute position."""
    record = record_for(engine.to_candidate(cid=0))
    sent = sorted(engine.cells.values())
    got = sorted(block.raw_state for block in record.initial)
    assert sent == got

    def normalise(cells):
        xs = min(c[0] for c in cells)
        ys = min(c[1] for c in cells)
        zs = min(c[2] for c in cells)
        return sorted((c[0] - xs, c[1] - ys, c[2] - zs) for c in cells)

    assert normalise(engine.cells) == normalise([b.pos for b in record.initial])


def test_exactly_one_block_is_the_trigger(engine):
    record = record_for(engine.to_candidate(cid=0))
    triggers = [block for block in record.initial if block.is_trigger]
    assert len(triggers) == 1


def test_events_in_order_is_a_strict_ordering(engine):
    """Events are stored grouped by subject block, not chronologically, so anything replaying
    the run must sort. If activationSubtick were not unique the ordering would be ambiguous and
    replay would be non-deterministic."""
    record = record_for(engine.to_candidate(cid=0))
    ordered = record.events_in_order()
    subticks = [e.activation_subtick for e in ordered]
    assert subticks == sorted(subticks)
    assert len(set(subticks)) == len(subticks)


def test_every_event_kind_is_known(engine):
    record = record_for(engine.to_candidate(cid=0))
    for event in record.events:
        assert 0 <= event.kind < N_KINDS


def test_block_state_changed_carries_position_old_and_new(engine):
    """simulator.cpp:2074 sets from == to == the written position, targetKey = old state,
    reserved2 = new state. Everything in boards.py rests on those three."""
    record = record_for(engine.to_candidate(cid=0))
    writes = [e for e in record.events if e.kind == BLOCK_STATE_CHANGED]
    assert writes
    for event in writes:
        assert event.from_pos == event.to_pos
        assert event.target_key != event.reserved2  # a write that changes nothing is not logged
