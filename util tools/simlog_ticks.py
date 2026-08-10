"""Chronological, tick-grouped access to a decoded .simlog - the tick is the chunk unit the
tick-chunked training pipeline recurs over (see the tick-chunking plan, and
transformer gym/sim-log/loginfo.info). Builds on verify_simulation_data.py's per-block decode
functions rather than reimplementing the format.

On disk, events are stored block-major: SimEventLog::close() sorts the whole buffer to
(blockKey, activationSubtick) before writing (cpp simulator/src/sim_event_log.cpp), so
read_block_index()/iter_block_events() gives one block's history, pre-sorted by activationSubtick,
in O(1). A single global chronological stream needs a k-way merge across every block's already-
sorted run - cheap (O(N log blockCount)), not a full re-sort of N events. That merge is exactly
what iter_events_chronological does.

iter_ticks groups that merged stream by activationTick, which is monotonic non-decreasing in
globalSeq order: the simulator's outer loop is a plain `for (tick = 1; tick <= maxTicks; ++tick)`
(cpp simulator/src/simulator.cpp) that never rewinds, and activationSubtick/globalSeq is a single
counter incremented in emission order throughout that loop - confirmed by direct reading of the
tick loop and every SimEventLog::push() call site.
"""
from __future__ import annotations

import heapq
import itertools
import sys
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_simulation_data import SimEvent, iter_block_events, read_block_index  # noqa: E402


def _merge_event_streams(streams: list[Iterable[SimEvent]]) -> Iterator[SimEvent]:
    """The actual k-way merge, factored out so the self-check can exercise it directly on
    synthetic per-block streams without needing a real .simlog file."""
    yield from heapq.merge(*streams, key=lambda ev: ev.globalSeq)


def _group_by_tick(events: Iterable[SimEvent]) -> Iterator[tuple[int, list[SimEvent]]]:
    for tick, group in itertools.groupby(events, key=lambda ev: ev.activationTick):
        yield tick, list(group)


def iter_events_chronological(data: bytes, footer: dict) -> Iterator[SimEvent]:
    """Every event in the log, in true globalSeq order, k-way merged across all per-block runs."""
    entries = read_block_index(data, footer)
    streams = [iter_block_events(data, e) for e in entries if e.eventCount]
    yield from _merge_event_streams(streams)


def iter_ticks(data: bytes, footer: dict) -> Iterator[tuple[int, list[SimEvent]]]:
    """(activationTick, events) pairs, in tick order, each events list in globalSeq order."""
    yield from _group_by_tick(iter_events_chronological(data, footer))


def _self_check() -> None:
    """No exe/fixture needed: two synthetic per-block streams (already sorted by globalSeq, as
    iter_block_events always yields) with interleaved ticks - asserts the merge reproduces true
    chronological order and that tick-grouping is exact."""
    def ev(seq: int, tick: int) -> SimEvent:
        e = object.__new__(SimEvent)
        e.globalSeq = seq
        e.activationTick = tick
        return e

    block_a = [ev(0, 0), ev(2, 0), ev(5, 1), ev(9, 2)]
    block_b = [ev(1, 0), ev(3, 1), ev(4, 1), ev(6, 1), ev(7, 2), ev(8, 2)]

    merged = list(_merge_event_streams([iter(block_a), iter(block_b)]))
    seqs = [e.globalSeq for e in merged]
    assert seqs == list(range(10)), f"expected 0..9 chronological, got {seqs}"

    ticks = list(_group_by_tick(iter(merged)))
    tick_numbers = [t for t, _ in ticks]
    assert tick_numbers == [0, 1, 2], f"expected ticks [0,1,2], got {tick_numbers}"
    counts = {t: len(evs) for t, evs in ticks}
    assert counts == {0: 3, 1: 4, 2: 3}, f"wrong per-tick counts: {counts}"
    for _, evs in ticks:
        got = [e.globalSeq for e in evs]
        assert got == sorted(got), f"tick events not in globalSeq order: {got}"

    print("self-check PASS")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        print(__doc__)
