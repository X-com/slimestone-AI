"""Verifies transformer_gym/state.py's apply_tick against ground truth, at EVERY tick boundary of
every fixture, not sampled - see the tick-chunking plan's Phase 4. This is what makes "the compact
carried-forward SimState is a sufficient statistic" (the plan's central claim) a falsifiable test
rather than an assumption.

Two assertions, both re-derived independently here from the raw event stream rather than imported
from state.py or check_log_completeness.py, so a bug in one file can't hide behind another:

BOARD   - the world-write half. Walks BlockStateChanged in globalSeq order exactly like
    check_log_completeness.py's REPLAY check, but stops at each tick boundary and diffs the
    resulting board against what apply_tick's SimState.board holds at that same tick.

FLIGHTS - the in-flight half. At each tick boundary, computes the true "still open" flight set as
    (all BlockPushed so far) minus (all BlockSettled/MovingBlockDropped so far), keyed the same way
    state.py keys it - (pushGroupId, destination) - and diffs both the key set AND each entry's
    payload against SimState.in_flight.

Usage:
    py "util tools/check_state_builder.py" [fixture_name ...]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_simulation_data as vsd  # noqa: E402
from simlog_ticks import iter_ticks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer gym"))
from transformer_gym import state as st  # noqa: E402

BLOCK_STATE_CHANGED = 22
BLOCK_PUSHED = 2
BLOCK_SETTLED = 18
MOVING_BLOCK_DROPPED = 19


def check(name: str, data: bytes, footer: dict) -> tuple[int, int]:
    true_board: dict[tuple[int, int, int], int] = {
        (s.x, s.y, s.z): s.rawState for s in vsd.read_initial_state(data, footer)
    }
    true_flight: dict[tuple[int, tuple[int, int, int]], int] = {}  # key -> payload

    built = st.initial_state(data, footer, max_nodes=1_000_000)

    board_mismatches = 0
    flight_mismatches = 0
    ticks_checked = 0

    for tick, events in iter_ticks(data, footer):
        for ev in events:
            if ev.kind == BLOCK_STATE_CHANGED:
                pos = (ev.fromX, ev.fromY, ev.fromZ)
                if ev.extraWord == 0:
                    true_board.pop(pos, None)
                else:
                    true_board[pos] = ev.extraWord
            elif ev.kind == BLOCK_PUSHED:
                dest = (ev.toX, ev.toY, ev.toZ)
                true_flight[(ev.pushGroupId, dest)] = ev.extraWord
            elif ev.kind in (BLOCK_SETTLED, MOVING_BLOCK_DROPPED):
                dest = (ev.toX, ev.toY, ev.toZ)
                true_flight.pop((ev.pushGroupId, dest), None)

        built = st.apply_tick(built, tick, events)
        ticks_checked += 1

        if built.board != true_board:
            board_mismatches += 1
            if board_mismatches <= 3:
                only_true = set(true_board) - set(built.board)
                only_built = set(built.board) - set(true_board)
                diff_val = {p for p in set(true_board) & set(built.board)
                           if true_board[p] != built.board[p]}
                print(f"    BOARD MISMATCH {name} tick={tick}: "
                      f"missing_from_built={len(only_true)} extra_in_built={len(only_built)} "
                      f"value_diff={len(diff_val)}")

        built_flight = {k: v.payload for k, v in built.in_flight.items()}
        if built_flight != true_flight:
            flight_mismatches += 1
            if flight_mismatches <= 3:
                only_true = set(true_flight) - set(built_flight)
                only_built = set(built_flight) - set(true_flight)
                diff_val = {k for k in set(true_flight) & set(built_flight)
                           if true_flight[k] != built_flight[k]}
                print(f"    FLIGHT MISMATCH {name} tick={tick}: "
                      f"missing_from_built={len(only_true)} extra_in_built={len(only_built)} "
                      f"payload_diff={len(diff_val)}")
                if only_true:
                    print(f"        e.g. still-open in ground truth but not in built: {next(iter(only_true))}")
                if only_built:
                    print(f"        e.g. built thinks open but ground truth says closed: {next(iter(only_built))}")

    still_open = len(true_flight)
    print(f"  {name:24} ticks={ticks_checked:>6}  board_ok={ticks_checked - board_mismatches}/{ticks_checked}"
          f"  flight_ok={ticks_checked - flight_mismatches}/{ticks_checked}"
          f"  still_open_at_end={still_open}")
    return board_mismatches, flight_mismatches


def main() -> int:
    names = sys.argv[1:] or ["observer drop test", "simple_observer_engine", "simple_caterpillar",
                             "simple_no_sticky_loop", "movableRCA", "submarine", "tank",
                             "complex_machine", "1-wide with rails"]
    vsd.FIXTURE_DIR = Path(r"d:\Programmering\GitKraken\slimestone-AI\flying machines\json")
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for n in names:
            data = vsd.run_fixture(n, wd).read_bytes()
            footer = vsd.read_footer(data)
            b, f = check(n, data, footer)
            bad += b + f
    print()
    print("STATE BUILDER CORRECT - board and in-flight set match ground truth at every tick" if bad == 0
          else f"STATE BUILDER INCORRECT - {bad} tick-level mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
