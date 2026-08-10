"""Proves a .simlog is complete enough to reconstruct the simulation. Run after any logging change.

    py "util tools/check_log_completeness.py"              # default fixture set
    py "util tools/check_log_completeness.py" tank submarine

Two independent assertions, because there are two different kinds of "complete" and passing the
first does not imply the second:

REPLAY - read the log front to back and rebuild the world.
    Every BlockStateChanged carries the OLD state it overwrites as well as the new one. So replay
    from InitialBlockState and assert the board already holds exactly that old state at every
    change. Any world write missing from the log makes the NEXT write at that cell disagree about
    what was there, so a single missing write cannot hide. This tests completeness directly rather
    than by sampling.

RESUME - rebuild the FULL simulator state from any cut point, which is what a model predicting the
    next event does at every step.
    A front-to-back reader survives gaps by carrying values forward in its own memory from events
    that have already scrolled past; a model cannot. So this forbids that shortcut for the one
    piece of state it matters for - World::movingBuckets, the blocks in flight. Mid-flight the
    destination cell holds only the id-36 placeholder, whose meta encodes the push DIRECTION and
    nothing about the payload, so the payload has to be stated on the event that starts the flight
    (BlockPushed.reserved2) or it is unknowable. This checks the promise against what actually
    landed, taken from the world, not from the claim restated.

    Two failures are possible and both are reported:
      UNANNOUNCED - a flight ended that no BlockPushed started (the reader never knew it existed)
      WRONG       - the promised payload is not what landed
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_simulation_data as vsd  # noqa: E402

BLOCK_PUSHED, BLOCK_SETTLED, BLOCK_STATE_CHANGED = 2, 18, 22

DEFAULT_FIXTURES = ["observer drop test", "simple_observer_engine", "simple_caterpillar",
                    "simple_no_sticky_loop", "movableRCA", "submarine", "tank",
                    "complex_machine", "1-wide with rails"]


def _events(data: bytes) -> list:
    footer = vsd.read_footer(data)
    out = []
    for entry in vsd.read_block_index(data, footer):
        out.extend(vsd.iter_block_events(data, entry))
    out.sort(key=lambda e: e.globalSeq)
    return out


def check_replay(name: str, data: bytes, events: list) -> int:
    initial = vsd.read_initial_state(data, vsd.read_footer(data))
    board = {(s.x, s.y, s.z): s.rawState for s in initial}
    pending = {}
    changes = stale = settle_ok = settle_bad = 0

    for ev in events:
        if ev.kind == BLOCK_SETTLED:
            # Checked one step late on purpose: a settle is emitted immediately BEFORE its own
            # setBlockState so it precedes the cascade it triggers, so the write lands later.
            pending[(ev.toX, ev.toY, ev.toZ)] = ev.neighborSourceBlockId
            continue
        if ev.kind != BLOCK_STATE_CHANGED:
            continue
        pos = (ev.fromX, ev.fromY, ev.fromZ)
        old, new = ev.targetKey, ev.extraWord
        changes += 1

        have = board.get(pos, 0)
        if have != old:
            stale += 1
            if stale <= 3:
                print(f"    GAP {name} at {pos}: log says it overwrote {old & 0xFF}:{old >> 8}, "
                      f"replay holds {have & 0xFF}:{have >> 8} - a write is missing from the log")

        if new == 0:
            board.pop(pos, None)
        else:
            board[pos] = new

        want = pending.pop(pos, None)
        if want is not None:
            if (new & 0xFF) == want:
                settle_ok += 1
            else:
                settle_bad += 1
                if settle_bad <= 3:
                    print(f"    SETTLE MISMATCH {name} at {pos}: expected id {want}, got {new & 0xFF}")

    print(f"  {name:24} REPLAY writes={changes:>7}  old-state agree={changes - stale}/{changes}"
          f"  settles landed={settle_ok}/{settle_ok + settle_bad}")
    return stale + settle_bad


def check_resume(name: str, events: list) -> int:
    inflight = {}   # (pushGroupId, dest) -> promised raw state word
    awaiting = {}   # dest -> promised state, set by a settle, resolved by the next write there
    ok = unannounced = wrong = 0

    for ev in events:
        if ev.kind == BLOCK_PUSHED:
            inflight[(ev.pushGroupId, (ev.toX, ev.toY, ev.toZ))] = ev.extraWord
        elif ev.kind == BLOCK_SETTLED:
            dest = (ev.toX, ev.toY, ev.toZ)
            key = (ev.pushGroupId, dest)
            if key not in inflight:
                unannounced += 1
                if unannounced <= 3:
                    print(f"    UNANNOUNCED {name}: settle at {dest} grp={ev.pushGroupId} "
                          f"(id {ev.neighborSourceBlockId}) - no BlockPushed ever started it")
                continue
            awaiting[dest] = inflight.pop(key)
        elif ev.kind == BLOCK_STATE_CHANGED:
            promised = awaiting.pop((ev.fromX, ev.fromY, ev.fromZ), None)
            if promised is None:
                continue
            if promised == ev.extraWord:
                ok += 1
            else:
                wrong += 1
                if wrong <= 3:
                    print(f"    WRONG {name} at ({ev.fromX},{ev.fromY},{ev.fromZ}): promised "
                          f"{promised & 0xFF}:{promised >> 8}, got "
                          f"{ev.extraWord & 0xFF}:{ev.extraWord >> 8}")

    total = ok + wrong + unannounced
    extra = ("" if not unannounced else f"  UNANNOUNCED={unannounced}") \
            + ("" if not wrong else f"  WRONG={wrong}")
    print(f"  {name:24} RESUME flights={total:>7}  payload predicted={ok}/{total}{extra}")
    return unannounced + wrong


def main() -> int:
    names = sys.argv[1:] or DEFAULT_FIXTURES
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for n in names:
            data = vsd.run_fixture(n, wd).read_bytes()
            events = _events(data)
            bad += check_replay(n, data, events)
            bad += check_resume(n, events)
    print()
    if bad == 0:
        print("PASS - the log alone reproduces the world, and every flight is predictable from it")
        return 0
    print(f"FAIL - {bad} mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
