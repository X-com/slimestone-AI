"""Verify the C++ simulator's binary "simulation_data" event log end-to-end.

Runs a few flying-machine fixtures through cpp_simulator_stream.exe with --simulation-data, then
decodes the resulting per-candidate binary log and prints, per block, that block's complete
self-contained event history in order. This is the human-readable check that the log captures
everything (see the plan's per-block coverage tables); a visualizer can later import the decode
functions (read_footer / read_block_index / iter_block_events) instead of reparsing.

Usage:
    py verify_simulation_data.py [fixture_name ...]      # defaults to a small representative set
    py verify_simulation_data.py --self-check            # decode round-trip assert, no exe needed

On-disk layout (must stay in sync with cpp simulator/src/sim_event_log.h):
    event section : N x SimEvent (72 bytes), grouped by block, each block's run in sim order
    block index   : B x BlockIndexEntry (32 bytes), sorted by originalKey
    footer        : 48 bytes, magic "SDL2", counts, blockIndexOffset
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "cpp simulator" / "build" / "cpp_simulator_stream.exe"
FIXTURE_DIR = REPO / "flying machines" / "json"
GENETIC_ML = REPO / "genetic algorithm"
MSYS_BIN = r"C:\msys64\ucrt64\bin"

# Small fixtures that together exercise every event kind: observer fire/activate + piston
# queue/execute + block-push (observer_engine), and redstone activate/deactivate a directly
# adjacent piston (upwards_engine).
DEFAULT_FIXTURES = ["simple_observer_engine", "simple_upwards_engine"]

_EVENT = struct.Struct("<QQQqqqIIIIBBBBBBH")   # 72 bytes
_INDEX = struct.Struct("<QQIIII")               # 32 bytes
_FOOTER = struct.Struct("<4sIQQIIIIQ")          # 48 bytes
assert _EVENT.size == 72 and _INDEX.size == 32 and _FOOTER.size == 48

KIND_NAMES = {
    0: "PistonQueued", 1: "PistonMoveExecuted", 2: "BlockPushed", 3: "ObserverFired",
    4: "ObserverActivated", 5: "RedstoneBlockAppeared", 6: "RedstoneBlockRemoved",
    7: "RedstoneActivatedPiston", 8: "RedstoneDeactivatedPiston",
}
CAUSE_NAMES = {0: "scheduled", 1: "facing-changed", 2: "observer-moved"}
DIR_NAMES = {0: "DOWN", 1: "UP", 2: "NORTH", 3: "SOUTH", 4: "WEST", 5: "EAST", 0xFF: "-"}
BLOCK_NAMES = {
    0: "air", 29: "sticky_piston", 33: "piston", 34: "piston_head", 36: "piston_ext",
    27: "golden_rail", 28: "detector_rail", 66: "rail", 157: "activator_rail",
    96: "trapdoor", 167: "iron_trapdoor", 107: "fence_gate", 152: "redstone_block",
    165: "slime", 218: "observer", 123: "redstone_lamp", 124: "lit_redstone_lamp",
    49: "obsidian", 1: "stone", 20: "glass",
}

SEF_EXTEND = 1 << 0
SEF_SUCCESS = 1 << 1
SEF_TARGET_PISTON = 1 << 4


def _unpack21(v: int) -> int:
    v &= 0x1FFFFF
    if v & 0x100000:
        v -= 0x200000
    return v


def unpack_pos(key: int) -> tuple[int, int, int]:
    return (_unpack21(key), _unpack21(key >> 21), _unpack21(key >> 42))


class SimEvent:
    __slots__ = ("blockKey", "actorKey", "targetKey", "activationTick", "scheduledTick",
                 "executedTick", "activationSubtick", "scheduledSubtick", "executedSubtick",
                 "pushGroupId", "kind", "direction", "flags", "attemptedAmount", "actualAmount")

    def __init__(self, raw: tuple):
        (self.blockKey, self.actorKey, self.targetKey, self.activationTick, self.scheduledTick,
         self.executedTick, self.activationSubtick, self.scheduledSubtick, self.executedSubtick,
         self.pushGroupId, self.kind, self.direction, self.flags, self.attemptedAmount,
         self.actualAmount, _r0, _r1) = raw


class BlockIndexEntry:
    __slots__ = ("originalKey", "currentKey", "firstEventIdx", "eventCount", "originalState")

    def __init__(self, raw: tuple):
        (self.originalKey, self.currentKey, self.firstEventIdx, self.eventCount,
         self.originalState, _r) = raw


def read_footer(data: bytes) -> dict:
    magic, version, event_count, block_index_off, block_count, ev_sz, blk_sz, _r, _r2 = \
        _FOOTER.unpack_from(data, len(data) - _FOOTER.size)
    if magic != b"SDL2":
        raise ValueError(f"bad magic {magic!r}")
    if ev_sz != _EVENT.size or blk_sz != _INDEX.size:
        raise ValueError(f"record size mismatch ev={ev_sz} blk={blk_sz}")
    return {"eventCount": event_count, "blockIndexOffset": block_index_off, "blockCount": block_count}


def read_block_index(data: bytes, footer: dict) -> list[BlockIndexEntry]:
    off = footer["blockIndexOffset"]
    return [BlockIndexEntry(_INDEX.unpack_from(data, off + i * _INDEX.size))
            for i in range(footer["blockCount"])]


def iter_block_events(data: bytes, entry: BlockIndexEntry):
    for i in range(entry.eventCount):
        yield SimEvent(_EVENT.unpack_from(data, (entry.firstEventIdx + i) * _EVENT.size))


def _fmt_event(ev: SimEvent) -> str:
    kind = KIND_NAMES.get(ev.kind, f"?{ev.kind}")
    parts = [f"t={ev.activationTick:<4} s={ev.activationSubtick:<4} {kind}"]
    if ev.kind in (0, 1, 2):  # piston/push
        ext = "extend" if ev.flags & SEF_EXTEND else "retract"
        parts.append(ext)
        parts.append(f"dir={DIR_NAMES.get(ev.direction, ev.direction)}")
        if ev.kind == 1:
            parts.append("moved" if ev.flags & SEF_SUCCESS else "BLOCKED")
        if ev.kind == 2:
            parts.append(f"by piston{unpack_pos(ev.actorKey)}->{unpack_pos(ev.targetKey)}")
        parts.append(f"amt {ev.attemptedAmount}->{ev.actualAmount}")
        parts.append(f"grp={ev.pushGroupId}")
        parts.append(f"sched(t={ev.scheduledTick},s={ev.scheduledSubtick})")
    elif ev.kind == 3:  # ObserverFired
        parts.append(f"cause={CAUSE_NAMES.get((ev.flags >> 2) & 3, '?')}")
    elif ev.kind == 4:  # ObserverActivated
        tgt = "piston" if ev.flags & SEF_TARGET_PISTON else "observer"
        parts.append(f"-> {tgt}{unpack_pos(ev.targetKey)}")
    elif ev.kind in (7, 8):  # redstone activate/deactivate
        parts.append(f"-> piston{unpack_pos(ev.targetKey)}")
    return "  " + " ".join(parts)


def dump_log(path: Path) -> int:
    data = path.read_bytes()
    footer = read_footer(data)
    index = read_block_index(data, footer)
    print(f"--- {path.name}: {footer['eventCount']} events across {footer['blockCount']} blocks ---")
    empty = 0
    for entry in index:
        state = entry.originalState
        name = BLOCK_NAMES.get(state & 0xFF, f"id{state & 0xFF}")
        pos = unpack_pos(entry.originalKey)
        cur = unpack_pos(entry.currentKey)
        moved = "" if entry.currentKey == entry.originalKey else f" (now {cur})"
        print(f"block {pos} {name} meta={state >> 8}{moved}: {entry.eventCount} event(s)")
        if entry.eventCount == 0:
            empty += 1
        for ev in iter_block_events(data, entry):
            print(_fmt_event(ev))
    return empty


def _load_id(json_path: Path) -> int:
    import json
    return int(json.loads(json_path.read_text(encoding="utf-8").splitlines()[0])["id"])


def run_fixture(name: str, workdir: Path) -> Path:
    """Convert one JSON fixture to compact, run the exe with --simulation-data, return the .simlog."""
    sys.path.insert(0, str(GENETIC_ML))
    from genetic_ml.compact_format import json_file_to_compact

    json_path = FIXTURE_DIR / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    dat = workdir / f"{name}.dat"
    json_file_to_compact(json_path, dat)

    base = workdir / f"{name}.simlog"
    env = os.environ.copy()
    env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
    # Skip the +64 y-offset so logged coordinates match the input JSON exactly (these fixtures have
    # no negative y). Structural-verify stays on by default (observer triggering).
    env["MCP1122_CPP_NO_Y_OFFSET"] = "1"
    subprocess.run([str(EXE), str(dat), "--simulation-data", str(base)],
                   env=env, check=True, stdout=subprocess.DEVNULL)

    cid = _load_id(json_path)
    out = workdir / f"{name}-{cid}.simlog"
    if not out.exists():
        raise FileNotFoundError(f"expected log not produced: {out}")
    return out


def _self_check() -> None:
    """Round-trip a hand-built buffer through the reader (no exe needed)."""
    # Two blocks, interleaved emission order; block A must decode its 2 events in subtick order.
    kA, kB = 111, 222
    events = [
        (kA, kA, 0, 5, 5, 5, 0, 0, 0, 10, 2, 5, SEF_EXTEND | SEF_SUCCESS, 1, 1, 0, 0),
        (kB, kB, 0, 5, 5, 5, 1, 1, 1, 0, 3, 0xFF, 0, 0, 0, 0, 0),
        (kA, kA, 0, 18, 18, 18, 2, 2, 2, 11, 2, 5, SEF_EXTEND | SEF_SUCCESS, 1, 1, 0, 0),
    ]
    # group by block: A gets events[0],events[2]; B gets events[1]
    order = [events[0], events[2], events[1]]
    body = b"".join(_EVENT.pack(*e) for e in order)
    index = [
        _INDEX.pack(kA, kA, 0, 2, 165, 0),
        _INDEX.pack(kB, kB, 2, 1, 218, 0),
    ]
    idx_off = len(body)
    body += b"".join(index)
    footer = _FOOTER.pack(b"SDL2", 2, 3, idx_off, 2, 72, 32, 0, 0)
    data = body + footer

    f = read_footer(data)
    entries = read_block_index(data, f)
    a = next(e for e in entries if e.originalKey == kA)
    evs = list(iter_block_events(data, a))
    assert len(evs) == 2, evs
    assert evs[0].pushGroupId == 10 and evs[1].pushGroupId == 11
    assert evs[0].activationSubtick < evs[1].activationSubtick
    assert unpack_pos(kA) == unpack_pos(kA)
    print("self-check PASS")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--self-check":
        _self_check()
        return 0
    if not EXE.exists():
        print(f"error: exe not built: {EXE}\nbuild it via 'cpp simulator/build-cpp.bat' first.")
        return 1

    fixtures = argv if argv else DEFAULT_FIXTURES
    total_empty = 0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name in fixtures:
            log = run_fixture(name, workdir)
            total_empty += dump_log(log)
            print()
    if total_empty:
        print(f"note: {total_empty} block(s) had no events (cargo/decorative blocks may be normal)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
