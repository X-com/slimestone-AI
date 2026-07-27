"""Verify the C++ simulator's binary "simulation_data" event log end-to-end.

Runs a few flying-machine fixtures through cpp_simulator_stream.exe with --simulation-data, then
decodes the resulting per-candidate binary log and prints, per block, that block's complete
self-contained event history in order, plus the run's failure/push-group/initial-state/component
sections and its RunSummary. This is the human-readable check that the log captures everything; the
piston-event causality graph visualizer (graph_visualizer.py, same folder) is built on the decode
functions below (read_footer / read_block_index / iter_block_events / read_push_groups / etc.)
instead of reparsing.

Usage:
    py verify_simulation_data.py [fixture_name ...]      # defaults to a small representative set
    py verify_simulation_data.py --self-check            # decode round-trip assert, no exe needed

On-disk layout (must stay in sync with cpp simulator/src/sim_event_log.h):
    event section        : N x SimEvent (96 bytes), grouped by block, each block's run in sim order
    block index          : B x BlockIndexEntry (32 bytes), sorted by originalKey
    push-group section   : G x PushGroupRecord (48 bytes), one per piston firing ATTEMPT during the run
    push-group members   : flat uint64 array, indexed via memberOffset/memberCount
    initial-state        : one InitialBlockState (32 bytes) per original block - the model's input
    component section    : one ComponentRecord (32 bytes) per t=0 sticky group
    component members    : flat uint64 array, indexed via memberOffset/memberCount
    run summary          : one RunSummary (64 bytes)
    would-power section  : W x WouldPowerEdge (24 bytes) - static "would_power" relation, computed
                            once at load from the simulator's own power-resolution code, independent
                            of whether the activation ever actually happens during the run
    static push preview  : one PushGroupRecord (reused type) per piston, previewing whichever action
                            (extend/retract) its t=0 state implies is next - covers pistons that never
                            actually fire during the observed run, unlike the dynamic section above
    static push members   : flat uint64 array for the section above, indexed the same way, but a
                            SEPARATE array from the dynamic push-group members
    footer                : 188 bytes, magic "SDL6", offsets/counts for every section above
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

_EVENT = struct.Struct("<QQQQqqqIIIIhhhhhhBBBBBBBBI")  # 96 bytes
_INDEX = struct.Struct("<QQIIII")                       # 32 bytes (unchanged from SDL2)
_PUSHGROUP = struct.Struct("<QQiHBBBBBBIIIQ")           # 48 bytes
_INITIAL = struct.Struct("<QhhhHBBBBhBBII")             # 32 bytes
_COMPONENT = struct.Struct("<hBBIhhhhhhIQ")             # 32 bytes
_SUMMARY = struct.Struct("<BBbBii" + "h" * 12 + "I" * 7)  # 64 bytes
_WOULDPOWER = struct.Struct("<QQB7x")                   # 24 bytes (7x = 7 pad bytes)
_FOOTER = struct.Struct("<4sIQQQQIIIQIIQIIQIIQIIQIIQQIIQIIQQ")  # 188 bytes (SDL6)
assert _EVENT.size == 96, _EVENT.size
assert _INDEX.size == 32, _INDEX.size
assert _PUSHGROUP.size == 48, _PUSHGROUP.size
assert _INITIAL.size == 32, _INITIAL.size
assert _COMPONENT.size == 32, _COMPONENT.size
assert _SUMMARY.size == 64, _SUMMARY.size
assert _WOULDPOWER.size == 24, _WOULDPOWER.size
assert _FOOTER.size == 188, _FOOTER.size

KIND_NAMES = {
    0: "PistonQueued", 1: "PistonMoveExecuted", 2: "BlockPushed", 3: "ObserverFired",
    4: "ObserverActivated", 5: "RedstoneBlockAppeared", 6: "RedstoneBlockRemoved",
    7: "RedstoneActivatedPiston", 8: "RedstoneDeactivatedPiston",
    9: "PistonExtendBlocked", 10: "PistonRetractBlocked", 11: "BlockLeftBehind",
    12: "BlockDestroyed", 13: "ComponentSplit", 14: "ObserverSuppressed",
    15: "PistonNeighborNotified", 16: "ScheduledTickDropped", 17: "BlockPoweredChanged",
}
CAUSE_NAMES = {0: "scheduled", 1: "facing-changed", 2: "observer-moved"}
DIR_NAMES = {0: "DOWN", 1: "UP", 2: "NORTH", 3: "SOUTH", 4: "WEST", 5: "EAST", 0xFF: "-"}
BLOCK_NAMES = {
    0: "air", 29: "sticky_piston", 33: "piston", 34: "piston_head", 36: "piston_ext",
    27: "golden_rail", 28: "detector_rail", 66: "rail", 157: "activator_rail",
    96: "trapdoor", 167: "iron_trapdoor", 107: "fence_gate", 152: "redstone_block",
    165: "slime", 218: "observer", 123: "redstone_lamp", 124: "lit_redstone_lamp",
    49: "obsidian", 1: "stone", 20: "glass", 235: "glazed_terracotta",
}
FAILURE_REASON_NAMES = {
    0: "None", 1: "PushLimitExceeded", 2: "ImmovableBlockInPath", 3: "NoSpaceToExtend",
    4: "BlockCannotBePushed", 5: "AlreadyInTargetState", 6: "NotPowered", 7: "OutOfBounds",
}
TERMINATION_NAMES = {
    0: "CycleDetected", 1: "TickBudget", 2: "NothingHappened", 3: "StructureDestroyed",
    4: "OutOfBounds", 5: "InternalError",
}
MOVABILITY_NAMES = {0: "movable", 1: "immovable", 2: "pops"}
STICKINESS_NAMES = {0: "none", 1: "sticks-all", 2: "sticks-all-except-slime", 3: "never-sticks"}

SEF_EXTEND = 1 << 0
SEF_SUCCESS = 1 << 1
SEF_TARGET_PISTON = 1 << 4
SEF_OBSERVER_ON = 1 << 5  # ObserverFired/ObserverActivated: set = ON pulse, clear = OFF transition
SEF_POWERED_ON = 1 << 6  # BlockPoweredChanged: set = now on/open/lit


def _unpack21(v: int) -> int:
    v &= 0x1FFFFF
    if v & 0x100000:
        v -= 0x200000
    return v


def unpack_pos(key: int) -> tuple[int, int, int]:
    return (_unpack21(key), _unpack21(key >> 21), _unpack21(key >> 42))


class SimEvent:
    __slots__ = ("blockKey", "actorKey", "targetKey", "globalSeq", "activationTick",
                 "scheduledTick", "executedTick", "activationSubtick", "scheduledSubtick",
                 "executedSubtick", "pushGroupId", "fromX", "fromY", "fromZ", "toX", "toY", "toZ",
                 "kind", "direction", "flags", "attemptedAmount", "actualAmount", "failureReason",
                 "neighborSourceBlockId")

    def __init__(self, raw: tuple):
        (self.blockKey, self.actorKey, self.targetKey, self.globalSeq, self.activationTick,
         self.scheduledTick, self.executedTick, self.activationSubtick, self.scheduledSubtick,
         self.executedSubtick, self.pushGroupId, self.fromX, self.fromY, self.fromZ, self.toX,
         self.toY, self.toZ, self.kind, self.direction, self.flags, self.attemptedAmount,
         self.actualAmount, self.failureReason, self.neighborSourceBlockId, _r1, _r2) = raw


class BlockIndexEntry:
    __slots__ = ("originalKey", "currentKey", "firstEventIdx", "eventCount", "originalState")

    def __init__(self, raw: tuple):
        (self.originalKey, self.currentKey, self.firstEventIdx, self.eventCount,
         self.originalState, _r) = raw


class PushGroupRecord:
    __slots__ = ("globalSeq", "pistonKey", "tick", "subtick", "direction", "succeeded",
                 "failureReason", "memberCount", "memberOffset", "attemptedCount")

    def __init__(self, raw: tuple):
        (self.globalSeq, self.pistonKey, self.tick, self.subtick, self.direction, self.succeeded,
         self.failureReason, _p0, _p1, _p2, self.memberCount, self.memberOffset,
         self.attemptedCount, _r) = raw


class InitialBlockState:
    __slots__ = ("stableKey", "x", "y", "z", "blockTypeId", "facing", "stateFlags",
                 "movabilityClass", "stickinessClass", "componentId", "isTrigger", "rawState")

    def __init__(self, raw: tuple):
        (self.stableKey, self.x, self.y, self.z, self.blockTypeId, self.facing, self.stateFlags,
         self.movabilityClass, self.stickinessClass, self.componentId, self.isTrigger, _pad,
         self.rawState, _r) = raw


class ComponentRecord:
    __slots__ = ("componentId", "containsImmovable", "memberCount", "bboxMin", "bboxMax",
                 "memberOffset")

    def __init__(self, raw: tuple):
        (self.componentId, self.containsImmovable, _pad, self.memberCount,
         minx, miny, minz, maxx, maxy, maxz, self.memberOffset, _r) = raw
        self.bboxMin = (minx, miny, minz)
        self.bboxMax = (maxx, maxy, maxz)


class RunSummary:
    __slots__ = ("terminationReason", "validCycle", "travelAxis", "totalTicks", "period",
                 "netShift", "bboxMin", "bboxMax", "triggerPos", "totalEvents",
                 "distinctBlocksWithEvents", "maxObserverChainDepth", "maxPushGroupSize",
                 "pushLimitFailureCount", "blockCount")

    def __init__(self, raw: tuple):
        (self.terminationReason, self.validCycle, self.travelAxis, _pad, self.totalTicks,
         self.period, sx, sy, sz, minx, miny, minz, maxx, maxy, maxz, tx, ty, tz,
         self.totalEvents, self.distinctBlocksWithEvents, self.maxObserverChainDepth,
         self.maxPushGroupSize, self.pushLimitFailureCount, self.blockCount, _r) = raw
        self.netShift = (sx, sy, sz)
        self.bboxMin = (minx, miny, minz)
        self.bboxMax = (maxx, maxy, maxz)
        self.triggerPos = (tx, ty, tz)


class WouldPowerEdge:
    __slots__ = ("sourceKey", "pistonKey", "viaQC")

    def __init__(self, raw: tuple):
        self.sourceKey, self.pistonKey, self.viaQC = raw


def read_footer(data: bytes) -> dict:
    (magic, version, sim_build_hash, gen_seed, event_count, block_index_off, block_count, ev_sz,
     blk_sz, push_group_off, push_group_count, push_group_sz, push_member_off, push_member_count,
     _pad0, initial_off, initial_count, initial_sz, component_off, component_count, component_sz,
     component_member_off, component_member_count, summary_sz, summary_off,
     would_power_off, would_power_count, would_power_sz,
     static_push_group_off, static_push_group_count, static_push_member_count,
     static_push_member_off, _r) = \
        _FOOTER.unpack_from(data, len(data) - _FOOTER.size)
    if magic != b"SDL6":
        raise ValueError(f"bad magic {magic!r} (expected SDL6)")
    if ev_sz != _EVENT.size or blk_sz != _INDEX.size:
        raise ValueError(f"record size mismatch ev={ev_sz} blk={blk_sz}")
    return {
        "formatVersion": version, "eventCount": event_count, "blockIndexOffset": block_index_off,
        "blockCount": block_count, "pushGroupOffset": push_group_off,
        "pushGroupCount": push_group_count, "pushMemberOffset": push_member_off,
        "pushMemberCount": push_member_count, "initialOffset": initial_off,
        "initialCount": initial_count, "componentOffset": component_off,
        "componentCount": component_count, "componentMemberOffset": component_member_off,
        "componentMemberCount": component_member_count, "summaryOffset": summary_off,
        "wouldPowerOffset": would_power_off, "wouldPowerCount": would_power_count,
        "staticPushGroupOffset": static_push_group_off,
        "staticPushGroupCount": static_push_group_count,
        "staticPushMemberOffset": static_push_member_off,
        "staticPushMemberCount": static_push_member_count,
    }


def read_block_index(data: bytes, footer: dict) -> list[BlockIndexEntry]:
    off = footer["blockIndexOffset"]
    return [BlockIndexEntry(_INDEX.unpack_from(data, off + i * _INDEX.size))
            for i in range(footer["blockCount"])]


def iter_block_events(data: bytes, entry: BlockIndexEntry):
    for i in range(entry.eventCount):
        yield SimEvent(_EVENT.unpack_from(data, (entry.firstEventIdx + i) * _EVENT.size))


def read_push_groups(data: bytes, footer: dict) -> list[PushGroupRecord]:
    off = footer["pushGroupOffset"]
    return [PushGroupRecord(_PUSHGROUP.unpack_from(data, off + i * _PUSHGROUP.size))
            for i in range(footer["pushGroupCount"])]


def read_push_members(data: bytes, footer: dict) -> list[int]:
    off = footer["pushMemberOffset"]
    return list(struct.unpack_from(f"<{footer['pushMemberCount']}Q", data, off))


def read_initial_state(data: bytes, footer: dict) -> list[InitialBlockState]:
    off = footer["initialOffset"]
    return [InitialBlockState(_INITIAL.unpack_from(data, off + i * _INITIAL.size))
            for i in range(footer["initialCount"])]


def read_components(data: bytes, footer: dict) -> list[ComponentRecord]:
    off = footer["componentOffset"]
    return [ComponentRecord(_COMPONENT.unpack_from(data, off + i * _COMPONENT.size))
            for i in range(footer["componentCount"])]


def read_component_members(data: bytes, footer: dict) -> list[int]:
    off = footer["componentMemberOffset"]
    return list(struct.unpack_from(f"<{footer['componentMemberCount']}Q", data, off))


def read_summary(data: bytes, footer: dict) -> RunSummary:
    return RunSummary(_SUMMARY.unpack_from(data, footer["summaryOffset"]))


def read_would_power(data: bytes, footer: dict) -> list["WouldPowerEdge"]:
    off = footer["wouldPowerOffset"]
    return [WouldPowerEdge(_WOULDPOWER.unpack_from(data, off + i * _WOULDPOWER.size))
            for i in range(footer["wouldPowerCount"])]


def read_static_push_preview(data: bytes, footer: dict) -> list[PushGroupRecord]:
    off = footer["staticPushGroupOffset"]
    return [PushGroupRecord(_PUSHGROUP.unpack_from(data, off + i * _PUSHGROUP.size))
            for i in range(footer["staticPushGroupCount"])]


def read_static_push_members(data: bytes, footer: dict) -> list[int]:
    off = footer["staticPushMemberOffset"]
    return list(struct.unpack_from(f"<{footer['staticPushMemberCount']}Q", data, off))


def _fmt_event(ev: SimEvent) -> str:
    kind = KIND_NAMES.get(ev.kind, f"?{ev.kind}")
    parts = [f"t={ev.activationTick:<4} s={ev.activationSubtick:<4} {kind}"]
    if ev.kind in (0, 1, 2, 9, 10):  # piston queue/execute/blocked/push
        ext = "extend" if ev.flags & SEF_EXTEND else "retract"
        parts.append(ext)
        parts.append(f"dir={DIR_NAMES.get(ev.direction, ev.direction)}")
        if ev.kind == 1:
            parts.append("moved" if ev.flags & SEF_SUCCESS else "BLOCKED")
        if ev.kind == 2:
            parts.append(f"by piston{unpack_pos(ev.actorKey)}->{unpack_pos(ev.targetKey)}")
            parts.append(f"({ev.fromX},{ev.fromY},{ev.fromZ})->({ev.toX},{ev.toY},{ev.toZ})")
        if ev.kind in (9, 10) or ev.failureReason:
            parts.append(f"FAIL={FAILURE_REASON_NAMES.get(ev.failureReason, ev.failureReason)}")
        parts.append(f"amt {ev.attemptedAmount}->{ev.actualAmount}")
        parts.append(f"grp={ev.pushGroupId}")
        parts.append(f"sched(t={ev.scheduledTick},s={ev.scheduledSubtick})")
    elif ev.kind == 3:  # ObserverFired
        parts.append(f"cause={CAUSE_NAMES.get((ev.flags >> 2) & 3, '?')}")
        parts.append("ON" if ev.flags & SEF_OBSERVER_ON else "OFF")
    elif ev.kind == 4:  # ObserverActivated
        tgt = "piston" if ev.flags & SEF_TARGET_PISTON else "observer"
        parts.append(f"-> {tgt}{unpack_pos(ev.targetKey)}")
        parts.append("ON" if ev.flags & SEF_OBSERVER_ON else "OFF")
    elif ev.kind in (7, 8):  # redstone activate/deactivate
        parts.append(f"-> piston{unpack_pos(ev.targetKey)}")
    elif ev.kind == 15:  # PistonNeighborNotified (generic catch-all cause)
        src_name = BLOCK_NAMES.get(ev.neighborSourceBlockId, f"id{ev.neighborSourceBlockId}")
        parts.append(f"from {unpack_pos(ev.actorKey)} (was {src_name})")
    elif ev.kind == 16:  # ScheduledTickDropped
        src_name = BLOCK_NAMES.get(ev.neighborSourceBlockId, f"id{ev.neighborSourceBlockId}")
        parts.append(f"DROPPED reschedule of {src_name} at {unpack_pos(ev.blockKey)}")
    elif ev.kind == 17:  # BlockPoweredChanged
        src_name = BLOCK_NAMES.get(ev.neighborSourceBlockId, f"id{ev.neighborSourceBlockId}")
        parts.append(f"{src_name}")
        parts.append("ON" if ev.flags & SEF_POWERED_ON else "OFF")
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

    groups = read_push_groups(data, footer)
    members = read_push_members(data, footer)
    if groups:
        print(f"-- {len(groups)} push-group attempt(s) --")
        for g in groups:
            status = "OK" if g.succeeded else f"FAILED ({FAILURE_REASON_NAMES.get(g.failureReason, g.failureReason)})"
            mem = [unpack_pos(members[g.memberOffset + i]) for i in range(g.memberCount)]
            print(f"  piston{unpack_pos(g.pistonKey)} t={g.tick} attempted={g.attemptedCount} {status} members={mem}")

    initial = read_initial_state(data, footer)
    print(f"-- {len(initial)} initial-state record(s) --")
    components = read_components(data, footer)
    comp_members = read_component_members(data, footer)
    print(f"-- {len(components)} component(s) at t=0 --")
    for c in components:
        mem = [unpack_pos(comp_members[c.memberOffset + i]) for i in range(c.memberCount)]
        print(f"  component {c.componentId}: {c.memberCount} block(s) bbox={c.bboxMin}..{c.bboxMax} "
              f"containsImmovable={bool(c.containsImmovable)} members={mem}")

    summary = read_summary(data, footer)
    print(f"-- RunSummary: {TERMINATION_NAMES.get(summary.terminationReason, summary.terminationReason)} "
          f"validCycle={bool(summary.validCycle)} ticks={summary.totalTicks} period={summary.period} "
          f"netShift={summary.netShift} maxPushGroupSize={summary.maxPushGroupSize} "
          f"pushLimitFailures={summary.pushLimitFailureCount} --")

    would_power = read_would_power(data, footer)
    print(f"-- {len(would_power)} static would_power edge(s) --")
    for wp in would_power:
        via = " (via QC)" if wp.viaQC else ""
        print(f"  {unpack_pos(wp.sourceKey)} would power piston{unpack_pos(wp.pistonKey)}{via}")

    static_previews = read_static_push_preview(data, footer)
    static_members = read_static_push_members(data, footer)
    print(f"-- {len(static_previews)} static push-group preview(s) (t=0, whether-or-not-it-fires) --")
    for g in static_previews:
        status = "OK" if g.succeeded else f"FAILED ({FAILURE_REASON_NAMES.get(g.failureReason, g.failureReason)})"
        mem = [unpack_pos(static_members[g.memberOffset + i]) for i in range(g.memberCount)]
        print(f"  piston{unpack_pos(g.pistonKey)} dir={DIR_NAMES.get(g.direction, g.direction)} "
              f"attempted={g.attemptedCount} {status} members={mem}")

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

    def make_event(blockKey, actorKey, kind, subtick, tick, pushGroupId=0):
        return (blockKey, actorKey, 0, subtick, tick, tick, tick, subtick, subtick, subtick,
                pushGroupId, 0, 0, 0, 0, 0, 0, kind, 0xFF, 0, 0, 0, 0, 0, 0, 0)

    events = [
        make_event(kA, kA, 2, 0, 5, pushGroupId=10),
        make_event(kB, kB, 3, 1, 5),
        make_event(kA, kA, 2, 2, 18, pushGroupId=11),
    ]
    order = [events[0], events[2], events[1]]  # group by block: A's two, then B's one
    body = b"".join(_EVENT.pack(*e) for e in order)
    index = [
        _INDEX.pack(kA, kA, 0, 2, 165, 0),
        _INDEX.pack(kB, kB, 2, 1, 218, 0),
    ]
    idx_off = len(body)
    body += b"".join(index)

    pg_off = len(body)
    push_members = [kA, kB]
    body += b"".join(struct.pack("<Q", m) for m in push_members)
    pg = _PUSHGROUP.pack(99, kA, 18, 4, 0, 0, 1, 0, 0, 0, len(push_members), 0, 13, 0)
    pg_rec_off = len(body)
    body += pg

    initial_off = len(body)
    ib = _INITIAL.pack(kA, 1, 2, 3, 165, 0xFF, 0, 0, 1, 0, 0, 0, 0, 0)
    body += ib

    comp_member_off = len(body)
    body += b"".join(struct.pack("<Q", m) for m in [kA, kB])
    comp_off = len(body)
    body += _COMPONENT.pack(0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)

    summary_off = len(body)
    body += _SUMMARY.pack(0, 1, 0, 0, 20, 13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 13, 1, 2, 0)

    # Would-power edge (kA statically powers piston kB, via QC).
    would_power_off = len(body)
    body += _WOULDPOWER.pack(kA, kB, 1)

    # Static push-group preview (SDL4): piston kB, previewed regardless of whether it ever fires -
    # separate section AND separate member array from the dynamic push-group ones above.
    static_member_off = len(body)
    static_members = [kA, kB]
    body += b"".join(struct.pack("<Q", m) for m in static_members)
    static_group_off = len(body)
    body += _PUSHGROUP.pack(0, kB, 0, 0, 0, 1, 0, 0, 0, 0, len(static_members), 0, 2, 0)

    footer = _FOOTER.pack(
        b"SDL6", 6, 0, 0, 3, idx_off, 2, 96, 32,
        pg_rec_off, 1, 48,
        pg_off, len(push_members), 0,
        initial_off, 1, 32,
        comp_off, 1, 32,
        comp_member_off, 2,
        64, summary_off,
        would_power_off, 1, 24,
        static_group_off, 1, len(static_members),
        static_member_off, 0,
    )
    data = body + footer

    f = read_footer(data)
    entries = read_block_index(data, f)
    a = next(e for e in entries if e.originalKey == kA)
    evs = list(iter_block_events(data, a))
    assert len(evs) == 2, evs
    assert evs[0].pushGroupId == 10 and evs[1].pushGroupId == 11
    assert evs[0].activationSubtick < evs[1].activationSubtick
    assert unpack_pos(kA) == unpack_pos(kA)

    groups = read_push_groups(data, f)
    assert len(groups) == 1 and groups[0].succeeded == 0 and groups[0].attemptedCount == 13
    members = read_push_members(data, f)
    assert members == [kA, kB]

    initial = read_initial_state(data, f)
    assert len(initial) == 1 and initial[0].stableKey == kA and initial[0].blockTypeId == 165

    components = read_components(data, f)
    assert len(components) == 1 and components[0].memberCount == 2

    summary = read_summary(data, f)
    assert summary.terminationReason == 0 and summary.validCycle == 1 and summary.period == 13

    would_power = read_would_power(data, f)
    assert len(would_power) == 1 and would_power[0].sourceKey == kA and would_power[0].pistonKey == kB
    assert would_power[0].viaQC == 1

    static_previews = read_static_push_preview(data, f)
    assert len(static_previews) == 1 and static_previews[0].pistonKey == kB
    assert static_previews[0].succeeded == 1 and static_previews[0].attemptedCount == 2
    static_members_read = read_static_push_members(data, f)
    assert static_members_read == [kA, kB]

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
