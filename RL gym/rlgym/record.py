"""Reading a .simlog record - the SDLA / formatVersion 10 layout.

Struct layouts are copied from cpp simulator/src/sim_event_log.h rather than imported (there is
nothing to import from - it is C++). Both sides must stay in sync; the version check below is
what makes a drift loud instead of silent.

Only what the graph builder needs is decoded: the footer, the summary, the initial state, the
events, and the push groups. verify_simulation_data.py in util tools decodes more, but most of
that file is CLI and reporting.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

MAGIC = b"SDLA"
FORMAT_VERSION = 10

# --- event kinds (sim_event_log.h SimEventKind) ---
PISTON_QUEUED = 0
PISTON_MOVE_EXECUTED = 1
BLOCK_PUSHED = 2
OBSERVER_FIRED = 3
OBSERVER_ACTIVATED = 4
REDSTONE_BLOCK_APPEARED = 5
REDSTONE_BLOCK_REMOVED = 6
REDSTONE_ACTIVATED_PISTON = 7
REDSTONE_DEACTIVATED_PISTON = 8
BLOCK_DESTROYED = 12
PISTON_NEIGHBOR_NOTIFIED = 15
SCHEDULED_TICK_DROPPED = 16
BLOCK_POWERED_CHANGED = 17
BLOCK_SETTLED = 18
MOVING_BLOCK_DROPPED = 19
SCHEDULED_TICK_CREATED = 20
RAIL_SHAPE_CHANGED = 21
# The replication backstop: emitted from inside setBlockState for EVERY world write that
# actually changes something. targetKey = old raw state, reserved2 = new raw state.
BLOCK_STATE_CHANGED = 22

N_KINDS = 23

# --- failure reasons (SimFailureReason) ---
FAILURE_NAMES = (
    "none",
    "push_limit_exceeded",
    "immovable_in_path",
    "no_space_to_extend",
    "block_cannot_be_pushed",
    "already_in_target_state",
    "not_powered",
    "out_of_bounds",
)
N_FAILURE_REASONS = len(FAILURE_NAMES)

TERMINATION_NAMES = (
    "cycle_detected",
    "tick_budget",
    "nothing_happened",
    "structure_destroyed",
    "out_of_bounds",
    "internal_error",
)

# --- flags ---
SEF_EXTEND = 1 << 0
SEF_SUCCESS = 1 << 1
SEF_SELF_ARM = 1 << 2
SEF_TARGET_PISTON = 1 << 4
SEF_OBSERVER_ON = 1 << 5
SEF_POWERED_ON = 1 << 6
SEF_QUEUE_DEDUPED = 1 << 7

SE_NO_DIRECTION = 0xFF

_EVENT = struct.Struct("<QQQQqqqIIIIhhhhhhBBBBBBBBI")
_BLOCK_INDEX = struct.Struct("<QQIIII")
_PUSH_GROUP = struct.Struct("<QQiHBBB3sII I Q".replace(" ", ""))
_INITIAL = struct.Struct("<QhhhHBBBBhBBII")
_SUMMARY = struct.Struct("<BBbBii3h3h3h3hIIIIIII")
_WOULD_POWER = struct.Struct("<QQB7s")
_FOOTER = struct.Struct("<4sIQQQQIIIQIIQIIQIIQIIQIIQQIIQIIQQ")

assert _EVENT.size == 96, _EVENT.size
assert _INITIAL.size == 32, _INITIAL.size


def unpack_signed21(value: int) -> int:
    out = value & 0x1FFFFF
    if out & 0x100000:
        out |= ~0x1FFFFF
    return out


def unpack_pos(key: int) -> tuple[int, int, int]:
    return (
        unpack_signed21(key),
        unpack_signed21(key >> 21),
        unpack_signed21(key >> 42),
    )


def pack_pos(x: int, y: int, z: int) -> int:
    return (x & 0x1FFFFF) | ((y & 0x1FFFFF) << 21) | ((z & 0x1FFFFF) << 42)


@dataclass(frozen=True)
class Event:
    block_key: int
    actor_key: int
    target_key: int
    global_seq: int
    activation_tick: int
    scheduled_tick: int
    executed_tick: int
    activation_subtick: int
    scheduled_subtick: int
    executed_subtick: int
    push_group_id: int
    from_pos: tuple[int, int, int]
    to_pos: tuple[int, int, int]
    kind: int
    direction: int
    flags: int
    attempted_amount: int
    actual_amount: int
    failure_reason: int
    reserved0: int
    reserved1: int
    reserved2: int


@dataclass(frozen=True)
class InitialBlock:
    stable_key: int
    pos: tuple[int, int, int]
    block_type_id: int
    facing: int
    state_flags: int
    movability: int
    stickiness: int
    component_id: int
    is_trigger: int
    raw_state: int


@dataclass(frozen=True)
class Summary:
    termination_reason: int
    valid_cycle: int
    travel_axis: int
    total_ticks: int
    period: int
    net_shift: tuple[int, int, int]
    bbox_min: tuple[int, int, int]
    bbox_max: tuple[int, int, int]
    trigger_pos: tuple[int, int, int]
    total_events: int
    distinct_blocks_with_events: int
    max_observer_chain_depth: int
    max_push_group_size: int
    push_limit_failure_count: int
    block_count: int


@dataclass(frozen=True)
class PushGroup:
    global_seq: int
    piston_key: int
    tick: int
    subtick: int
    direction: int
    succeeded: int
    failure_reason: int
    member_count: int
    member_offset: int
    attempted_count: int


class SimLogError(ValueError):
    pass


def read_footer(data: bytes) -> dict[str, Any]:
    if len(data) < _FOOTER.size:
        raise SimLogError(f"file is {len(data)} bytes, shorter than the {_FOOTER.size}-byte footer")
    fields = _FOOTER.unpack_from(data, len(data) - _FOOTER.size)
    names = (
        "magic", "formatVersion", "simulatorBuildHash", "generatorSeed", "eventCount",
        "blockIndexOffset", "blockCount", "eventRecSize", "blockRecSize",
        "pushGroupOffset", "pushGroupCount", "pushGroupRecSize",
        "pushMemberOffset", "pushMemberCount", "pad0",
        "initialOffset", "initialCount", "initialRecSize",
        "componentOffset", "componentCount", "componentRecSize",
        "componentMemberOffset", "componentMemberCount", "summaryRecSize",
        "summaryOffset", "wouldPowerOffset", "wouldPowerCount", "wouldPowerRecSize",
        "staticPushGroupOffset", "staticPushGroupCount", "staticPushMemberCount",
        "staticPushMemberOffset", "reserved",
    )
    footer = dict(zip(names, fields))
    if footer["magic"] != MAGIC:
        raise SimLogError(f"bad magic {footer['magic']!r}, expected {MAGIC!r}")
    if footer["formatVersion"] != FORMAT_VERSION:
        raise SimLogError(
            f"format version {footer['formatVersion']} but this reader speaks {FORMAT_VERSION}. "
            "The C++ side bumps the magic and version together on every layout change - update "
            "rlgym/record.py against sim_event_log.h rather than loosening this check."
        )
    if footer["eventRecSize"] != _EVENT.size:
        raise SimLogError(
            f"event record is {footer['eventRecSize']} bytes on disk, {_EVENT.size} here"
        )
    return footer


def read_events(data: bytes, footer: dict[str, Any]) -> list[Event]:
    """Events live at offset 0, one contiguous run per subject block (close() groups them)."""
    out: list[Event] = []
    for index in range(footer["eventCount"]):
        f = _EVENT.unpack_from(data, index * _EVENT.size)
        out.append(
            Event(
                block_key=f[0], actor_key=f[1], target_key=f[2], global_seq=f[3],
                activation_tick=f[4], scheduled_tick=f[5], executed_tick=f[6],
                activation_subtick=f[7], scheduled_subtick=f[8], executed_subtick=f[9],
                push_group_id=f[10],
                from_pos=(f[11], f[12], f[13]), to_pos=(f[14], f[15], f[16]),
                kind=f[17], direction=f[18], flags=f[19], attempted_amount=f[20],
                actual_amount=f[21], failure_reason=f[22], reserved0=f[23], reserved1=f[24],
                reserved2=f[25],
            )
        )
    return out


def read_initial_state(data: bytes, footer: dict[str, Any]) -> list[InitialBlock]:
    out: list[InitialBlock] = []
    base = footer["initialOffset"]
    for index in range(footer["initialCount"]):
        f = _INITIAL.unpack_from(data, base + index * _INITIAL.size)
        out.append(
            InitialBlock(
                stable_key=f[0], pos=(f[1], f[2], f[3]), block_type_id=f[4], facing=f[5],
                state_flags=f[6], movability=f[7], stickiness=f[8], component_id=f[9],
                is_trigger=f[10], raw_state=f[12],
            )
        )
    return out


def read_summary(data: bytes, footer: dict[str, Any]) -> Summary:
    f = _SUMMARY.unpack_from(data, footer["summaryOffset"])
    return Summary(
        termination_reason=f[0], valid_cycle=f[1], travel_axis=f[2],
        total_ticks=f[4], period=f[5],
        net_shift=(f[6], f[7], f[8]),
        bbox_min=(f[9], f[10], f[11]), bbox_max=(f[12], f[13], f[14]),
        trigger_pos=(f[15], f[16], f[17]),
        total_events=f[18], distinct_blocks_with_events=f[19],
        max_observer_chain_depth=f[20], max_push_group_size=f[21],
        push_limit_failure_count=f[22], block_count=f[23],
    )


def read_push_groups(data: bytes, footer: dict[str, Any]) -> tuple[list[PushGroup], list[int]]:
    groups: list[PushGroup] = []
    base = footer["pushGroupOffset"]
    size = footer["pushGroupRecSize"]
    for index in range(footer["pushGroupCount"]):
        f = struct.unpack_from("<QQiHBBB3xIII", data, base + index * size)
        groups.append(
            PushGroup(
                global_seq=f[0], piston_key=f[1], tick=f[2], subtick=f[3], direction=f[4],
                succeeded=f[5], failure_reason=f[6], member_count=f[7], member_offset=f[8],
                attempted_count=f[9],
            )
        )
    members = list(
        struct.unpack_from(
            f"<{footer['pushMemberCount']}Q", data, footer["pushMemberOffset"]
        )
    ) if footer["pushMemberCount"] else []
    return groups, members


@dataclass(frozen=True)
class WouldPower:
    """Static t=0 relation: source_key powers piston_key, computed once by the simulator's own
    power-resolution code rather than reimplemented. via_qc distinguishes direct adjacency from
    quasi-connectivity - power reaching the block ABOVE the piston.

    The distinction has to be kept because quasi-connectivity powers a piston without ever
    firing a block update, so a model that cannot see it sees pistons activating uncaused.
    """

    source_key: int
    piston_key: int
    via_qc: bool


def read_would_power(data: bytes, footer: dict[str, Any]) -> list[WouldPower]:
    out: list[WouldPower] = []
    count = footer.get("wouldPowerCount", 0)
    if not count:
        return out
    base = footer["wouldPowerOffset"]
    size = footer["wouldPowerRecSize"]
    for index in range(count):
        source, piston, via = struct.unpack_from("<QQB", data, base + index * size)
        out.append(WouldPower(source_key=source, piston_key=piston, via_qc=bool(via)))
    return out


@dataclass
class Record:
    footer: dict[str, Any]
    summary: Summary
    initial: list[InitialBlock]
    events: list[Event]
    push_groups: list[PushGroup]
    push_members: list[int]
    would_power: list[WouldPower]

    @classmethod
    def load(cls, path: str | Path) -> "Record":
        data = Path(path).read_bytes()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Record":
        footer = read_footer(data)
        groups, members = read_push_groups(data, footer)
        return cls(
            footer=footer,
            summary=read_summary(data, footer),
            initial=read_initial_state(data, footer),
            events=read_events(data, footer),
            push_groups=groups,
            push_members=members,
            would_power=read_would_power(data, footer),
        )

    def events_in_order(self) -> list[Event]:
        """Global emission order. Events are stored grouped by subject block, not chronologically,
        so anything replaying the run must sort - activationSubtick is the global counter."""
        return sorted(self.events, key=lambda e: e.activation_subtick)

    def initial_board(self) -> dict[tuple[int, int, int], int]:
        return {block.pos: block.raw_state for block in self.initial}
