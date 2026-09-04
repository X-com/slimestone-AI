"""Record + boards + placements -> the graph the model reads. md/ALPHAZERO.md Part 3.

Three item types:

    cell     one per (candidate cell, tick) - what is here, at this moment
    event    one per logged event, carrying its order - because a cell-tick averages 3-4 events
             and reaches 13, and collapsing them destroys the ordering that decides outcomes
    summary  exactly one per machine, connected to everything

Most relations come straight out of the record and are not recomputed here: WouldPowerEdge gives
direct and quasi-connectivity power, ComponentRecord gives the sticky components, PushGroupRecord
gives push-group membership. Only touching, observes and the time links are derived, and all
three are geometric.

**This file has no oracle.** Structural invariants catch indices out of range and wrong counts;
they cannot catch "I connected the observer to the wrong cell". test_graph_semantics.py carries
hand-written assertions for that, and it is the only thing standing between a plausible-looking
graph and a wrong one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from rlgym.blocks import (
    FACING_OFFSETS,
    block_id_of,
    can_provide_power,
    is_full_block,
    is_immovable,
    is_normal_cube,
    is_sticky,
    meta_of,
    push_reaction,
)
from rlgym.boards import board_at, boards_by_tick
from rlgym.game import Cell, Machine, Placement, candidate_cells
from rlgym.record import (
    BLOCK_PUSHED,
    BLOCK_SETTLED,
    MOVING_BLOCK_DROPPED,
    N_FAILURE_REASONS,
    N_KINDS,
    Record,
    SEF_SELF_ARM,
    SEF_SUCCESS,
    unpack_pos,
)

# --- item types ---
ITEM_CELL = 0
ITEM_EVENT = 1
ITEM_SUMMARY = 2
N_ITEM_TYPES = 3

# --- relation types. 64 slots reserved (ALPHAZERO.md), ~32 used. Ids are part of a trained
# model's meaning: appending is safe, renumbering is not.
REL_SELF = 0  # self-loop, so every item always has at least one key to attend over
REL_TOUCHING = 1  # symmetric, emitted both ways with this same id
REL_SAME_PUSH_GROUP = 2  # symmetric
REL_STICKY_COMPONENT = 3  # symmetric
REL_OBSERVES = 4
REL_OBSERVED_BY = 5
REL_POWERS_DIRECT = 6
REL_POWERED_BY_DIRECT = 7
REL_POWERS_QC = 8
REL_POWERED_BY_QC = 9
REL_TIME_FWD = 10  # 10..16, one per gap in TIME_GAPS
REL_TIME_BWD = 17  # 17..23
REL_EVENT_AT_CELL = 24
REL_CELL_HAS_EVENT = 25
REL_EVENT_NEXT = 26
REL_EVENT_PREV = 27
REL_EVENT_CAUSED_BY = 28
REL_EVENT_CAUSES = 29
REL_TO_SUMMARY = 30
REL_FROM_SUMMARY = 31
N_RELATION_SLOTS = 64

# Skip gaps. Reserved as relation ids whether or not a given machine is long enough to use them,
# so the model's shape never depends on cycle length (the Part 3 invariant). Only gaps shorter
# than the encoded span are actually emitted.
TIME_GAPS = (1, 2, 4, 8, 16, 64, 256)

# --- scalar feature columns. Adding one here is a retrain (point 1), which the version string
# below makes loud rather than silent.
SCALARS = (
    "is_air",
    "is_trigger",
    "cell_is_noted",
    "is_moving",
    "push_group_size",
    "in_flight",
    "tick_phase",
    "is_full_block",
    "is_normal_cube",
    "can_provide_power",
    "is_immovable",
    "is_sticky",
    "is_push_only",
    "event_order_in_tick",
    "event_success",
    "event_self_arm",
    "summary_period",
    "summary_shift_x",
    "summary_shift_y",
    "summary_shift_z",
    "summary_block_count",
    "view_is_complete",
)
N_SCALARS = len(SCALARS)
FEATURE_VERSION = "graph-v1"

NO_FACING = 6
NO_EVENT_KIND = N_KINDS
DEFAULT_TICK_CAP = 32
PUSH_LIMIT = 12


@dataclass
class Graph:
    """One machine, encoded. Arrays are flat over items; edges are (relation, src, dst)."""

    n_items: int
    item_type: np.ndarray
    block_type: np.ndarray
    facing: np.ndarray
    note_type: np.ndarray
    note_facing: np.ndarray
    event_kind: np.ndarray
    failure_reason: np.ndarray
    scalars: np.ndarray
    edges: np.ndarray  # [3, n_edges]
    summary_item: int
    policy_cells: list[Cell]
    cell_items: list[list[int]]  # per policy cell, its tick items, for the learned pooling
    n_ticks: int
    feature_version: str = FEATURE_VERSION
    # Every graph cell in MACHINE space -> its tick items, tick 0 first. Only apply_notes uses
    # it; it is a superset of cell_items because the graph covers the swept volume.
    cell_index: dict[Cell, list[int]] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[1])


class GraphTooLarge(ValueError):
    """The machine exceeds the tick cap.

    Refused loudly rather than truncated silently. The cap is the time half of OPEN.md's
    windowing plan - code that handles "you are seeing part of this machine's life" from the
    start is code the spatial half can slot into later.
    """


def build(
    machine: Machine,
    record: Record,
    placements: tuple[Placement, ...] = (),
    tick_cap: int = DEFAULT_TICK_CAP,
) -> Graph:
    span = record.summary.period if record.summary.period > 0 else record.summary.total_ticks
    span = max(1, min(span, record.summary.total_ticks))
    if span > tick_cap:
        raise GraphTooLarge(
            f"{record.summary.period}-tick cycle exceeds the {tick_cap}-tick cap. Raise "
            "tick_cap or exclude this machine; do not truncate silently."
        )
    ticks = list(range(span + 1))
    n_ticks = len(ticks)

    # The record lives in the simulator's own coordinates (a fixture at y=0 is logged at y=64),
    # so everything below works in record space and the candidate cells are translated into it.
    offset = _record_offset(machine, record)
    policy_cells = [_shift(cell, offset) for cell in machine.cell_list]
    boards = boards_by_tick(record)

    # Policy cells and graph cells are NOT the same set, and conflating them loses information
    # the model cannot do without.
    #
    # Policy cells are the R-shell of the tick-0 layout, because a placement edits the starting
    # configuration. But the machine FLIES. Encoding only those cells was measured to break
    # observers outright: on simple_observer_engine an observer reaches (2,65,2) at tick 4 and
    # watches (2,65,3), which nothing ever occupies and which sits at distance 2 from the tick-0
    # layout - so it is in neither the policy shell nor the swept volume, and the model could
    # not see what the observer was looking at.
    #
    # So graph cells are a shell around the SWEPT volume: every cell the machine ever occupies,
    # plus its face neighbours. That is what guarantees every observer's watched cell and every
    # push destination is present. The action space stays tied to tick 0 regardless.
    swept: dict[Cell, int] = {}
    for tick in ticks:
        swept.update(board_at(boards, tick))
    # ...and every cell a legal placement can WRITE, which is one face-step beyond the policy
    # shell because an extended piston writes its head there. Measured across the corpus: 23% of
    # extended-piston actions put their head outside the swept-plus-policy set, so without this
    # nearly a quarter of the 12 extended-piston palette entries were noted as a body with no
    # head - a machine that does not exist. Same class of bug as the observer watching a cell
    # outside the graph, and just as invisible to any structural check.
    writable = candidate_cells({cell: 0 for cell in policy_cells}, 1)
    cells = sorted(
        set(candidate_cells(swept, max(1, machine.radius)))
        | set(policy_cells)
        | set(writable)
    )

    noted = _noted_cells(placements, offset)

    builder = _Builder(record, cells, ticks, boards, noted)
    builder.add_cell_items()
    builder.add_event_items()
    builder.add_summary_item()
    builder.link_time()
    builder.link_touching()
    builder.link_from_record()
    builder.link_observes()
    builder.link_events()
    builder.link_summary()
    return builder.finish(machine.cell_list, n_ticks, policy_cells, offset)


NOTE_SCALAR = SCALARS.index("cell_is_noted")


def apply_notes(graph: Graph, placements: tuple[Placement, ...]) -> Graph:
    """The same graph with different notes, without rebuilding it.

    A placement changes features, never structure - measured, and asserted by
    test_placements_do_not_change_the_shape. So the 157 ms build happens once per machine and
    every one of its ~1,250 candidates is this patch instead, which is microseconds. That is
    what makes training on 207,935 examples affordable at all.

    Kept deliberately equivalent to passing `placements` to build(): the same two columns at
    tick 0, the same flag on every tick.
    """
    if not placements:
        return graph
    note_type = graph.note_type.copy()
    note_facing = graph.note_facing.copy()
    scalars = graph.scalars.copy()
    for placement in placements:
        entry = placement.entry
        facing = entry.facing if entry.facing is not None else NO_FACING
        for cell in placement.written_cells():
            items = graph.cell_index.get(cell)
            if not items:
                continue  # written outside the encoded volume; nothing to mark
            note_type[items[0]] = placement.slot + 1
            note_facing[items[0]] = facing + 1
            scalars[items, NOTE_SCALAR] = 1.0
    return replace(graph, note_type=note_type, note_facing=note_facing, scalars=scalars)


def _record_offset(machine: Machine, record: Record) -> tuple[int, int, int]:
    """The translation from candidate space to record space.

    Derived from the two block sets rather than assumed: both are the same machine, so the
    difference of their minimum corners is the offset. A machine whose block multiset does not
    match its own record would be a serious bug elsewhere, so it is checked.
    """
    mine = sorted(machine.cells.values())
    theirs = sorted(block.raw_state for block in record.initial)
    if mine != theirs:
        raise ValueError(
            "machine and record describe different block multisets - the record does not "
            "belong to this machine"
        )
    mx = min(c[0] for c in machine.cells)
    my = min(c[1] for c in machine.cells)
    mz = min(c[2] for c in machine.cells)
    rx = min(b.pos[0] for b in record.initial)
    ry = min(b.pos[1] for b in record.initial)
    rz = min(b.pos[2] for b in record.initial)
    return (rx - mx, ry - my, rz - mz)


def _shift(cell: Cell, offset: tuple[int, int, int]) -> Cell:
    return (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])


def _noted_cells(
    placements: tuple[Placement, ...], offset: tuple[int, int, int]
) -> dict[Cell, tuple[int, int]]:
    """cell -> (palette slot + 1, facing + 1). Both cells of an extended piston are noted."""
    out: dict[Cell, tuple[int, int]] = {}
    for placement in placements:
        entry = placement.entry
        facing = entry.facing if entry.facing is not None else NO_FACING
        for index, cell in enumerate(placement.written_cells()):
            out[_shift(cell, offset)] = (placement.slot + 1, facing + 1)
    return out


class _Builder:
    def __init__(self, record, cells, ticks, boards, noted):
        self.record = record
        self.cells = cells
        self.cell_set = set(cells)
        self.ticks = ticks
        self.boards = boards
        self.noted = noted

        self.item_type: list[int] = []
        self.block_type: list[int] = []
        self.facing: list[int] = []
        self.note_type: list[int] = []
        self.note_facing: list[int] = []
        self.event_kind: list[int] = []
        self.failure_reason: list[int] = []
        self.scalars: list[list[float]] = []
        self.edges: list[tuple[int, int, int]] = []

        self.cell_item: dict[tuple[Cell, int], int] = {}
        self.event_item: dict[int, int] = {}  # activation_subtick -> item
        self.summary_item = -1

        self.moving, self.group_size = self._movement_by_tick()
        self.in_flight = self._flight_by_tick()

    # --- items ---------------------------------------------------------------------------

    def _new_item(self, kind: int) -> int:
        index = len(self.item_type)
        self.item_type.append(kind)
        self.block_type.append(0)
        self.facing.append(NO_FACING)
        self.note_type.append(0)
        self.note_facing.append(0)
        self.event_kind.append(NO_EVENT_KIND)
        self.failure_reason.append(0)
        self.scalars.append([0.0] * N_SCALARS)
        self.edges.append((REL_SELF, index, index))
        return index

    def _set(self, item: int, name: str, value: float) -> None:
        self.scalars[item][SCALARS.index(name)] = value

    def add_cell_items(self) -> None:
        trigger = tuple(self.record.summary.trigger_pos)
        span = max(1, len(self.ticks) - 1)
        for tick in self.ticks:
            board = board_at(self.boards, tick)
            for cell in self.cells:
                item = self._new_item(ITEM_CELL)
                self.cell_item[(cell, tick)] = item
                state = board.get(cell, 0)
                self.block_type[item] = block_id_of(state)
                self.facing[item] = _facing_of(state)
                note = self.noted.get(cell)
                if note is not None:
                    # note_block_type only at tick 0 - the only moment the placement is a fact.
                    # cell_is_noted on every tick, meaning "what you read here describes a
                    # machine that no longer exists".
                    if tick == 0:
                        self.note_type[item] = note[0]
                        self.note_facing[item] = note[1]
                    self._set(item, "cell_is_noted", 1.0)
                self._set(item, "is_air", 1.0 if state == 0 else 0.0)
                self._set(item, "is_trigger", 1.0 if cell == trigger else 0.0)
                self._set(item, "tick_phase", tick / span)
                self._set(item, "is_moving", 1.0 if (cell, tick) in self.moving else 0.0)
                self._set(
                    item,
                    "push_group_size",
                    self.group_size.get((cell, tick), 0) / PUSH_LIMIT,
                )
                self._set(item, "in_flight", 1.0 if (cell, tick) in self.in_flight else 0.0)
                # Table lookups from block_registry.cpp (point 1): supplied rather than
                # learned, because the simulator already knows them.
                block = block_id_of(state)
                meta = meta_of(state)
                self._set(item, "is_full_block", float(is_full_block(block)))
                self._set(item, "is_normal_cube", float(is_normal_cube(block)))
                self._set(item, "can_provide_power", float(can_provide_power(block)))
                self._set(item, "is_immovable", float(is_immovable(block, meta)))
                self._set(item, "is_sticky", float(is_sticky(block)))
                self._set(item, "is_push_only", 1.0 if push_reaction(block) == 3 else 0.0)

    def add_event_items(self) -> None:
        last_tick = self.ticks[-1]
        for event in self.record.events_in_order():
            if event.executed_tick > last_tick:
                continue
            item = self._new_item(ITEM_EVENT)
            self.event_item[event.activation_subtick] = item
            self.event_kind[item] = event.kind
            self.failure_reason[item] = min(event.failure_reason, N_FAILURE_REASONS - 1)
            self._set(item, "event_success", 1.0 if event.flags & SEF_SUCCESS else 0.0)
            self._set(item, "event_self_arm", 1.0 if event.flags & SEF_SELF_ARM else 0.0)
            self._set(item, "event_order_in_tick", _log_norm(event.activation_subtick))
            self._set(item, "tick_phase", event.executed_tick / max(1, last_tick))

    def add_summary_item(self) -> None:
        item = self._new_item(ITEM_SUMMARY)
        self.summary_item = item
        summary = self.record.summary
        self._set(item, "summary_period", _log_norm(summary.period))
        self._set(item, "summary_shift_x", float(summary.net_shift[0]))
        self._set(item, "summary_shift_y", float(summary.net_shift[1]))
        self._set(item, "summary_shift_z", float(summary.net_shift[2]))
        self._set(item, "summary_block_count", _log_norm(summary.block_count))
        # Always 1.0 while the whole cycle is encoded. Reserved so that when a window is used
        # later, the model can tell "the cycle ended here" from "my view was cut off here".
        self._set(item, "view_is_complete", 1.0)

    # --- derived per-tick facts, read off the record rather than recomputed ----------------

    def _movement_by_tick(self):
        moving: set[tuple[Cell, int]] = set()
        sizes: dict[tuple[Cell, int], int] = {}
        members = self.record.push_members
        for group in self.record.push_groups:
            size = group.attempted_count or group.member_count
            for offset in range(group.member_count):
                index = group.member_offset + offset
                if index >= len(members):
                    continue
                cell = unpack_pos(members[index])
                key = (cell, group.tick)
                if group.succeeded:
                    moving.add(key)
                sizes[key] = max(sizes.get(key, 0), size)
        return moving, sizes

    def _flight_by_tick(self) -> set[tuple[Cell, int]]:
        """Cells holding an in-flight placeholder. Mid-flight the destination holds only the
        id-36 marker, so the fact that something is on its way there is not visible from the
        board alone."""
        out: set[tuple[Cell, int]] = set()
        open_flights: dict[Cell, int] = {}
        last_tick = self.ticks[-1]
        for event in self.record.events_in_order():
            if event.kind == BLOCK_PUSHED:
                open_flights[unpack_pos(event.target_key)] = event.executed_tick
            elif event.kind in (BLOCK_SETTLED, MOVING_BLOCK_DROPPED):
                cell = event.to_pos
                start = open_flights.pop(cell, None)
                if start is None:
                    continue
                for tick in range(start, min(event.executed_tick, last_tick) + 1):
                    out.add((cell, tick))
        return out

    # --- edges ----------------------------------------------------------------------------

    def _edge(self, relation: int, src: int, dst: int) -> None:
        self.edges.append((relation, src, dst))

    def _both(self, relation: int, a: int, b: int) -> None:
        self.edges.append((relation, a, b))
        self.edges.append((relation, b, a))

    def link_time(self):
        span = len(self.ticks)
        for gap_index, gap in enumerate(TIME_GAPS):
            if gap >= span:
                break
            for cell in self.cells:
                for tick in self.ticks:
                    later = tick + gap
                    if (cell, later) not in self.cell_item:
                        continue
                    a = self.cell_item[(cell, tick)]
                    b = self.cell_item[(cell, later)]
                    self._edge(REL_TIME_FWD + gap_index, a, b)
                    self._edge(REL_TIME_BWD + gap_index, b, a)

    def link_touching(self):
        for tick in self.ticks:
            for cell in self.cells:
                a = self.cell_item[(cell, tick)]
                for offset in FACING_OFFSETS:
                    other = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                    if other <= cell or other not in self.cell_set:
                        continue  # emit each pair once, then both directions
                    self._both(REL_TOUCHING, a, self.cell_item[(other, tick)])

    def link_from_record(self):
        """Push groups, sticky components and power - all read from the record, none recomputed."""
        members = self.record.push_members
        for group in self.record.push_groups:
            if group.tick not in self.ticks:
                continue
            items = []
            for offset in range(group.member_count):
                index = group.member_offset + offset
                if index >= len(members):
                    continue
                cell = unpack_pos(members[index])
                item = self.cell_item.get((cell, group.tick))
                if item is not None:
                    items.append(item)
            # Every member links to every other, so a 12-block group is ONE step across
            # regardless of its size or its shape.
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    self._both(REL_SAME_PUSH_GROUP, a, b)

        by_component: dict[int, list[Cell]] = defaultdict(list)
        for block in self.record.initial:
            if block.component_id >= 0:
                by_component[block.component_id].append(block.pos)
        for cells in by_component.values():
            for tick in self.ticks:
                items = [
                    self.cell_item[(cell, tick)]
                    for cell in cells
                    if (cell, tick) in self.cell_item
                ]
                for i, a in enumerate(items):
                    for b in items[i + 1 :]:
                        self._both(REL_STICKY_COMPONENT, a, b)

    def link_observes(self):
        """An observer watches the single block it faces - not a line of sight.

        Geometric rather than event-derived on purpose: it must hold even where nothing has
        happened yet, which an events-only view would miss entirely.
        """
        for tick in self.ticks:
            board = board_at(self.boards, tick)
            for cell in self.cells:
                state = board.get(cell, 0)
                if block_id_of(state) != 218:  # observer
                    continue
                facing = meta_of(state) & 0x7
                if facing >= len(FACING_OFFSETS):
                    continue
                offset = FACING_OFFSETS[facing]
                watched = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                if watched not in self.cell_set:
                    continue
                a = self.cell_item[(cell, tick)]
                b = self.cell_item[(watched, tick)]
                self._edge(REL_OBSERVES, a, b)
                self._edge(REL_OBSERVED_BY, b, a)

        for edge in _would_power(self.record):
            source, piston, via_qc = edge
            for tick in self.ticks:
                a = self.cell_item.get((source, tick))
                b = self.cell_item.get((piston, tick))
                if a is None or b is None:
                    continue
                if via_qc:
                    self._edge(REL_POWERS_QC, a, b)
                    self._edge(REL_POWERED_BY_QC, b, a)
                else:
                    self._edge(REL_POWERS_DIRECT, a, b)
                    self._edge(REL_POWERED_BY_DIRECT, b, a)

    def link_events(self):
        by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for event in self.record.events_in_order():
            item = self.event_item.get(event.activation_subtick)
            if item is None:
                continue
            cell_item = self.cell_item.get((event.from_pos, event.executed_tick))
            if cell_item is not None:
                self._edge(REL_EVENT_AT_CELL, item, cell_item)
                self._edge(REL_CELL_HAS_EVENT, cell_item, item)
            by_tick[event.executed_tick].append((event.activation_subtick, item))

            actor = unpack_pos(event.actor_key) if event.actor_key else None
            if actor is not None and actor != event.from_pos:
                cause = self.cell_item.get((actor, event.executed_tick))
                if cause is not None:
                    self._edge(REL_EVENT_CAUSED_BY, item, cause)
                    self._edge(REL_EVENT_CAUSES, cause, item)

        # Within-tick ordering. This is what makes update order visible, and it is the whole
        # reason events are separate items rather than folded into their cell.
        for events in by_tick.values():
            events.sort()
            for (_, a), (_, b) in zip(events, events[1:]):
                self._edge(REL_EVENT_NEXT, a, b)
                self._edge(REL_EVENT_PREV, b, a)

    def link_summary(self):
        for item in range(len(self.item_type)):
            if item == self.summary_item:
                continue
            self._edge(REL_TO_SUMMARY, item, self.summary_item)
            self._edge(REL_FROM_SUMMARY, self.summary_item, item)

    def finish(
        self,
        policy_cells: list[Cell],
        n_ticks: int,
        shifted: list[Cell],
        offset: tuple[int, int, int],
    ) -> Graph:
        # Pooling reads only the POLICY cells, in action order - the swept-volume extras are
        # there for the model to see, never to place into.
        cell_items = [
            [self.cell_item[(cell, tick)] for tick in self.ticks] for cell in shifted
        ]
        back = (-offset[0], -offset[1], -offset[2])
        cell_index = {
            _shift(cell, back): [self.cell_item[(cell, tick)] for tick in self.ticks]
            for cell in self.cells
        }
        edges = np.asarray(self.edges, dtype=np.int64).T
        return Graph(
            n_items=len(self.item_type),
            item_type=np.asarray(self.item_type, dtype=np.int64),
            block_type=np.asarray(self.block_type, dtype=np.int64),
            facing=np.asarray(self.facing, dtype=np.int64),
            note_type=np.asarray(self.note_type, dtype=np.int64),
            note_facing=np.asarray(self.note_facing, dtype=np.int64),
            event_kind=np.asarray(self.event_kind, dtype=np.int64),
            failure_reason=np.asarray(self.failure_reason, dtype=np.int64),
            scalars=np.asarray(self.scalars, dtype=np.float32),
            edges=edges,
            summary_item=self.summary_item,
            policy_cells=list(policy_cells),
            cell_items=cell_items,
            n_ticks=n_ticks,
            cell_index=cell_index,
        )


def _facing_of(state: int) -> int:
    if state == 0:
        return NO_FACING
    block = block_id_of(state)
    if block in (29, 33, 34, 218):
        return meta_of(state) & 0x7
    return NO_FACING


def _log_norm(value: int) -> float:
    return float(np.log1p(max(0, value)) / 10.0)


def _would_power(record: Record) -> list[tuple[Cell, Cell, bool]]:
    """The WouldPowerEdge section: which block statically powers which piston at t=0, computed
    by the simulator's own power resolution rather than reimplemented here.

    Quasi-connectivity is kept as its own relation because it powers a piston without ever
    firing an update - so without it the model sees pistons activating for no visible reason.
    """
    return [
        (unpack_pos(edge.source_key), unpack_pos(edge.piston_key), edge.via_qc)
        for edge in record.would_power
    ]
