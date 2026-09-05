"""Redundant-block detection - what replaced the cargo metric.

**Why the cargo metric was retired.** It asked "did this modification change what the machine
does", by comparing period and net shift. Two things break that question:

  1. almost any modification can be deemed cargo under it, so it barely discriminates;
  2. **"useful" cannot be read off a flight.** A block may be placed for looks, as a floor, a
     marker, a cover, or any other purpose in the game. It has nothing to do with whether the
     machine flies, and calling it worthless because the period did not move is simply wrong.
     Intent is not recoverable from a simulation, so the system stops guessing at it.

**The replacement: assume every block is valid as long as it serves a function, and detect only
the blocks that serve none.** That question is objective.

A block is FUNCTIONAL when both hold:

  1. it is part of a group of blocks that leads to a piston pushing other blocks, and
  2. removing it DISTURBS THE OPERATIONS OF THE PISTONS.

A block failing (2) is not load-bearing: it is redundant, and redundancy is the only thing here
that can be measured without guessing at intent.

**"Disturbing piston operations" means exactly:** a piston fires on a different tick, OR a piston
changes order relative to other pistons firing in the same tick. Period and net shift are never
consulted.

**The machine must also still work, and that is not implied by the signature.** Measured on
`simple_observer_engine`: removing the block at (0,0,1) leaves the piston signature byte-identical
while `validCycle` flips True -> False and the period collapses from 10 to 2. Testing the pistons
alone would delete load-bearing blocks and admit broken machines to the library. So redundancy is
"removable **and the same flying machine still works**", which is the whole of the stated
requirement rather than half of it. `Summary.valid_cycle` is already in every record, so this
costs no extra simulator call.

**The unit of work is the push group.** The added block is only the trigger. If a modification
changes a push group then every member of that group is re-checked, because adding a block can
make a previously-necessary neighbour redundant - and one modification can touch several groups.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rlgym.blocks import (
    BLOCK_PISTON,
    BLOCK_PISTON_HEAD,
    BLOCK_STICKY_PISTON,
    EXTENDED_BIT,
    FACING_OFFSETS,
    block_id_of,
    meta_of,
)
from rlgym.game import Cell, Machine
from rlgym.graph import record_offset
from rlgym.record import PISTON_MOVE_EXECUTED, SEF_EXTEND, SEF_SUCCESS, Record, unpack_pos
from rlgym.sim import SimConfig
from rlgym.simlog import records_for

# (tick, rank within tick, piston position, direction, is_extend, succeeded)
PistonOp = tuple[int, int, tuple[int, int, int], int, bool, bool]


def piston_signature(record: Record) -> tuple[PistonOp, ...]:
    """The ordered piston operations. Equality here **is** the definition of unchanged behaviour.

    `rank` is the index within its own tick, after sorting by `activation_subtick`. That is what
    makes the second half of the definition checkable at all: moving a piston to another tick
    changes field 0, and reordering two pistons inside one tick changes field 1. A set comparison
    would see neither, which is why this returns an ordered tuple.
    """
    by_tick: dict[int, list] = defaultdict(list)
    for event in record.events_in_order():
        if event.kind == PISTON_MOVE_EXECUTED:
            by_tick[event.executed_tick].append(event)

    out: list[PistonOp] = []
    for tick in sorted(by_tick):
        ordered = sorted(by_tick[tick], key=lambda e: e.activation_subtick)
        for rank, event in enumerate(ordered):
            out.append(
                (
                    tick,
                    rank,
                    event.from_pos,
                    event.direction,
                    bool(event.flags & SEF_EXTEND),
                    bool(event.flags & SEF_SUCCESS),
                )
            )
    return tuple(out)


def unchanged(record: Record | None, signature: tuple[PistonOp, ...]) -> bool:
    """Did removing a block leave the machine intact?

    Both halves are required. The signature alone is not enough - see the module docstring for
    the measured case where it stays identical while the machine stops flying.
    """
    if record is None:
        return False  # the simulator refused it outright: maximal disturbance
    if not record.summary.valid_cycle:
        return False
    return piston_signature(record) == signature


def _group_key(record: Record, group) -> tuple:
    members = frozenset(
        unpack_pos(record.push_members[group.member_offset + index])
        for index in range(group.member_count)
        if group.member_offset + index < len(record.push_members)
    )
    return (group.tick, group.subtick, unpack_pos(group.piston_key), group.direction, members)


def changed_push_groups(base: Record, candidate: Record) -> list:
    """Every push group of the candidate that does not appear identically in the base.

    One set difference covers gained members, lost members, a different piston, a different tick
    and a different direction - there is no separate case for each.
    """
    known = {_group_key(base, group) for group in base.push_groups}
    return [g for g in candidate.push_groups if _group_key(candidate, g) not in known]


def block_cells(cells: dict[Cell, int], cell: Cell) -> frozenset[Cell]:
    """Every cell of the block occupying `cell` - two for an extended piston, one otherwise.

    A piston is one block, not two cells. A lone head is destroyed as an orphan and replacing an
    extended piston's head kills the base, so body and head are removed together or not at all.
    """
    state = cells.get(cell)
    if state is None:
        return frozenset()
    block, meta = block_id_of(state), meta_of(state)
    if block in (BLOCK_PISTON, BLOCK_STICKY_PISTON) and meta & EXTENDED_BIT:
        dx, dy, dz = FACING_OFFSETS[meta & 0x7]
        head = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
        return frozenset({cell, head}) if head in cells else frozenset({cell})
    if block == BLOCK_PISTON_HEAD:
        dx, dy, dz = FACING_OFFSETS[meta & 0x7]
        body = (cell[0] - dx, cell[1] - dy, cell[2] - dz)
        return frozenset({body, cell}) if body in cells else frozenset({cell})
    return frozenset({cell})


def redundancy_candidates(
    base: Record,
    candidate: Record,
    cells: dict[Cell, int],
    added: set[Cell],
    offset: tuple[int, int, int],
) -> list[frozenset[Cell]]:
    """The blocks worth testing: what was added, plus every member of every modified push group.

    Push-group members arrive in **record space** (a fixture at y=0 is logged at y=64) while
    `cells` is in candidate space, so `offset` translates them back. Getting this wrong would
    silently produce an empty candidate set and report every discovery as fully load-bearing.
    """
    wanted: set[Cell] = set(added)
    for group in changed_push_groups(base, candidate):
        for index in range(group.member_count):
            position = group.member_offset + index
            if position >= len(candidate.push_members):
                continue
            x, y, z = unpack_pos(candidate.push_members[position])
            wanted.add((x - offset[0], y - offset[1], z - offset[2]))

    units: list[frozenset[Cell]] = []
    seen: set[frozenset[Cell]] = set()
    for cell in sorted(wanted):
        unit = block_cells(cells, cell)
        if unit and unit not in seen:
            seen.add(unit)
            units.append(unit)
    return units


def _record_of(
    machine: Machine, cells: dict[Cell, int], config: SimConfig | None = None
) -> Record | None:
    """One simulation with logging. None when the simulator refuses the machine outright."""
    if not cells:
        return None
    with records_for([machine.to_candidate(cells, cid=0)], config) as records:
        return records.get(0)


def find_redundant(
    machine: Machine,
    cells: dict[Cell, int],
    signature: tuple[PistonOp, ...],
    units: list[frozenset[Cell]],
    config: SimConfig | None = None,
) -> list[frozenset[Cell]]:
    """Which of `units` can be removed without disturbing the pistons.

    One batched simulator call for the whole set rather than one per block. A removal that
    destroys the trigger produces no record at all, so the lookup returns None and the block is
    correctly kept - the failure mode handles itself instead of needing a special case.
    """
    variants = []
    index: dict[int, frozenset[Cell]] = {}
    for cid, unit in enumerate(units):
        trimmed = {cell: state for cell, state in cells.items() if cell not in unit}
        if not trimmed:
            continue
        variants.append(machine.to_candidate(trimmed, cid=cid))
        index[cid] = unit

    redundant: list[frozenset[Cell]] = []
    if not variants:
        return redundant
    with records_for(variants, config) as records:
        for cid, unit in index.items():
            if unchanged(records.get(cid), signature):
                redundant.append(unit)
    return redundant


@dataclass
class Trim:
    cells: dict[Cell, int]
    removed: list[Cell]
    added_is_load_bearing: bool
    signature_length: int
    simulator_calls: int

    @property
    def redundant_removed(self) -> int:
        return len(self.removed)


def trim(
    machine: Machine,
    base_record: Record,
    cells: dict[Cell, int],
    added: set[Cell],
    config: SimConfig | None = None,
) -> Trim:
    """Strip every redundant block from a working discovery, verifying as it goes.

    Never returns cells it has not verified. **Individually redundant is not jointly redundant**
    - two blocks can each be droppable alone and necessary together, because each leaves the
    other holding the structure - so the all-at-once result is checked and falls back to removing
    one at a time when it fails.
    """
    calls = 0
    candidate_record = _record_of(machine, cells, config)
    calls += 1
    if candidate_record is None:
        return Trim(cells, [], True, 0, calls)

    if not candidate_record.summary.valid_cycle:
        return Trim(cells, [], True, 0, calls)  # not a working discovery; nothing to trim

    signature = piston_signature(candidate_record)
    if not signature:
        # No piston ever fired, so "unchanged piston operations" is vacuously true and every
        # block would test as redundant. That is not a flying machine; refuse to trim it.
        return Trim(cells, [], True, 0, calls)

    offset = record_offset(cells, candidate_record)
    units = redundancy_candidates(base_record, candidate_record, cells, added, offset)
    redundant = find_redundant(machine, cells, signature, units, config)
    calls += len(units)
    if not redundant:
        return Trim(cells, [], True, len(signature), calls)

    drop: set[Cell] = set().union(*redundant)
    together = {cell: state for cell, state in cells.items() if cell not in drop}
    calls += 1
    if unchanged(_record_of(machine, together, config), signature):
        return Trim(together, sorted(drop), not (added & drop), len(signature), calls)

    kept, removed = dict(cells), set()
    for unit in redundant:
        attempt = {cell: state for cell, state in kept.items() if cell not in unit}
        if not attempt:
            continue
        calls += 1
        if unchanged(_record_of(machine, attempt, config), signature):
            kept, removed = attempt, removed | unit
    return Trim(kept, sorted(removed), not (added & removed), len(signature), calls)


def added_cells(base: dict[Cell, int], candidate: dict[Cell, int]) -> set[Cell]:
    """Cells the modification wrote: anything present in the candidate that the base did not
    have, or had with a different state."""
    return {
        cell for cell, state in candidate.items() if base.get(cell) != state
    }
