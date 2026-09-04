"""Candidate cells, action indexing and legality - md/ALPHAZERO.md Part 2."""
from __future__ import annotations

from rlgym.blocks import PALETTE, PALETTE_SLOTS, RESERVED_SLOTS
from rlgym.game import (
    GameState,
    Machine,
    Placement,
    action_index,
    apply_placement,
    candidate_cells,
    decode_action,
    is_no_op,
    legal_mask,
)

SLIME_SLOT = next(i for i, e in enumerate(PALETTE) if e and e.name == "slime")
AIR_SLOT = next(i for i, e in enumerate(PALETTE) if e and e.name == "air")
STICKY_EXT_EAST = next(
    i for i, e in enumerate(PALETTE) if e and e.name == "sticky_piston_extended_east"
)


def test_radius_zero_is_the_machine_itself(engine_cells):
    cells = {c: 1 for c in engine_cells}
    assert set(candidate_cells(cells, radius=0)) == engine_cells


def test_face_step_distance_not_chebyshev():
    """A single block at the origin has exactly 6 neighbours at R=1, not 26.

    A diagonal-only cell touches nothing, so a block there can never be pushed or dragged and
    is always left behind. Including it would add actions that are always wrong.
    """
    cells = {(0, 0, 0): 1}
    reached = set(candidate_cells(cells, radius=1))
    assert len(reached) == 7  # the block plus its 6 face neighbours
    assert (1, 1, 0) not in reached  # in-plane diagonal
    assert (1, 1, 1) not in reached  # corner


def test_radius_two_strictly_contains_radius_one(engine_candidate):
    r1 = set(Machine.from_candidate(engine_candidate, radius=1).cell_list)
    r2 = set(Machine.from_candidate(engine_candidate, radius=2).cell_list)
    assert r1 < r2


def test_candidate_cells_are_sorted_and_unique(engine):
    assert engine.cell_list == sorted(set(engine.cell_list))


def test_cell_list_includes_every_machine_block(engine, engine_cells):
    assert engine_cells <= set(engine.cell_list)


def test_action_count_is_cells_times_slots_plus_stop(engine):
    assert engine.action_count == len(engine.cell_list) * PALETTE_SLOTS + 1


def test_action_index_and_decode_round_trip(engine):
    for cell_index in (0, 3, len(engine.cell_list) - 1):
        for slot in (0, SLIME_SLOT, PALETTE_SLOTS - 1):
            index = action_index(cell_index, slot)
            placement = decode_action(index, engine.cell_list)
            assert placement.cell == engine.cell_list[cell_index]
            assert placement.slot == slot


def test_reserved_slots_are_never_legal(engine):
    mask = GameState(engine, k=1).legal_mask()
    for cell_index in range(len(engine.cell_list)):
        for slot in RESERVED_SLOTS:
            assert mask[action_index(cell_index, slot)] is False


def test_stop_is_always_masked(engine):
    """Reserved in the head so unmasking it later is not a shape change. It cannot simply be
    turned on: stopping at zero placements leaves the machine unmodified and therefore
    certainly valid, which is a guaranteed win and a degenerate optimum."""
    mask = GameState(engine, k=1).legal_mask()
    assert mask[-1] is False


def test_exactly_one_no_op_per_cell(engine):
    """Every cell has precisely one action that changes nothing: placing what is already there
    for an occupied cell, or placing air for an empty one. 26 cells x 39 live entries minus 26
    no-ops is the 988 legal actions the labeller reports."""
    cells = engine.cells
    for cell in engine.cell_list:
        no_ops = [
            slot
            for slot in range(PALETTE_SLOTS)
            if PALETTE[slot] is not None and is_no_op(cells, Placement(cell, slot))
        ]
        assert len(no_ops) == 1, f"{cell} had {len(no_ops)} no-ops"


def test_legal_action_total_matches_the_arithmetic(engine):
    mask = GameState(engine, k=1).legal_mask()
    live = len(engine.cell_list) * 39
    assert sum(mask) == live - len(engine.cell_list)


def test_air_on_an_empty_cell_is_a_no_op(engine):
    empty = next(c for c in engine.cell_list if c not in engine.cells)
    assert is_no_op(engine.cells, Placement(empty, AIR_SLOT))


def test_air_on_an_occupied_cell_removes_it(engine, engine_cells):
    occupied = sorted(engine_cells)[0]
    after = apply_placement(engine.cells, Placement(occupied, AIR_SLOT))
    assert occupied not in after
    assert len(after) == len(engine.cells) - 1


def test_replacement_is_the_same_operation_as_addition(engine, engine_cells):
    """DESIGN.md: add, remove and replace are all one operation. A placement is legal on any
    candidate cell, occupied or not."""
    occupied = sorted(engine_cells)[0]
    after = apply_placement(engine.cells, Placement(occupied, SLIME_SLOT))
    assert after[occupied] == PALETTE[SLIME_SLOT].state
    assert len(after) == len(engine.cells)


def test_extended_piston_writes_two_cells(engine):
    placement = Placement((5, 5, 5), STICKY_EXT_EAST)
    written = placement.written_cells()
    assert written == ((5, 5, 5), (6, 5, 5))  # east is +x
    after = apply_placement({}, placement)
    assert set(after) == {(5, 5, 5), (6, 5, 5)}


def test_written_cell_masking_covers_both_cells_of_an_extended_piston(engine):
    """Forbidding a second write to a cell is what makes placements a genuine set, so order
    never matters. For an extended piston that has to cover the head cell too, or two orders
    of the same pair would produce different machines."""
    base = engine.cell_list[0]
    head = (base[0] + 1, base[1], base[2])
    mask = legal_mask(engine.cells, engine.cell_list, frozenset({head}))
    if head in engine.cell_list:
        head_index = engine.cell_list.index(head)
        assert not any(
            mask[action_index(head_index, s)] for s in range(PALETTE_SLOTS)
        )
    base_index = engine.cell_list.index(base)
    assert mask[action_index(base_index, STICKY_EXT_EAST)] is False


def test_step_is_deterministic_and_order_independent(engine):
    """Two placements on different cells give the same machine in either order - which is what
    prevents point 23's factorial duplicate explosion structurally."""
    empties = [c for c in engine.cell_list if c not in engine.cells][:2]
    a = Placement(empties[0], SLIME_SLOT)
    b = Placement(empties[1], SLIME_SLOT)
    forward = GameState(engine, k=2).step(a).step(b).current_cells()
    reverse = GameState(engine, k=2).step(b).step(a).current_cells()
    assert forward == reverse


def test_terminal_state_has_no_legal_actions(engine):
    empty = next(c for c in engine.cell_list if c not in engine.cells)
    state = GameState(engine, k=1).step(Placement(empty, SLIME_SLOT))
    assert state.is_terminal
    assert sum(state.legal_mask()) == 0


def test_cell_count_is_a_data_dimension_not_a_weight_one(engine_candidate):
    """The invariant from Part 3 stated at the game layer: changing R changes how many actions
    exist, and nothing else about the action encoding."""
    for radius in (1, 2, 3):
        machine = Machine.from_candidate(engine_candidate, radius=radius)
        assert machine.action_count == len(machine.cell_list) * PALETTE_SLOTS + 1
