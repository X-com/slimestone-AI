from __future__ import annotations

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)

from rl_ml.positions import candidate_positions, neighbor_states

_MACHINE = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [
        {"x": 0, "y": 0, "z": 0, "state": 1},
        {"x": 1, "y": 0, "z": 0, "state": 2},
    ],
}


def test_candidate_positions_are_air_neighbors_excluding_occupied_cells():
    positions = candidate_positions(_MACHINE)

    assert (0, 0, 0) not in positions
    assert (1, 0, 0) not in positions
    assert (2, 0, 0) in positions  # face-neighbor of the block at (1,0,0), currently air
    assert len(positions) == len(set(positions))  # deduped


def test_neighbor_states_finds_both_blocks_touching_the_shared_face_position():
    # (0,0,0) and (1,0,0) are face-adjacent to each other; probing from (1,0,0)'s far side...
    neighbors = neighbor_states(_MACHINE, (2, 0, 0))

    assert len(neighbors) == 1
    direction_index, state = neighbors[0]
    assert state == 2  # the block at (1,0,0)
    # direction FROM (2,0,0) TO (1,0,0) is -x (west) - offset index 4 per _FACING_OFFSETS.
    assert direction_index == 4


def test_neighbor_states_is_empty_when_position_touches_nothing():
    assert neighbor_states(_MACHINE, (10, 10, 10)) == []
