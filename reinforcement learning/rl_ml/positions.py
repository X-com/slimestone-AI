"""Shared candidate-position logic: every currently-air, face-adjacent (6-connected) neighbor of
an existing block in a machine - the pool of legal attachment points any block-placement Task draws
from. Reuses genetic_ml/mutation.py's face-offset table so machine-adjacency logic isn't redefined
per task.
"""
from __future__ import annotations

from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.mutation import _FACING_OFFSETS

Candidate = dict[str, Any]


def candidate_positions(machine: Candidate) -> list[tuple[int, int, int]]:
    occupied = {(b["x"], b["y"], b["z"]) for b in machine["blocks"]}
    positions: set[tuple[int, int, int]] = set()
    for block in machine["blocks"]:
        for dx, dy, dz in _FACING_OFFSETS:
            pos = (block["x"] + dx, block["y"] + dy, block["z"] + dz)
            if pos not in occupied:
                positions.add(pos)
    return sorted(positions)


def neighbor_states(machine: Candidate, position: tuple[int, int, int]) -> list[tuple[int, int]]:
    """Returns (direction_index, state) for every block face-adjacent to position, where
    direction_index indexes _FACING_OFFSETS as the direction FROM position TO that neighbor."""
    occupied = {(b["x"], b["y"], b["z"]): b["state"] for b in machine["blocks"]}
    neighbors: list[tuple[int, int]] = []
    for i, (dx, dy, dz) in enumerate(_FACING_OFFSETS):
        pos = (position[0] + dx, position[1] + dy, position[2] + dz)
        if pos in occupied:
            neighbors.append((i, occupied[pos]))
    return neighbors
