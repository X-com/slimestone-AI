"""Reinforcement-learning training on top of the C++ flying-machine simulator.

genetic_ml (../genetic algorithm/genetic_ml) has no build-system in its pyproject.toml, so it
isn't pip-installable - this sys.path insertion is the pragmatic way to reuse its
SimulatorPool/config/blocks/candidate_io/archive modules without duplicating them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_GENETIC_ML_ROOT = Path(__file__).resolve().parents[2] / "genetic algorithm"
if str(_GENETIC_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENETIC_ML_ROOT))
