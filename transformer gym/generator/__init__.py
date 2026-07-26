"""Synthetic training-data generator for the physics transformer (transformer_gym).

genetic_ml has no build-system in its pyproject.toml, so it isn't pip-installable - this
sys.path insertion is the pragmatic way to reuse its blocks/mutation/compact_format/candidate_io/
population modules without duplicating them, same shim pattern as reinforcement learning/rl_ml.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_GENETIC_ML_ROOT = REPO / "genetic algorithm"
if str(_GENETIC_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENETIC_ML_ROOT))

_UTIL_TOOLS = REPO / "util tools"
if str(_UTIL_TOOLS) not in sys.path:
    sys.path.insert(0, str(_UTIL_TOOLS))

_TRANSFORMER_GYM = Path(__file__).resolve().parents[1]
if str(_TRANSFORMER_GYM) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_GYM))

# generator's own modules cross-import each other as bare top-level names (e.g. gen_forward.py
# does `from geometry import ...`, not `from .geometry import ...`) so ad-hoc scripts/tests can
# import any one of them directly without going through the package - this makes that resolve
# regardless of which module triggered this __init__ first.
_GENERATOR_DIR = Path(__file__).resolve().parent
if str(_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_DIR))
