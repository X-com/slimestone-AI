"""Wave Function Collapse generator - a local-constraint paradigm, distinct from the six
constructive generators (gen_perturb/forward/pushgroup/puzzle/splice/random.py). Standalone for
now (see run_wfc.py) - not wired into corpus.py's GENERATOR_WEIGHTS mix per this session's
decision. See generator-design-handoff.md and the plan for the full rationale.

sys.path bootstrap mirrors generator/__init__.py: needed so this package resolves regardless of
whether it's reached via `generator.wave_function...` or a script run directly from within it
(run_wfc.py does its own extra bootstrap for that direct-script case).
"""
from __future__ import annotations

import sys
from pathlib import Path

_TRANSFORMER_GYM = Path(__file__).resolve().parents[2]
if str(_TRANSFORMER_GYM) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_GYM))

import generator  # noqa: F401,E402  (runs generator's own sys.path shim)
