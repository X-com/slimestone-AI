"""Shared fixtures. See md/VERIFICATION.md for what these tests are for and why.

The governing rule, inherited from util tools/check_state_builder.py: re-derive facts
independently here rather than importing the thing under test, so a bug in one file cannot
hide behind another.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlgym.game import Machine  # noqa: E402
from rlgym.sim import DEFAULT_SIMULATOR  # noqa: E402

FIXTURE_DIR = _ROOT.parent / "flying machines" / "json"


def pytest_collection_modifyitems(config, items):
    """Slow tests shell out to the C++ simulator. Skip rather than fail when it is not built,
    so the fast suite still runs on a machine that has never compiled it."""
    if DEFAULT_SIMULATOR.exists():
        return
    skip = pytest.mark.skip(reason=f"simulator not built at {DEFAULT_SIMULATOR}")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def engine_candidate() -> dict:
    """simple_observer_engine: 6 blocks, 10-tick cycle, trigger on the north-facing observer.

    Small enough that its expected properties can be stated by hand, which is what makes the
    semantic assertions in these tests possible at all.
    """
    return json.loads(
        (FIXTURE_DIR / "simple_observer_engine.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def engine(engine_candidate) -> Machine:
    return Machine.from_candidate(engine_candidate, radius=1)


@pytest.fixture(scope="session")
def engine_cells(engine_candidate) -> set[tuple[int, int, int]]:
    """The six occupied cells, written out independently of blocks_to_map."""
    return {(b["x"], b["y"], b["z"]) for b in engine_candidate["blocks"]}
