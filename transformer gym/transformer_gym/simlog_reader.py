"""Re-exports of the .simlog binary decoder in `util tools/verify_simulation_data.py`.

Do not reimplement the format here - that file is the single source of truth for the
on-disk layout (kept in lockstep with cpp simulator/src/sim_event_log.h) and already has
its own --self-check.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "util tools"))

from verify_simulation_data import (  # noqa: E402
    BLOCK_NAMES,
    CAUSE_NAMES,
    DIR_NAMES,
    FAILURE_REASON_NAMES,
    KIND_NAMES,
    MOVABILITY_NAMES,
    STICKINESS_NAMES,
    TERMINATION_NAMES,
    BlockIndexEntry,
    ComponentRecord,
    InitialBlockState,
    PushGroupRecord,
    RunSummary,
    SimEvent,
    WouldPowerEdge,
    iter_block_events,
    read_block_index,
    read_component_members,
    read_components,
    read_footer,
    read_initial_state,
    read_push_groups,
    read_push_members,
    read_static_push_members,
    read_static_push_preview,
    read_summary,
    read_would_power,
    run_fixture,
    unpack_pos,
)

__all__ = [
    "BLOCK_NAMES", "CAUSE_NAMES", "DIR_NAMES", "FAILURE_REASON_NAMES", "KIND_NAMES",
    "MOVABILITY_NAMES", "STICKINESS_NAMES", "TERMINATION_NAMES",
    "BlockIndexEntry", "ComponentRecord", "InitialBlockState", "PushGroupRecord",
    "RunSummary", "SimEvent", "WouldPowerEdge",
    "iter_block_events", "read_block_index", "read_component_members", "read_components",
    "read_footer", "read_initial_state", "read_push_groups", "read_push_members",
    "read_static_push_members", "read_static_push_preview", "read_summary",
    "read_would_power", "run_fixture", "unpack_pos",
]
