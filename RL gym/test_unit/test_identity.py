"""The no-op identity test - md/VERIFICATION.md, the first of the three weighted tests.

One assertion covering candidate construction, the id | meta<<8 packing, trigger placement, the
stdin protocol, batch framing and canonical_hash - with no expected values written by hand. If
any of those is wrong, a placement that should change nothing changes something, and this fails.
"""
from __future__ import annotations

import pytest

from rlgym.blocks import PALETTE, PALETTE_SLOTS
from rlgym.game import Placement, apply_placement, canonical_hash, is_no_op
from rlgym.sim import SimulatorProcess

pytestmark = pytest.mark.slow


def _no_op_slot(cells, cell) -> int:
    """The one palette entry that leaves this cell as it already is."""
    return next(
        slot
        for slot in range(PALETTE_SLOTS)
        if PALETTE[slot] is not None and is_no_op(cells, Placement(cell, slot))
    )


def test_no_op_placement_leaves_the_cell_map_identical(engine):
    for cell in engine.cell_list:
        slot = _no_op_slot(engine.cells, cell)
        assert apply_placement(engine.cells, Placement(cell, slot)) == engine.cells


def test_no_op_placement_leaves_the_hash_identical(engine):
    base = canonical_hash(engine.to_candidate())
    for cell in engine.cell_list:
        slot = _no_op_slot(engine.cells, cell)
        cells = apply_placement(engine.cells, Placement(cell, slot))
        assert canonical_hash(engine.to_candidate(cells)) == base


def test_no_op_placement_gives_an_identical_verdict(engine):
    """The end-to-end version: through the encoder, through the pipe, through the C++ engine,
    and back. Every no-op must return exactly what the untouched machine returns."""
    base_candidate = engine.to_candidate(cid=0)
    candidates = [base_candidate]
    for index, cell in enumerate(engine.cell_list, start=1):
        slot = _no_op_slot(engine.cells, cell)
        cells = apply_placement(engine.cells, Placement(cell, slot))
        candidates.append(engine.to_candidate(cells, cid=index))

    with SimulatorProcess() as proc:
        results = proc.simulate_batch(candidates)

    def core(result: dict) -> tuple:
        return (
            result["ok"],
            result["working"],
            result["validCycle"],
            result["period"],
            result["ticks"],
            tuple(sorted(result["finalShift"].items())),
        )

    expected = core(results[0])
    assert expected[2] is True, "base machine must have a valid cycle for this test to mean anything"
    for result in results[1:]:
        assert core(result) == expected


def test_initial_state_we_send_is_what_the_simulator_receives(engine):
    """The round trip that verifies our encoder against the C++ decoder across the language
    boundary. A machine whose blocks are re-serialised from a decoded record must produce the
    same verdict as the original - and the same structural hash."""
    from rlgym.game import decode_candidate, encode_candidate
    import io

    original = engine.to_candidate(cid=0)
    decoded = decode_candidate(io.BytesIO(encode_candidate(original)))
    assert canonical_hash(decoded) == canonical_hash(original)

    with SimulatorProcess() as proc:
        a = proc.simulate(original)
        b = proc.simulate({**decoded, "id": 1})
    assert a["validCycle"] == b["validCycle"]
    assert a["period"] == b["period"]
    assert a["finalShift"] == b["finalShift"]
