"""The stdin/stdout protocol - md/VERIFICATION.md.

Batch-independence is the important one here. main.cpp:57 processStream reuses ONE Simulator
instance across the entire stream, so state leaking between candidates is a live risk - and it
would be completely invisible. Nothing would crash; the labels would simply be slightly wrong,
forever, in a way no aggregate number would reveal.
"""
from __future__ import annotations

import io
import struct

import pytest

from rlgym.blocks import PALETTE
from rlgym.game import GameState, Machine, Placement, decode_candidate
from rlgym.sim import SimulatorProcess, simulate_all

pytestmark = pytest.mark.slow

SLIME_SLOT = next(i for i, e in enumerate(PALETTE) if e and e.name == "slime")


def _verdict_core(result: dict) -> tuple:
    """Everything except timing, which legitimately varies run to run."""
    return (
        result.get("ok"),
        result.get("working"),
        result.get("validCycle"),
        result.get("period"),
        result.get("ticks"),
        result.get("start"),
        result.get("end"),
        tuple(sorted((result.get("finalShift") or {}).items())),
    )


def test_base_machine_has_the_expected_cycle(engine):
    """simple_observer_engine: 6 blocks, period 10, net shift one block in +x. Stated as
    literals so a change in simulator behaviour shows up here rather than silently shifting
    every label downstream."""
    with SimulatorProcess() as proc:
        result = proc.simulate(engine.to_candidate(cid=0))
    assert result["ok"] is True
    assert result["validCycle"] is True
    assert result["period"] == 10
    assert result["finalShift"] == {"x": 1, "y": 0, "z": 0}


def test_determinism(engine):
    """Same candidate twice through one process must give the same verdict."""
    with SimulatorProcess() as proc:
        first = proc.simulate(engine.to_candidate(cid=1))
        second = proc.simulate(engine.to_candidate(cid=1))
    assert _verdict_core(first) == _verdict_core(second)


def test_fresh_process_agrees_with_a_reused_one(engine):
    candidate = engine.to_candidate(cid=2)
    with SimulatorProcess() as first:
        a = first.simulate(candidate)
    with SimulatorProcess() as second:
        second.simulate(engine.to_candidate(cid=99))
        b = second.simulate(candidate)
    assert _verdict_core(a) == _verdict_core(b)


def test_batch_independence(engine):
    """A candidate simulated alone and the same candidate buried in a long batch must produce
    identical verdicts. This is the test for cross-candidate state leaking through the reused
    Simulator instance."""
    empties = [c for c in engine.cell_list if c not in engine.cells]
    target = engine.to_candidate(
        GameState(engine, k=1).step(Placement(empties[0], SLIME_SLOT)).current_cells(),
        cid=1234,
    )

    with SimulatorProcess() as proc:
        alone = proc.simulate(target)

    filler = [
        engine.to_candidate(
            GameState(engine, k=1).step(Placement(cell, SLIME_SLOT)).current_cells(), cid=i
        )
        for i, cell in enumerate(empties[1:])
    ]
    batch = filler[:10] + [target] + filler[10:]
    with SimulatorProcess() as proc:
        results = proc.simulate_batch(batch)

    in_batch = next(r for r in results if r["id"] == 1234)
    assert _verdict_core(alone) == _verdict_core(in_batch)


def test_verdict_order_matches_input_order(engine):
    """processStream is a strict read-one/print-one loop, so the Nth line answers the Nth
    candidate. simulate_batch checks the ids agree; if it ever stopped holding, every label
    after the desync point would be attached to the wrong machine."""
    candidates = [engine.to_candidate(cid=i) for i in range(20)]
    with SimulatorProcess() as proc:
        results = proc.simulate_batch(candidates)
    assert [r["id"] for r in results] == list(range(20))


def test_pool_preserves_order_and_agrees_with_single_process(engine):
    empties = [c for c in engine.cell_list if c not in engine.cells]
    candidates = [
        engine.to_candidate(
            GameState(engine, k=1).step(Placement(cell, SLIME_SLOT)).current_cells(), cid=i
        )
        for i, cell in enumerate(empties)
    ]
    pooled = simulate_all(candidates, workers=4)
    with SimulatorProcess() as proc:
        serial = proc.simulate_batch(candidates)
    assert [r["id"] for r in pooled] == [c["id"] for c in candidates]
    assert [_verdict_core(r) for r in pooled] == [_verdict_core(r) for r in serial]


def test_destroying_the_trigger_is_rejected_not_crashed(engine):
    """Overwriting the trigger cell gives a clean refusal with an explanatory message, and a
    reward of 0 - not a crash and not a silent success."""
    trigger_index = engine.cell_list.index(engine.trigger)
    candidate = engine.to_candidate(
        GameState(engine, k=1)
        .step(Placement(engine.cell_list[trigger_index], SLIME_SLOT))
        .current_cells(),
        cid=0,
    )
    with SimulatorProcess() as proc:
        result = proc.simulate(candidate)
    assert result["ok"] is False
    assert "trigger" in result.get("error", "")
    assert not result.get("validCycle")


def test_extended_piston_head_lands_where_facing_says(engine):
    """End-to-end confirmation of the compound action, using the simulator as the independent
    witness: an extended piston placed one cell west of the trigger, facing east, puts its head
    ON the trigger - and the C++ side notices, refusing the candidate for exactly that reason."""
    slot = next(
        i for i, e in enumerate(PALETTE) if e and e.name == "sticky_piston_extended_east"
    )
    tx, ty, tz = engine.trigger
    placement = Placement((tx - 1, ty, tz), slot)
    assert placement.written_cells()[1] == engine.trigger

    cells = GameState(engine, k=1).step(placement).current_cells()
    with SimulatorProcess() as proc:
        result = proc.simulate(engine.to_candidate(cells, cid=0))
    assert result["ok"] is False
    assert "trigger" in result.get("error", "")


def test_empty_batch_is_a_no_op():
    with SimulatorProcess() as proc:
        assert proc.simulate_batch([]) == []
    assert simulate_all([]) == []


def test_truncated_record_does_not_hang(engine):
    """A cut-off record must produce a parse_error verdict rather than blocking forever. The
    C++ side stops the stream on a truncated record because there is nothing sensible to
    recover into."""
    payload = struct.pack("<iiiiI", 0, 0, 0, 0, 5) + struct.pack("<iiiI", 1, 1, 1, 1)
    with SimulatorProcess() as proc:
        proc.write_raw(payload)
        proc.close_stdin()
        line = proc.read_raw_line(timeout=10.0)
    assert b"parse_error" in line, line


def test_what_we_send_is_what_decodes_back(engine):
    """Round-trip across the encoder that actually feeds the simulator, so a packing error
    cannot hide behind a matching decoder bug in the same direction."""
    candidate = engine.to_candidate(cid=5)
    from rlgym.game import encode_candidate

    decoded = decode_candidate(io.BytesIO(encode_candidate(candidate)))
    assert decoded == candidate
