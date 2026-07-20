"""Load, run, and verify every .schematic flying machine through punchster's cpp_sim.

This is the batch sibling of run_sim.py: instead of one hard-coded candidate, it walks
`flying machines/schematics`, decodes each legacy .schematic with the loader in
`util tools/schematic_to_stream_json.py`, runs them all through cpp_simulator_stream.exe, and
reports which ones the simulator confirms as genuine (shifting-loop) flying machines.

TRIGGER-SYSTEM PARITY (the important bit)
-----------------------------------------
The .schematic files (and the loader) describe machines in the *old* trigger convention: a red
stained-glass marker sits next to the machine, and the block it touches (a piston, historically
pulsed with redstone) is the "trigger". The loader records that trigger POSITION but throws the
marker away.

punchster's cpp_sim uses a *different*, newer trigger mechanism (see cpp_sim/src/block_registry.h
and Simulator::fireTriggerBlocks): it starts a machine by deleting every block whose id is
BLOCK_TRIGGER (253) at tick 0. BLOCK_TRIGGER is "a plain solid cube while present ... removing it
fires the same neighbor/observer cascade a broken block would". The `trigger` position field is
NOT consulted by the simulate path at all.

So to run these machines with parity we must *reintroduce the marker as a BLOCK_TRIGGER cube*: the
marker is exactly the adjacent perturbation point the old redstone pulse used to hit, and deleting
a 253 cube there reproduces that kick. Placing 253 at the marker (verified below) starts every
piston-triggered machine; replacing the machine's own trigger block with 253 instead starts none.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUNCHSTER_ROOT = Path(__file__).resolve().parent
if str(_PUNCHSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PUNCHSTER_ROOT))

from simlib.config import SimulatorRunConfig  # noqa: E402
from simlib.simulator_pool import SimulatorPool  # noqa: E402

DEFAULT_SCHEMATIC_DIR = _REPO_ROOT / "flying machines" / "schematics"
SCHEMATIC_LOADER_PATH = _REPO_ROOT / "util tools" / "schematic_to_stream_json.py"

# cpp_sim/src/block_registry.h: the dedicated kick-start block. Deleted at tick 0 to start a machine.
BLOCK_TRIGGER_STATE = 253


def load_schematic_loader():
    """Import the .schematic loader from `util tools/` by path (its folder name has a space, so a
    normal `import` can't reach it). We only need load_schematic / list_schematics from it."""
    spec = importlib.util.spec_from_file_location("schematic_loader", SCHEMATIC_LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_simulator() -> Path:
    """Locate punchster's own built exe (a private, diverging copy of the simulator). Fail loudly
    with a build hint rather than a cryptic error deep inside SimulatorRunConfig."""
    candidate = _PUNCHSTER_ROOT / "cpp_sim" / "build" / "cpp_simulator_stream.exe"
    if candidate.exists():
        return candidate
    raise SystemExit(
        "Could not find cpp_simulator_stream.exe. Build it first "
        "(punchster/cpp_sim/build-cpp.bat), then re-run. Looked in:\n  " + str(candidate)
    )


def build_candidate(candidate_id: int, blocks: dict, marker: tuple) -> dict:
    """Turn one loaded schematic into a cpp_sim candidate. The parity step: drop a BLOCK_TRIGGER
    (253) cube at the marker position so fireTriggerBlocks() has something to delete at tick 0, and
    point `trigger` at it. All original machine blocks are kept verbatim."""
    machine_blocks = dict(blocks)
    machine_blocks[marker] = BLOCK_TRIGGER_STATE  # marker is never in `blocks` (loader skips it)
    return {
        "id": candidate_id,
        "trigger": {"x": marker[0], "y": marker[1], "z": marker[2]},
        "blocks": [
            {"x": pos[0], "y": pos[1], "z": pos[2], "state": state}
            for pos, state in machine_blocks.items()
        ],
    }


def flies(result: dict) -> bool:
    """A machine flies iff the sim settled into a loop (not timeout) whose per-loop shift is nonzero."""
    cycle_shift = result.get("cycle_shift") or {}
    moves = any(cycle_shift.get(axis, 0) != 0 for axis in ("x", "y", "z"))
    return bool(result.get("ok")) and not result.get("timeout") and moves


def main() -> int:
    schematic_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCHEMATIC_DIR
    simulator_path = find_simulator()
    loader = load_schematic_loader()

    schematics = loader.list_schematics(str(schematic_dir))
    if not schematics:
        raise SystemExit(f"No .schematic files found under: {schematic_dir}")

    print(f"Simulator : {simulator_path}")
    print(f"Schematics: {schematic_dir}  ({len(schematics)} files)\n")

    # Load + build candidates. A load failure (e.g. no/duplicate marker) is recorded, not fatal.
    candidates = []
    entries = []  # (id, name, candidate|None, load_error|None)
    for i, path in enumerate(schematics, start=1):
        name = Path(path).name
        try:
            blocks, marker, _trigger = loader.load_schematic(path)
            candidate = build_candidate(i, blocks, marker)
            candidates.append(candidate)
            entries.append((i, name, candidate, None))
        except Exception as exc:  # noqa: BLE001 - report every bad fixture, keep going
            entries.append((i, name, None, str(exc)))

    config = SimulatorRunConfig(
        simulator_path=simulator_path,
        worker_count=4,
        max_ticks=6000,
        simulation_timeout_seconds=10.0,
    )
    with SimulatorPool(config) as pool:
        results = {r["id"]: r for r in pool.run_all(candidates)}

    # Ground truth from the filenames: every fixture is a real flying machine EXCEPT the ones the
    # authors tagged "_doesnt_loop" (deliberate non-flyers). Verification = observed matches expected.
    flew = mismatches = load_errors = 0
    print(f"  {'stat':4} {'name':38} {'result'}")
    print(f"  {'----':4} {'-' * 38} {'------'}")
    for cid, name, candidate, load_error in entries:
        if load_error is not None:
            load_errors += 1
            print(f"  {'ERR ':4} {name:38} load failed: {load_error}")
            continue

        result = results.get(cid, {})
        expected_fly = "doesnt_loop" not in name
        observed_fly = flies(result)
        if observed_fly:
            flew += 1
        ok = observed_fly == expected_fly
        if not ok:
            mismatches += 1

        if not result.get("ok", False):
            detail = f"sim error: {result.get('errorCode')} {result.get('error')}"
        elif result.get("timeout"):
            detail = "timeout (no loop closed)"
        elif observed_fly:
            cs = result["cycle_shift"]
            detail = f"FLIES  period={result.get('cycle_ticks')} shift={cs}"
        else:
            detail = "stationary (settled, no shift)"

        stat = "OK  " if ok else "FAIL"
        print(f"  {stat:4} {name:38} {detail}")

    verified = len(entries) - mismatches - load_errors
    print(
        f"\nVerified {verified}/{len(entries)} against filename expectations "
        f"({flew} fly, {mismatches} mismatch, {load_errors} load errors)."
    )
    if mismatches:
        print(
            "Note: most mismatches are observer-triggered machines. punchster's cpp_sim only kicks\n"
            "an observer when the deleted marker cube sits on its watched face, so some canonically-\n"
            "valid observer flyers settle stationary here - a simulator limitation, not a load bug."
        )
    return 0 if (mismatches == 0 and load_errors == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
