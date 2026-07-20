"""Minimal, self-contained demo of the Python <-> C++ simulator interface.

Runs a single known-good flying machine through cpp_simulator_stream.exe and validates that
the simulator reports it as a real flying machine (validCycle == True).

The interface, in three layers (this script uses the top one, and peeks at the bottom one):
  1. compact_format.encode_candidate  - packs a candidate dict into the raw little-endian bytes
                                         that go to the exe's stdin.
  2. SimulatorProcess                  - owns one long-lived exe process, writes one record,
                                         reads back one JSON-line result.
  3. SimulatorPool                     - fans a batch across N worker processes; crash/hang-safe.

Run it from anywhere:
    python "punchster/run_flying_machine.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- Make the bundled simlib package importable --------------------------------------------
# simlib/ is a self-contained copy of the genetic_ml simulator bridge that lives next to this
# script (see simlib/__init__.py), so punchster/ no longer depends on the "genetic algorithm"
# package being importable. Add this script's own directory to sys.path so `import simlib` works
# no matter what the current working directory is.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUNCHSTER_ROOT = Path(__file__).resolve().parent
if str(_PUNCHSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PUNCHSTER_ROOT))

from simlib.compact_format import encode_candidate  # noqa: E402
from simlib.config import SimulatorRunConfig  # noqa: E402
from simlib.simulator_pool import SimulatorPool  # noqa: E402

# --- The machine to test -------------------------------------------------------------------
# A genuinely-valid flying machine ("simple_machine3", copied verbatim from
# flying machines/json-output/simple_machine3.json) so this script is fully self-contained. A
# candidate is just: an id, a trigger position (where the initial redstone pulse is applied),
# and a list of blocks {x, y, z, state}. `state` is a packed block id + metadata - see
# genetic_ml/blocks.py and the cpp block_registry for the encoding. This one settles into an
# exact translated copy of itself (shift +1 in x, period 20), which is what validCycle checks.
SIMPLE_MACHINE = {
    "id": 1,
    "trigger": {"x": 4, "y": 0, "z": 2},
    "blocks": [
        {"x": 0, "y": 0, "z": 0, "state": 165},
        {"x": 1, "y": 0, "z": 0, "state": 165},
        {"x": 2, "y": 0, "z": 0, "state": 1053},
        {"x": 3, "y": 0, "z": 0, "state": 165},
        {"x": 4, "y": 0, "z": 0, "state": 165},
        {"x": 5, "y": 0, "z": 0, "state": 1053},
        {"x": 0, "y": 0, "z": 1, "state": 165},
        {"x": 5, "y": 0, "z": 1, "state": 165},
        {"x": 0, "y": 0, "z": 2, "state": 1242},
        {"x": 1, "y": 0, "z": 2, "state": 1313},
        {"x": 3, "y": 0, "z": 2, "state": 165},
        {"x": 4, "y": 0, "z": 2, "state": 1313},
        {"x": 5, "y": 0, "z": 2, "state": 165},
        {"x": 3, "y": 1, "z": 0, "state": 1498},
        {"x": 5, "y": 1, "z": 1, "state": 986},
        {"x": 3, "y": 1, "z": 2, "state": 1242},
    ],
}


def find_simulator() -> Path:
    """Locate punchster's own built exe. This is a private copy of the simulator that lives under
    punchster/cpp_sim/ so its behavior (trigger-block start, hash-only cycle detection) can diverge
    from the shared 'cpp simulator/' build that the GA/RL code use. Fail loudly with a build hint
    rather than a cryptic FileNotFoundError deep inside SimulatorRunConfig."""
    candidates = [
        _PUNCHSTER_ROOT / "cpp_sim" / "build" / "cpp_simulator_stream.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        "Could not find cpp_simulator_stream.exe. Build it first "
        "(punchster/cpp_sim/build-cpp.bat), then re-run. Looked in:\n  " + searched
    )


def main() -> int:
    simulator_path = find_simulator()
    machine = SIMPLE_MACHINE

    # Peek at layer 1: this is the exact byte payload that gets written to the exe's stdin.
    wire_bytes = encode_candidate(machine)
    print(f"Simulator : {simulator_path}")
    print(f"Machine   : id={machine['id']}, {len(machine['blocks'])} blocks, "
          f"trigger={machine['trigger']}")
    print(f"Wire       : {len(wire_bytes)} bytes of compact binary "
          f"(16-byte header + {len(machine['blocks'])} x 16-byte blocks)")

    # Layer 3: run it. worker_count=1 is enough for a single machine; max_ticks bounds the
    # simulated tick loop; simulation_timeout_seconds is a wall-clock safety net for a machine
    # that hangs the simulator before tick counting even starts.
    config = SimulatorRunConfig(
        simulator_path=simulator_path,
        worker_count=1,
        max_ticks=6000,
        simulation_timeout_seconds=10.0,
    )

    with SimulatorPool(config) as pool:
        (result,) = pool.run_all([machine])

    # The simulator (punchster/cpp_sim) deletes the dedicated trigger block(s) at tick 0 to start the
    # machine, then runs until either the translation-invariant world-state hash repeats (the machine
    # has entered a loop) or the tick limit is hit (timeout). Every result dict has these keys:
    #
    #   id           - the candidate id echoed back.
    #   ok           - True if the simulation ran without error; False means see errorCode/error
    #                  (and the numeric fields below are meaningless).
    #   burnin_ticks - length of the one-off start-up transient: ticks from launch until the machine
    #                  first reaches the state it will loop on. On timeout this is the whole run.
    #   burnin_shift - net {x,y,z} block translation the machine underwent during that transient.
    #   cycle_ticks  - the loop period: number of ticks per repetition of the loop. 0 on timeout.
    #   cycle_shift  - net {x,y,z} block translation per loop. {0,0,0} => the machine stopped or is in
    #                  a stationary loop; nonzero => a shifting loop, i.e. it genuinely flies. Total
    #                  displacement after N loops = burnin_shift + N * cycle_shift.
    #   timeout      - True if no loop closed within the tick limit (cycle_* are then unusable).
    #   elapsedNs    - wall-clock time the C++ sim spent on this candidate, in nanoseconds.
    #   errorCode/error - present only when ok is False (crash, hang, parse/simulate error).
    print("\nResult from simulator:")
    for key in ("id", "ok", "burnin_ticks", "burnin_shift", "cycle_ticks", "cycle_shift",
                "timeout", "errorCode", "error"):
        if key in result:
            print(f"  {key:13}= {result[key]!r}")

    # A machine flies iff it settled into a loop (not timeout) whose per-loop shift is nonzero.
    cycle_shift = result.get("cycle_shift") or {}
    moves = any(cycle_shift.get(axis, 0) != 0 for axis in ("x", "y", "z"))
    is_valid = bool(result.get("ok")) and not result.get("timeout") and moves
    if is_valid:
        print(f"\nPASS: flying machine - loop period {result.get('cycle_ticks')} ticks, "
              f"shift/loop {cycle_shift}.")
    else:
        print("\nFAIL: not a flying machine (no shifting loop found).")
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())