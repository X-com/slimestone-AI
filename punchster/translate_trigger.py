"""Translate candidates from the RL/GA trigger system into punchster's trigger system.

The two simulators start a machine in fundamentally different ways (see run_schematics.py's header
for the full write-up):

  RL / GA  (`cpp simulator/`) - the candidate's `trigger` field points at one of the machine's own
      pistons. The simulator fires a short (2-tick) redstone pulse at that piston to kick the
      machine, then lets the trigger "burn out". The trigger MUST be a piston for this to work.

  punchster (`punchster/cpp_sim/`) - the simulator ignores the `trigger` field and instead deletes
      every block whose id is BLOCK_TRIGGER (253) at tick 0; the resulting break/neighbor cascade is
      what starts the machine. Nothing pulses a piston, so the trigger can be a plain block.

So translating means: keep the whole machine (including its trigger piston, which is structural),
and ADD a deletable BLOCK_TRIGGER cube on each empty face orthogonally adjacent to the trigger
position. Deleting those cubes at tick 0 reproduces "something changed right next to the trigger",
which is the punchster equivalent of the RL pulse.

Why every empty face and not just one: empirically (all 36 flying-machines/schematics fixtures),
adding a cube on every empty face of the trigger starts strictly more machines than a single-face
placement and introduces no false positives - the `_doesnt_loop` fixtures stay stationary, and the
extra cubes are deleted at tick 0 before the cycle seed is taken, so they never affect the cycle
comparison. A single-face placement ("first empty face") is offered for a more minimal translation.

IMPORTANT - this translates the *trigger format only*. It does not make the two engines agree on
the resulting motion: punchster's hash-shift cycle detection and the RL sim's validCycle (exact
translated-copy) check differ, and the block-delete kick and the piston pulse can drive a machine
into different dynamics. A translated candidate is guaranteed to *start* in punchster, not to earn
the same fly/no-fly verdict the RL sim gave it.

Usage:
    python punchster/translate_trigger.py INPUT.jsonl [OUTPUT.jsonl] [--first-face] [--verify]

INPUT may be JSON-lines candidates (one JSON object per line, the RL/GA on-disk format) or a compact
.data file. OUTPUT defaults to INPUT with a `.punchster.jsonl` suffix. --verify additionally runs
the translated candidates through punchster's simulator and prints which ones start into a loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PUNCHSTER_ROOT = Path(__file__).resolve().parent
if str(_PUNCHSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PUNCHSTER_ROOT))

from simlib.candidate_io import load_candidates_from_file, validate_candidate  # noqa: E402
from simlib.compact_format import read_compact_file  # noqa: E402

Candidate = dict[str, Any]

# cpp_sim/src/block_registry.h: the dedicated kick-start block, deleted at tick 0 to start a machine.
BLOCK_TRIGGER_STATE = 253

# The six orthogonal neighbours of a block (the faces a break cascade propagates across).
_FACE_OFFSETS = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0)]


def translate_candidate(candidate: Candidate, first_face_only: bool = False) -> Candidate:
    """Return a punchster-format copy of an RL/GA candidate.

    All original blocks are preserved; one BLOCK_TRIGGER (253) cube is added per empty face adjacent
    to `trigger` (or just the first empty face if first_face_only). The `trigger` field is repointed
    at the (first) added cube so it names the block punchster will actually delete - though punchster
    ignores the field, this keeps the candidate self-describing. Raises ValueError if the trigger has
    no empty adjacent face (nowhere to attach a deletable cube)."""
    validate_candidate(candidate)
    trigger = candidate["trigger"]
    tx, ty, tz = trigger["x"], trigger["y"], trigger["z"]

    blocks = [dict(block) for block in candidate["blocks"]]
    occupied = {(block["x"], block["y"], block["z"]) for block in blocks}

    added: list[tuple[int, int, int]] = []
    for dx, dy, dz in _FACE_OFFSETS:
        pos = (tx + dx, ty + dy, tz + dz)
        if pos in occupied:
            continue
        added.append(pos)
        blocks.append({"x": pos[0], "y": pos[1], "z": pos[2], "state": BLOCK_TRIGGER_STATE})
        occupied.add(pos)
        if first_face_only:
            break

    if not added:
        raise ValueError(
            f"candidate id={candidate.get('id')}: trigger ({tx},{ty},{tz}) has no empty adjacent "
            "face to attach a BLOCK_TRIGGER cube; cannot translate to the punchster trigger system"
        )

    trigger_pos = added[0]
    out = dict(candidate)
    out["trigger"] = {"x": trigger_pos[0], "y": trigger_pos[1], "z": trigger_pos[2]}
    out["blocks"] = blocks
    return out


def translate_candidates(candidates: list[Candidate], first_face_only: bool = False) -> list[Candidate]:
    return [translate_candidate(c, first_face_only=first_face_only) for c in candidates]


def _load_any(path: Path) -> list[Candidate]:
    if path.suffix == ".data":
        return read_compact_file(path)
    return load_candidates_from_file(path)


def _verify(candidates: list[Candidate]) -> None:
    """Run translated candidates through punchster's simulator and report which start into a loop."""
    from simlib.config import SimulatorRunConfig
    from simlib.simulator_pool import SimulatorPool

    exe = _PUNCHSTER_ROOT / "cpp_sim" / "build" / "cpp_simulator_stream.exe"
    if not exe.exists():
        print(f"\n(skipping --verify: simulator not built at {exe})", file=sys.stderr)
        return

    config = SimulatorRunConfig(
        simulator_path=exe, worker_count=4, max_ticks=6000, simulation_timeout_seconds=10.0
    )
    with SimulatorPool(config) as pool:
        results = {r["id"]: r for r in pool.run_all(candidates)}

    print("\nVerification (translated candidates in punchster):")
    started = 0
    for candidate in candidates:
        result = results.get(candidate["id"], {})
        cycle_shift = result.get("cycle_shift") or {}
        moves = any(cycle_shift.get(axis, 0) != 0 for axis in ("x", "y", "z"))
        flies = bool(result.get("ok")) and not result.get("timeout") and moves
        started += flies
        name = candidate.get("name", f"id={candidate['id']}")
        detail = f"FLIES shift={cycle_shift}" if flies else (
            "sim error" if not result.get("ok") else "stationary/timeout"
        )
        print(f"  {'FLY ' if flies else '    '} {name:34} {detail}")
    print(f"  -> {started}/{len(candidates)} start into a shifting loop")


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    first_face_only = "--first-face" in argv
    do_verify = "--verify" in argv

    if not args:
        print(__doc__)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1]) if len(args) > 1 else input_path.with_suffix(".punchster.jsonl")

    candidates = _load_any(input_path)
    translated = translate_candidates(candidates, first_face_only=first_face_only)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in translated:
            handle.write(json.dumps(candidate, separators=(",", ":")) + "\n")

    faces = "first empty face" if first_face_only else "every empty face"
    print(f"Translated {len(translated)} candidate(s) ({faces} of trigger) -> {output_path}")

    if do_verify:
        _verify(translated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))