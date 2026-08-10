"""Verifies the C++ simulator against the real Minecraft (mcp1122) engine: runs both on the same
55 flying-machine fixtures and diffs their sim-logs tick by tick. This is the actual behavioral
oracle check - cpp simulator/VERIFICATION.md's "the Java extract remains the behavior oracle" - now
runnable end to end.

WHAT "IDENTICAL" MEANS HERE - three deliberate, documented exclusions from the comparison, all
because the two writers use independent internal bookkeeping that was never meant to match in
absolute value (see mcp1122main/SimEventLog.java's class doc):

  - blockKey/actorKey/targetKey position-identity fields are NOT compared. C++ uses a persistent
    stableKey (re-keyed across pushes); Java uses the live packed position. Positions (fromX/Y/Z,
    toX/Y/Z) ARE compared - same coordinate system, same +64 Y offset on both sides (see below) -
    and are what actually pins an event to a place and moment.
  - globalSeq/activationSubtick/scheduledSubtick/executedSubtick/pushGroupId are NOT compared -
    independent monotonic counters per writer. Events are compared as a per-tick multiset, not a
    strict sub-tick order (a stronger check worth adding once this passes).
  - PistonMoveExecuted.failureReason is NOT compared when blocked. Java's BlockPistonStructureHelper
    .canMove() returns only a boolean (see BlockPistonBase.doMove's comment); a full port would need
    to expose WHY it failed, which the vanilla helper doesn't do today.

Y-OFFSET: kernel.main.Mcp1122FlyingMachineMain always shifts candidate Y by +64 (its own chunk
storage can't hold negative Y). The C++ side normally cancels its OWN +64 offset for verification
runs (MCP1122_CPP_NO_Y_OFFSET=1, see util tools/verify_simulation_data.py's run_fixture) - but this
tool deliberately does NOT set that env var, so the C++ engine's default +64 offset stays on and
both engines' logged coordinates already agree without any manual adjustment.

Usage:
    py "util tools/compare_java_cpp_simlog.py" [fixture_name ...]
    py "util tools/compare_java_cpp_simlog.py" --no-recompile   # skip the javac step
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_simulation_data as vsd  # noqa: E402
from simlog_ticks import iter_ticks as iter_ticks_cpp  # noqa: E402
from read_java_simlog import read_java_simlog  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MCP_ROOT = REPO / "ignore" / "mcp1122"
MCP_SRC = MCP_ROOT / "src"
MCP_BIN = MCP_ROOT / "bin" / "main"
MCP_RESOURCES = MCP_ROOT / "resources" / "1.12.2.jar"
DEPS_DIR = MCP_ROOT / "build" / "simlog-deps"
CLASSES_DIR = MCP_ROOT / "build" / "simlog-classes"
JAVAC = Path(r"D:\ProgramFiles\java21\bin\javac.exe")
JAVA8 = Path(r"D:\ProgramFiles\java8\bin\java.exe")
FIXTURE_DIR = REPO / "flying machines" / "json"

GUAVA = Path(r"C:\Users\Xcom\.gradle\caches\modules-2\files-2.1\com.google.guava\guava\18.0"
             r"\cce0823396aa693798f8882e64213b1772032b09\guava-18.0.jar")
JSR305 = Path(r"C:\Users\Xcom\.gradle\caches\modules-2\files-2.1\com.google.code.findbugs\jsr305"
              r"\3.0.2\25ea2e8b0c338a877313bd4672d3fe056ea78f0d\jsr305-3.0.2.jar")

TOUCHED_JAVA_FILES = [
    "mcp1122main/SimEventLog.java",
    "net/minecraft/block/BlockPistonBase.java",
    "net/minecraft/block/BlockPistonExtension.java",
    "net/minecraft/block/BlockObserver.java",
    "net/minecraft/tileentity/TileEntityPiston.java",
    "net/minecraft/world/World.java",
    "net/minecraft/world/WorldServer.java",
    "net/minecraft/block/BlockRailBase.java",
    "net/minecraft/block/BlockRailPowered.java",
    "net/minecraft/block/BlockFenceGate.java",
    "net/minecraft/block/BlockTrapDoor.java",
    "net/minecraft/block/BlockRedstoneLight.java",
    "net/minecraft/server/MinecraftServer.java",
    "kernel/main/Mcp1122FlyingMachineMain.java",
]

# Kind IDs that carry no from/to position at all (see loginfo.info's catalogue) - excluded from
# per-tick comparison entirely, not just from the key: without a position they cannot be matched
# to a specific place, and the redundant-fan-out kinds (5-8) are already covered indirectly by
# BlockStateChanged. Kept out rather than compared loosely to avoid a false sense of precision.
_SKIP_KINDS = frozenset()  # currently empty - every mirrored kind carries from/to; see docstring


def _comparable_key(ev) -> tuple:
    """The cross-engine comparison key for one event: kind + position + whichever fields carry
    real payload for that kind, deliberately excluding the three categories in the module
    docstring (identity keys, counters, failureReason)."""
    base = (ev.kind, ev.direction, ev.fromX, ev.fromY, ev.fromZ, ev.toX, ev.toY, ev.toZ)

    if ev.kind == 0:  # PistonQueued
        return base + (ev.flags,)
    if ev.kind == 1:  # PistonMoveExecuted - failureReason excluded, see module doc
        return base + (ev.flags, ev.attemptedAmount, ev.actualAmount)
    if ev.kind == 2:  # BlockPushed - reserved2/extraWord is the carried payload state word
        return base + (ev.flags, ev.neighborSourceBlockId, ev.extraWord)
    if ev.kind in (3, 4):  # ObserverFired / ObserverActivated
        return base + (ev.flags,)
    if ev.kind == 12:  # BlockDestroyed
        return base + (ev.neighborSourceBlockId, ev.movingEndCause)
    if ev.kind == 15:  # PistonNeighborNotified
        return base + (ev.neighborSourceBlockId,)
    if ev.kind in (18, 19):  # BlockSettled / MovingBlockDropped
        return base + (ev.neighborSourceBlockId, ev.movingEndCause)
    if ev.kind == 20:  # ScheduledTickCreated - reserved2 (order) is a counter, excluded
        return base + (ev.neighborSourceBlockId, ev.attemptedAmount)
    if ev.kind == 21:  # RailShapeChanged - reserved2/extraWord is the new state word
        return base + (ev.neighborSourceBlockId, ev.extraWord)
    if ev.kind == 22:  # BlockStateChanged - old/new state words are the actual payload
        return base + (ev.targetKey & 0xFFFFFFFF, ev.extraWord & 0xFFFFFFFF, ev.attemptedAmount)
    return base + (ev.flags, ev.neighborSourceBlockId, ev.movingEndCause)


def compile_java(force: bool) -> None:
    CLASSES_DIR.mkdir(parents=True, exist_ok=True)
    if not force and any(CLASSES_DIR.rglob("*.class")):
        return

    cp = [str(MCP_BIN), str(GUAVA), str(JSR305)] + [str(p) for p in DEPS_DIR.glob("*.jar")]
    sources = [str(MCP_SRC / f) for f in TOUCHED_JAVA_FILES]
    cmd = [str(JAVAC), "-nowarn", "-Xlint:none", "--release", "8",
           "-d", str(CLASSES_DIR), "-cp", ";".join(cp), "-sourcepath", str(MCP_SRC)] + sources
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("javac failed - see output above")
    print("javac: compiled clean")


def run_java_batch(names: list[str], workdir: Path) -> Path:
    """Runs kernel.main.Mcp1122FlyingMachineMain in ONE process over all requested fixtures
    (copied into workdir/input), sim-logging enabled - see SimEventLog.beginCandidate/endCandidate
    for the per-candidate reset that makes one .simlog per fixture safe within one JVM run."""
    input_dir = workdir / "input"
    input_dir.mkdir()
    for name in names:
        shutil.copy(FIXTURE_DIR / f"{name}.json", input_dir / f"{name}.json")

    cp = [str(CLASSES_DIR), str(MCP_BIN), str(MCP_RESOURCES), str(GUAVA), str(JSR305)] \
        + [str(p) for p in DEPS_DIR.glob("*.jar")]
    cmd = [str(JAVA8), "-Xmx4g", "-cp", ";".join(cp), "-Dmcp1122main.trace=false",
           "-Dmcp1122main.simlog=true", "kernel.main.Mcp1122FlyingMachineMain", str(input_dir)]
    result = subprocess.run(cmd, cwd=input_dir, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"Java batch run failed, exit {result.returncode}")

    simlog_dir = input_dir / "outlog" / "simlog"
    if not simlog_dir.is_dir():
        print(result.stdout[-4000:])
        raise RuntimeError(f"no {simlog_dir} produced")
    return simlog_dir


def run_cpp_one(name: str, workdir: Path) -> Path:
    """Same shape as verify_simulation_data.run_fixture, but deliberately does NOT set
    MCP1122_CPP_NO_Y_OFFSET - see module docstring's Y-OFFSET note."""
    sys.path.insert(0, str(vsd.GENETIC_ML))
    from genetic_ml.compact_format import json_file_to_compact

    json_path = FIXTURE_DIR / f"{name}.json"
    dat = workdir / f"{name}.dat"
    json_file_to_compact(json_path, dat)

    base = workdir / f"{name}.simlog"
    env = os.environ.copy()
    env["PATH"] = vsd.MSYS_BIN + os.pathsep + env.get("PATH", "")
    subprocess.run([str(vsd.EXE), str(dat), "--simulation-data", str(base)],
                    env=env, check=True, stdout=subprocess.DEVNULL)

    # The exe appends "-<candidateId>" to the base name (see verify_simulation_data.run_fixture).
    cid = vsd._load_id(json_path)
    out = workdir / f"{name}-{cid}.simlog"
    if not out.exists():
        raise FileNotFoundError(f"expected log not produced: {out}")
    return out


def compare_one(name: str, cpp_events: list, java_events: list) -> tuple[int, int]:
    """Groups both engines' events by activationTick and compares as a multiset per tick.
    Returns (mismatched_ticks, total_ticks_checked)."""
    from collections import Counter, defaultdict

    def by_tick(events):
        out: dict[int, Counter] = defaultdict(Counter)
        for ev in events:
            if ev.kind in _SKIP_KINDS:
                continue
            out[ev.activationTick][_comparable_key(ev)] += 1
        return out

    cpp_by_tick = by_tick(cpp_events)
    java_by_tick = by_tick(java_events)
    all_ticks = sorted(set(cpp_by_tick) | set(java_by_tick))

    bad = 0
    shown = 0
    for tick in all_ticks:
        c, j = cpp_by_tick.get(tick, Counter()), java_by_tick.get(tick, Counter())
        if c != j:
            bad += 1
            if shown < 3:
                shown += 1
                only_cpp = c - j
                only_java = j - c
                print(f"    MISMATCH {name} tick={tick}:")
                for k, n in list(only_cpp.items())[:5]:
                    print(f"        cpp-only  x{n}  {vsd.KIND_NAMES.get(k[0], k[0])} {k[1:]}")
                for k, n in list(only_java.items())[:5]:
                    print(f"        java-only x{n}  {vsd.KIND_NAMES.get(k[0], k[0])} {k[1:]}")

    return bad, len(all_ticks)


def main(argv: list[str]) -> int:
    no_recompile = "--no-recompile" in argv
    names = [a for a in argv if not a.startswith("--")] or sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))

    compile_java(force=not no_recompile)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        print(f"running Java batch over {len(names)} fixtures...")
        java_dir = run_java_batch(names, workdir)

        cpp_dir = workdir / "cpp"
        cpp_dir.mkdir()

        total_bad = total_ticks = 0
        failures = []
        for name in names:
            try:
                cpp_path = run_cpp_one(name, cpp_dir)
            except subprocess.CalledProcessError as e:
                print(f"  {name:30} CPP RUN FAILED: {e}")
                failures.append(name)
                continue

            java_path = java_dir / f"{name}.simlog"
            if not java_path.exists():
                print(f"  {name:30} NO JAVA OUTPUT")
                failures.append(name)
                continue

            footer = vsd.read_footer(cpp_path.read_bytes())
            cpp_events = list(iter_ticks_cpp(cpp_path.read_bytes(), footer))
            cpp_flat = [ev for _, evs in cpp_events for ev in evs]
            java_flat = read_java_simlog(java_path)

            bad, ticks = compare_one(name, cpp_flat, java_flat)
            total_bad += bad
            total_ticks += ticks
            status = "OK" if bad == 0 else f"{bad}/{ticks} ticks differ"
            print(f"  {name:30} cpp_events={len(cpp_flat):>7}  java_events={len(java_flat):>7}  {status}")
            if bad:
                failures.append(name)

    print()
    if not failures:
        print(f"IDENTICAL - all {len(names)} fixtures match ({total_ticks} ticks checked)")
        return 0
    print(f"MISMATCH - {len(failures)}/{len(names)} fixtures differ: {failures}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
