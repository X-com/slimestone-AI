from __future__ import annotations

from pathlib import Path

from genetic_ml.candidate_io import load_candidates_from_glob
from genetic_ml.compact_format import read_compact_file
from genetic_ml.compact_working_writer import CompactWorkingWriter
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.ga_loop import GAConfig, run_ga
from genetic_ml.working_folder import WorkingFolderWriter

# Edit these values in VS Code, then run this file.
PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_EXE = (
    PROJECT_ROOT.parent
    / "java-mcp simulator"
    / "cpp simulator"
    / "build"
    / "cpp_simulator_stream.exe"
)

# Every file in here is one verified-working flying machine (id/trigger/blocks), used only as
# seed input - never written to by the GA. Drop JSON files here by hand to seed a run.
WORKING_DIR = PROJECT_ROOT / "data" / "working"

# "json" reproduces the exact previous behavior: every newly discovered machine gets saved as
# its own file in WORKING_DIR (discovered_0001.json, ...), which also makes it available as a
# seed on the next run. "compact" instead appends every discovery as one binary record onto a
# single file, COMPACT_DIR/flyers.data - no per-discovery files, no clutter - which also gets
# read back in as extra seeds below, alongside WORKING_DIR's JSON seeds, on every run.
WORKING_STORAGE_FORMAT = "compact"  # "json" or "compact"

# Default name new discoveries get in "json" mode (edit freely); an ascending postfix is
# appended, e.g. discovered_0001.json, discovered_0002.json, ...
DISCOVERED_NAME_PREFIX = "discovered"
COMPACT_DIR = PROJECT_ROOT / "data" / "compact-working"

# How many discoveries to buffer in memory before writing them to disk in one batch (applies to
# both the archive and, in "compact" mode, flyers.data - see genetic_ml/archive.py and
# genetic_ml/compact_working_writer.py). Higher = fewer disk writes on a long run, at the cost
# of losing more buffered-but-unflushed discoveries if the process is killed uncleanly (a
# graceful Ctrl+C or normal exit always flushes everything regardless of this value).
DISCOVERY_FLUSH_EVERY = 200

ARCHIVE_JSONL = PROJECT_ROOT / "data" / "outputs" / "ga_archive.jsonl"
# Every mutated candidate that crashes or hangs the simulator is written here as a
# bare candidate JSON (crash_0001.json, hung_0001.json, ...), for later use as
# regression/repro cases while hardening the C++ simulator.
CRASH_DIR = PROJECT_ROOT / "data" / "crash"
HANGS_DIR = PROJECT_ROOT / "data" / "hangs"

WORKER_COUNT = 4
MAX_TICKS = 6000
# Mutants can hang the simulator in an unbounded update cascade that MAX_TICKS does
# not bound (that only caps the simulated tick loop, not pre-tick block propagation).
# A wall-clock cap keeps one hung candidate from stalling the whole run.
SIMULATION_TIMEOUT_SECONDS = 5.0

POPULATION_CAPACITY = 16
OFFSPRING_PER_LINEAGE = 6
# None runs generations indefinitely until you stop it (Ctrl+C); set an int to cap it.
GENERATIONS: int | None = None
MUTATION_OPS_MIN = 1
MUTATION_OPS_MAX = 3
ACCEPT_BROKEN_PROBABILITY = 0.1
EXPLORATION_CAPACITY = 8
EXPLORATION_TTL = 3
RNG_SEED = 1667


def main() -> None:
    simulator_config = SimulatorRunConfig(
        simulator_path=SIMULATOR_EXE,
        worker_count=WORKER_COUNT,
        max_ticks=MAX_TICKS,
        simulation_timeout_seconds=SIMULATION_TIMEOUT_SECONDS,
    )
    ga_config = GAConfig(
        population_capacity=POPULATION_CAPACITY,
        offspring_per_lineage=OFFSPRING_PER_LINEAGE,
        generations=GENERATIONS,
        mutation_ops_min=MUTATION_OPS_MIN,
        mutation_ops_max=MUTATION_OPS_MAX,
        accept_broken_probability=ACCEPT_BROKEN_PROBABILITY,
        exploration_capacity=EXPLORATION_CAPACITY,
        exploration_ttl=EXPLORATION_TTL,
        seed=RNG_SEED,
    )

    all_seeds = load_candidates_from_glob(WORKING_DIR / "*.json")
    compact_seeds = read_compact_file(COMPACT_DIR / "flyers.data")
    all_seeds += compact_seeds
    seeds = sorted(all_seeds, key=lambda candidate: len(candidate["blocks"]))[:POPULATION_CAPACITY]
    if not seeds:
        raise RuntimeError(f"No seed candidates found in {WORKING_DIR} or {COMPACT_DIR}")
    print(
        f"Seeded population with {len(seeds)} working candidate(s) "
        f"({len(all_seeds) - len(compact_seeds)} from {WORKING_DIR}, {len(compact_seeds)} from {COMPACT_DIR})"
    )

    if WORKING_STORAGE_FORMAT == "compact":
        working_writer = CompactWorkingWriter(COMPACT_DIR, flush_every=DISCOVERY_FLUSH_EVERY)
        working_output_desc = working_writer.path
    else:
        working_writer = WorkingFolderWriter(WORKING_DIR, DISCOVERED_NAME_PREFIX)
        working_output_desc = WORKING_DIR

    result = run_ga(
        simulator_config,
        ga_config,
        seeds,
        str(ARCHIVE_JSONL),
        crash_dir=str(CRASH_DIR),
        hang_dir=str(HANGS_DIR),
        working_writer=working_writer,
        archive_flush_every=DISCOVERY_FLUSH_EVERY,
    )

    print(
        f"Done: {result.generations_run} generation(s), "
        f"{result.total_simulated} candidate(s) simulated, "
        f"{len(result.archive)} distinct working machine(s) in archive, "
        f"{len(result.population)} lineage(s) in final population, "
        f"{result.crashes_logged} crash(es) logged, {result.hangs_logged} hang(s) logged"
    )
    print(f"Archive: {ARCHIVE_JSONL}")
    print(f"Working output ({WORKING_STORAGE_FORMAT}): {working_output_desc}")
    print(f"Crash log: {CRASH_DIR}")
    print(f"Hang log: {HANGS_DIR}")


if __name__ == "__main__":
    main()
