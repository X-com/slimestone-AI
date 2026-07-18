from __future__ import annotations

import random
from pathlib import Path

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import BLOCK_SLIME
from genetic_ml.candidate_io import load_candidates_from_glob
from genetic_ml.compact_working_writer import CompactWorkingWriter
from genetic_ml.config import SimulatorRunConfig

from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task
from rl_ml.tasks.block_attachment import BlockAttachmentTask
from rl_ml.train_loop import train

# Edit these values in VS Code, then run this file.
PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_EXE = (
    PROJECT_ROOT.parent
    / "cpp simulator"
    / "build"
    / "cpp_simulator_stream.exe"
)

WORKING_DIR = PROJECT_ROOT / "data" / "working"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints" / "latest.json"
# Fixed-width binary hash logs (see genetic_ml/hash_log.py) - no candidate/result payload, just
# hash bytes back-to-back. The full candidate for a working discovery lives in
# COMPACT_DIR/flyers.data instead (via working_writer below), never duplicated here.
# WORKING_HASHES keeps the full 32-byte hash (exact - a false "yes, this works" would corrupt
# the reward signal). NOT_WORKING_HASHES truncates to 8 bytes: that side is the volume problem
# (every simulated candidate that fails, not just discoveries), and a rare false positive there
# only ever costs a missed opportunity to try a fresh candidate, never a wrong answer.
WORKING_HASHES = PROJECT_ROOT / "data" / "outputs" / "working_hashes.log"
NOT_WORKING_HASHES = PROJECT_ROOT / "data" / "outputs" / "not_working_hashes.log"
COMPACT_DIR = PROJECT_ROOT / "data" / "compact-working"
# How often CompactWorkingWriter/HashLog write their buffered records to disk in one batch, in
# seconds. A graceful exit (Ctrl+C or normal completion) always flushes everything regardless of
# this value - it only controls how often mid-run writes happen.
FLUSH_INTERVAL_SECONDS = 1.0

# Which training objective to run - swap this for a different rl_ml/tasks/*.py implementation, or
# pass more candidate_block_ids to train one shared model across several block types, to change
# what's being learned; nothing else in this file (or train_loop.py/policy.py) needs to change
# when you do.
TASK: Task = BlockAttachmentTask(candidate_block_ids=[BLOCK_SLIME])

WORKER_COUNT = 4
MAX_TICKS = 6000
SIMULATION_TIMEOUT_SECONDS = 5.0

BATCH_SIZE = 16
# None runs continuously against data/working/'s engines until you stop it (Ctrl+C); set an int
# (e.g. 200) to cap it for a quick local test run.
ITERATIONS: int | None = None
LEARNING_RATE = 0.1
CHECKPOINT_EVERY = 20
PROGRESS_EVERY = 10
RNG_SEED = 1667

# If True and CHECKPOINT_PATH already exists, resume training from its saved weights/iteration
# count instead of starting a fresh policy.
RESUME = False


def _feature_count(task: Task, base_machines: list[dict], rng: random.Random) -> int:
    contexts = task.sample_contexts(base_machines, rng, 1)
    if not contexts:
        raise RuntimeError("Task produced no contexts to determine feature count from")
    return len(task.features(contexts[0]))


def main() -> None:
    rng = random.Random(RNG_SEED)

    seeds = load_candidates_from_glob(WORKING_DIR / "*.json")
    print(f"Loaded {len(seeds)} base machine(s) from {WORKING_DIR}")

    if RESUME and CHECKPOINT_PATH.exists():
        policy = SharedLinearPolicy.load(CHECKPOINT_PATH)
        print(f"Resumed policy from {CHECKPOINT_PATH} (iteration {policy.iteration})")
    else:
        feature_count = _feature_count(TASK, seeds, rng)
        policy = SharedLinearPolicy(
            feature_count=feature_count,
            learning_rate=LEARNING_RATE,
            task_name=type(TASK).__name__,
        )

    simulator_config = SimulatorRunConfig(
        simulator_path=SIMULATOR_EXE,
        worker_count=WORKER_COUNT,
        max_ticks=MAX_TICKS,
        simulation_timeout_seconds=SIMULATION_TIMEOUT_SECONDS,
    )
    working_writer = CompactWorkingWriter(COMPACT_DIR, flush_interval_seconds=FLUSH_INTERVAL_SECONDS)

    train(
        TASK,
        policy,
        seeds,
        simulator_config,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
        progress_every=PROGRESS_EVERY,
        rng=rng,
        working_writer=working_writer,
        working_hashes_path=str(WORKING_HASHES),
        not_working_hashes_path=str(NOT_WORKING_HASHES),
        flush_interval_seconds=FLUSH_INTERVAL_SECONDS,
    )

    print(f"Done: {policy.iteration} iteration(s) run. Final weights: {policy.weights}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Working hashes: {WORKING_HASHES}")
    print(f"Not-working hashes: {NOT_WORKING_HASHES}")
    print(f"Compact working machines: {working_writer.path}")


if __name__ == "__main__":
    main()
