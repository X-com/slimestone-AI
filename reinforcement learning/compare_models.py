from __future__ import annotations

import random
from pathlib import Path

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.blocks import BLOCK_SLIME
from genetic_ml.candidate_io import load_candidates_from_glob
from genetic_ml.config import SimulatorRunConfig

from rl_ml.evaluate import evaluate
from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task
from rl_ml.tasks.block_attachment import BlockAttachmentTask

# Edit these values in VS Code, then run this file.
PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_EXE = (
    PROJECT_ROOT.parent
    / "cpp simulator"
    / "build"
    / "cpp_simulator_stream.exe"
)

WORKING_DIR = PROJECT_ROOT / "data" / "working"

# Every checkpoint below is evaluated against this same Task - meaningful only if it's a fair
# common task for all of them (e.g. don't compare a slime-only checkpoint against a
# slime+redstone+observer task; see the plan's "Comparing trained models" section).
TASK: Task = BlockAttachmentTask(candidate_block_ids=[BLOCK_SLIME])

# (label, checkpoint path) pairs to compare - edit this list.
CHECKPOINTS: list[tuple[str, Path]] = [
    ("run-a", PROJECT_ROOT / "data" / "checkpoints" / "run-a.json"),
    ("run-b", PROJECT_ROOT / "data" / "checkpoints" / "run-b.json"),
]

WORKER_COUNT = 4
MAX_TICKS = 6000
SIMULATION_TIMEOUT_SECONDS = 5.0
SAMPLE_COUNT = 200
RNG_SEED = 1667


def main() -> None:
    seeds = load_candidates_from_glob(WORKING_DIR / "*.json")
    simulator_config = SimulatorRunConfig(
        simulator_path=SIMULATOR_EXE,
        worker_count=WORKER_COUNT,
        max_ticks=MAX_TICKS,
        simulation_timeout_seconds=SIMULATION_TIMEOUT_SECONDS,
    )

    probe_contexts = TASK.sample_contexts(seeds, random.Random(RNG_SEED), 1)
    if not probe_contexts:
        raise RuntimeError("Task produced no contexts to determine feature count from")
    expected_feature_count = len(TASK.features(probe_contexts[0]))

    print(f"{'label':<20}{'success_rate':>14}{'mean_reward':>14}{'samples':>10}")
    for label, checkpoint_path in CHECKPOINTS:
        policy = SharedLinearPolicy.load(checkpoint_path)

        if policy.task_name != type(TASK).__name__:
            raise ValueError(
                f"{label}: checkpoint was trained on task {policy.task_name!r}, "
                f"but comparing against {type(TASK).__name__!r}"
            )
        if len(policy.weights) != expected_feature_count:
            raise ValueError(
                f"{label}: checkpoint has {len(policy.weights)} weight(s), "
                f"but {type(TASK).__name__!r} expects {expected_feature_count} feature(s)"
            )

        result = evaluate(TASK, policy, seeds, simulator_config, SAMPLE_COUNT, rng=random.Random(RNG_SEED))
        print(
            f"{label:<20}{result['success_rate']:>14.2f}{result['mean_reward']:>14.2f}"
            f"{result['samples']:>10}"
        )


if __name__ == "__main__":
    main()
