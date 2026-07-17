"""Deterministic evaluation of a trained policy - no learning, no exploration. Reuses the exact
same sample-contexts -> build-candidates -> SimulatorPool.run_all() wiring train_loop.py has (via
its _simulate_rewards helper), but picks each policy's greedy action instead of a stochastic
sample, and never calls policy.update(). Used to measure a checkpoint's success rate, and by
compare_models.py to put several checkpoints' numbers side by side.
"""
from __future__ import annotations

import itertools
import random
from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.archive import Archive
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.simulator_pool import SimulatorPool

from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task
from rl_ml.train_loop import _simulate_rewards

Candidate = dict[str, Any]


def evaluate(
    task: Task,
    policy: SharedLinearPolicy,
    base_machines: list[Candidate],
    simulator_config: SimulatorRunConfig,
    sample_count: int,
    rng: random.Random | None = None,
    archive_path: str | None = None,
    working_writer: Any = None,
) -> dict[str, float]:
    """Draws sample_count contexts, takes the policy's greedy action on each, and reports how
    often that action actually held up in the simulator. A genuinely new working machine found
    during evaluation is still real - archive_path/working_writer are accepted for the same
    reason train() accepts them, so a discovery isn't lost just because it happened while
    measuring rather than training."""
    rng = rng if rng is not None else random.Random()
    next_id = itertools.count(1)
    archive = Archive(archive_path) if archive_path is not None else None

    contexts = task.sample_contexts(base_machines, rng, sample_count)
    feats = [task.features(context) for context in contexts]
    actions = [policy.greedy(f) for f in feats]

    with SimulatorPool(simulator_config) as pool:
        rewards = _simulate_rewards(task, contexts, actions, pool, next_id, archive, working_writer, 0)

    if archive is not None:
        archive.flush()
    if working_writer is not None:
        working_writer.flush()

    total_true_actions = sum(1 for action in actions if action)
    total_true_successes = sum(1 for action, reward in zip(actions, rewards) if action and reward > 0)

    return {
        "success_rate": total_true_successes / total_true_actions if total_true_actions else 0.0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "samples": len(contexts),
        "true_actions": total_true_actions,
        "true_successes": total_true_successes,
    }
