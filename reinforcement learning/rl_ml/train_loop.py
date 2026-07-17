"""Generic, task-agnostic training driver. Attaches to genetic_ml.SimulatorPool (worker threads
driving multiple long-lived cpp_simulator_stream.exe processes) and runs the
sample -> simulate -> reward -> policy-update loop for any Task implementation - this file never
needs to change when the training objective (rl_ml/tasks/*.py) does.
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import rl_ml  # noqa: F401  (sys.path shim for genetic_ml, must run before the imports below)
from genetic_ml.archive import Archive
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.population import canonical_hash
from genetic_ml.simulator_pool import SimulatorPool

from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task

Candidate = dict[str, Any]


def train(
    task: Task,
    policy: SharedLinearPolicy,
    base_machines: list[Candidate],
    simulator_config: SimulatorRunConfig,
    iterations: int,
    batch_size: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 20,
    progress_every: int = 10,
    rng: random.Random | None = None,
    archive_path: str | Path | None = None,
    working_writer: Any = None,
) -> SharedLinearPolicy:
    """working_writer: anything with a save(candidate) -> Path method and a flush() method - e.g.
    genetic_ml.compact_working_writer.CompactWorkingWriter - called once per newly-discovered
    distinct working candidate (not every time it's resampled). Optional; None skips this output."""
    rng = rng if rng is not None else random.Random()
    next_id = itertools.count(1)
    archive = Archive(archive_path) if archive_path is not None else None

    total_true_actions = 0
    total_true_successes = 0

    with SimulatorPool(simulator_config) as pool:
        for iteration in range(1, iterations + 1):
            contexts = task.sample_contexts(base_machines, rng, batch_size)
            if not contexts:
                continue

            feats = [task.features(context) for context in contexts]
            actions = [policy.sample(f, rng) for f in feats]

            to_simulate: list[tuple[int, Candidate]] = []
            for i, (context, action) in enumerate(zip(contexts, actions)):
                candidate = task.build_candidate(context, action, next(next_id))
                if candidate is not None:
                    to_simulate.append((i, candidate))

            results = pool.run_all([candidate for _, candidate in to_simulate]) if to_simulate else []

            rewards = [task.reward_of(actions[i], None) for i in range(len(contexts))]
            for (i, candidate), result in zip(to_simulate, results):
                rewards[i] = task.reward_of(actions[i], result)
                if archive is not None and rewards[i] > 0:
                    newly_discovered = archive.record(
                        candidate,
                        canonical_hash(candidate),
                        result,
                        iteration,
                        origin=type(task).__name__,
                    )
                    if newly_discovered and working_writer is not None:
                        working_writer.save(candidate)

            for action, reward in zip(actions, rewards):
                if action:
                    total_true_actions += 1
                    if reward > 0:
                        total_true_successes += 1

            policy.update(list(zip(feats, actions, rewards)))

            if checkpoint_path is not None and iteration % checkpoint_every == 0:
                policy.save(checkpoint_path)

            if iteration % progress_every == 0 or iteration == iterations:
                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                success_rate = total_true_successes / total_true_actions if total_true_actions else 0.0
                print(
                    f"iter={iteration}/{iterations} mean_reward={mean_reward:.2f} "
                    f"action_success_rate={success_rate:.2f} "
                    f"({total_true_successes}/{total_true_actions} actions succeeded)"
                )

        if checkpoint_path is not None:
            policy.save(checkpoint_path)
        if archive is not None:
            archive.flush()
        if working_writer is not None:
            working_writer.flush()

    return policy
