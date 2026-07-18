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
from genetic_ml.tried_log import TriedLog

from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task

Candidate = dict[str, Any]


def _simulate_rewards(
    task: Task,
    contexts: list[Any],
    actions: list[bool],
    pool: Any,
    next_id: "itertools.count[int]",
    archive: Archive | None,
    working_writer: Any,
    generation: int,
    tried_log: TriedLog | None = None,
) -> list[float]:
    """Builds candidates for every action, runs them through pool.run_all() in one batch, and
    returns one reward per context (in context order). A newly-discovered distinct working
    candidate (archive.record(...) returning True) is also handed to working_writer, if given.
    Shared between train() and evaluate.py so both go through identical simulate/reward/archive
    wiring - only what happens *before* this (how actions are chosen) differs between them.

    tried_log, if given, is checked before simulating each candidate: a shape whose canonical
    hash was already tried (working or not, in this run or an earlier one) skips the simulator
    entirely and its reward is recomputed straight from the cached outcome via
    task.reward_of(...) - the whole point of caching working/failing per hash is that a reward
    function needs nothing else to reproduce the exact same reward. Every candidate that IS
    simulated gets its outcome recorded afterward, whether or not it turns out to work."""
    to_simulate: list[tuple[int, Candidate, str]] = []
    cache_hit_rewards: dict[int, float] = {}
    for i, (context, action) in enumerate(zip(contexts, actions)):
        candidate = task.build_candidate(context, action, next(next_id))
        if candidate is None:
            continue
        candidate_hash = canonical_hash(candidate)
        cached_working = tried_log.outcome(candidate_hash) if tried_log is not None else None
        if cached_working is not None:
            cache_hit_rewards[i] = task.reward_of(actions[i], {"validCycle": cached_working})
            continue
        to_simulate.append((i, candidate, candidate_hash))

    results = pool.run_all([candidate for _, candidate, _ in to_simulate]) if to_simulate else []

    rewards = [task.reward_of(actions[i], None) for i in range(len(contexts))]
    for i, reward in cache_hit_rewards.items():
        rewards[i] = reward
    for (i, candidate, candidate_hash), result in zip(to_simulate, results):
        rewards[i] = task.reward_of(actions[i], result)
        working = result.get("validCycle") is True
        if tried_log is not None:
            tried_log.record(candidate_hash, working)
        if archive is not None and rewards[i] > 0:
            newly_discovered = archive.record(
                candidate,
                candidate_hash,
                result,
                generation,
                origin=type(task).__name__,
            )
            if newly_discovered and working_writer is not None:
                working_writer.save(candidate)

    return rewards


def train(
    task: Task,
    policy: SharedLinearPolicy,
    base_machines: list[Candidate],
    simulator_config: SimulatorRunConfig,
    iterations: int | None,
    batch_size: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 20,
    progress_every: int = 10,
    rng: random.Random | None = None,
    archive_path: str | Path | None = None,
    working_writer: Any = None,
    tried_log_path: str | Path | None = None,
    flush_interval_seconds: float = 1.0,
) -> SharedLinearPolicy:
    """working_writer: anything with a save(candidate) -> Path method and a flush() method - e.g.
    genetic_ml.compact_working_writer.CompactWorkingWriter - called once per newly-discovered
    distinct working candidate (not every time it's resampled). Optional; None skips this output.

    tried_log_path, if given, enables genetic_ml.tried_log.TriedLog: a compact hash-only record
    (not the full candidate/result) of every simulated shape, working or not. Once a shape has
    been simulated once, _simulate_rewards recovers its cached outcome and reward directly on
    every later resample instead of paying for another simulation - this is what actually gets
    checked/updated across a resumed run, since archive_path alone only ever remembers working
    shapes. flush_interval_seconds is passed to both Archive and TriedLog: each buffers new
    records and writes them to disk in one batch at most once per this many seconds, rather than
    a write per discovery.

    iterations: None runs indefinitely until interrupted (Ctrl+C) instead of stopping after a fixed
    count - same convention as genetic_ml.ga_loop.GAConfig.generations. Either way, a KeyboardInterrupt
    stops the loop gracefully and still flushes the checkpoint/archive/working_writer before
    returning, so an interrupted continuous run doesn't lose buffered discoveries."""
    rng = rng if rng is not None else random.Random()
    next_id = itertools.count(1)
    archive = Archive(archive_path, flush_interval_seconds=flush_interval_seconds) if archive_path is not None else None
    tried_log = (
        TriedLog(tried_log_path, flush_interval_seconds=flush_interval_seconds)
        if tried_log_path is not None
        else None
    )

    total_true_actions = 0
    total_true_successes = 0
    completed_iterations = 0
    iteration_label = "inf" if iterations is None else str(iterations)
    iteration_iter = itertools.count(1) if iterations is None else range(1, iterations + 1)

    with SimulatorPool(simulator_config) as pool:
        try:
            for iteration in iteration_iter:
                completed_iterations = iteration
                contexts = task.sample_contexts(base_machines, rng, batch_size)
                if not contexts:
                    continue

                feats = [task.features(context) for context in contexts]
                actions = [policy.sample(f, rng) for f in feats]

                rewards = _simulate_rewards(
                    task, contexts, actions, pool, next_id, archive, working_writer, iteration, tried_log
                )

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
                        f"iter={iteration}/{iteration_label} mean_reward={mean_reward:.2f} "
                        f"action_success_rate={success_rate:.2f} "
                        f"({total_true_successes}/{total_true_actions} actions succeeded)"
                    )
        except KeyboardInterrupt:
            print(f"\nInterrupted after {completed_iterations} iteration(s) - stopping gracefully...")
        finally:
            if checkpoint_path is not None:
                policy.save(checkpoint_path)
            if archive is not None:
                archive.flush()
            if tried_log is not None:
                tried_log.flush()
            if working_writer is not None:
                working_writer.flush()

    return policy
