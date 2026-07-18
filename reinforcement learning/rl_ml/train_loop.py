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
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.hash_log import HashLog
from genetic_ml.population import canonical_hash
from genetic_ml.simulator_pool import SimulatorPool

from rl_ml.policy import SharedLinearPolicy
from rl_ml.task import Task

Candidate = dict[str, Any]


def _simulate_rewards(
    task: Task,
    contexts: list[Any],
    actions: list[bool],
    pool: Any,
    next_id: "itertools.count[int]",
    working_writer: Any,
    working_hashes: HashLog | None = None,
    not_working_hashes: HashLog | None = None,
) -> list[float]:
    """Builds candidates for every action, runs them through pool.run_all() in one batch, and
    returns one reward per context (in context order). A newly-discovered distinct working
    candidate (working_hashes.record(...) returning True) is also handed to working_writer, if
    given - that's the only place the full candidate is persisted; the hash logs never store
    more than the hash. Shared between train() and evaluate.py so both go through identical
    simulate/reward/hash-log wiring - only what happens *before* this (how actions are chosen)
    differs between them.

    working_hashes/not_working_hashes, if given, are checked before simulating each candidate: a
    shape whose canonical hash was already tried (in this run or an earlier one) skips the
    simulator entirely and its reward is recomputed straight from the cached outcome via
    task.reward_of(...) - the whole point of caching per hash is that a reward function needs
    nothing else to reproduce the exact same reward. Every candidate that IS simulated gets its
    outcome recorded afterward into whichever log matches its outcome."""
    to_simulate: list[tuple[int, Candidate, str]] = []
    cache_hit_rewards: dict[int, float] = {}
    for i, (context, action) in enumerate(zip(contexts, actions)):
        candidate = task.build_candidate(context, action, next(next_id))
        if candidate is None:
            continue
        candidate_hash = canonical_hash(candidate)
        if working_hashes is not None and working_hashes.has(candidate_hash):
            cache_hit_rewards[i] = task.reward_of(actions[i], {"validCycle": True})
            continue
        if not_working_hashes is not None and not_working_hashes.has(candidate_hash):
            cache_hit_rewards[i] = task.reward_of(actions[i], {"validCycle": False})
            continue
        to_simulate.append((i, candidate, candidate_hash))

    results = pool.run_all([candidate for _, candidate, _ in to_simulate]) if to_simulate else []

    rewards = [task.reward_of(actions[i], None) for i in range(len(contexts))]
    for i, reward in cache_hit_rewards.items():
        rewards[i] = reward
    for (i, candidate, candidate_hash), result in zip(to_simulate, results):
        rewards[i] = task.reward_of(actions[i], result)
        working = result.get("validCycle") is True
        if working:
            if working_hashes is not None:
                newly_discovered = working_hashes.record(candidate_hash)
                if newly_discovered and working_writer is not None:
                    working_writer.save(candidate)
        elif not_working_hashes is not None:
            not_working_hashes.record(candidate_hash)

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
    working_writer: Any = None,
    working_hashes_path: str | Path | None = None,
    not_working_hashes_path: str | Path | None = None,
    flush_interval_seconds: float = 1.0,
) -> SharedLinearPolicy:
    """working_writer: anything with a save(candidate) -> Path method and a flush() method - e.g.
    genetic_ml.compact_working_writer.CompactWorkingWriter - called once per newly-discovered
    distinct working candidate (not every time it's resampled). Optional; None skips this output
    - it's the only place the full candidate is persisted, the hash logs below never store more
    than the hash.

    working_hashes_path/not_working_hashes_path, if given, enable genetic_ml.hash_log.HashLog: a
    compact, hash-only record (not the full candidate/result) of every simulated shape.
    working_hashes_path keeps the full 32-byte hash (exact - a false "yes, this works" would
    corrupt the reward signal). not_working_hashes_path truncates to 8 bytes, since that side is
    the volume problem and a rare false positive there only ever costs a missed opportunity, not
    a wrong answer. Once a shape has been simulated once, _simulate_rewards recovers its cached
    outcome and reward directly on every later resample instead of paying for another simulation
    - this is what actually gets checked/updated across a resumed run. flush_interval_seconds is
    passed to both: each buffers new records and writes them to disk in one batch at most once
    per this many seconds, rather than a write per record.

    iterations: None runs indefinitely until interrupted (Ctrl+C) instead of stopping after a fixed
    count - same convention as genetic_ml.ga_loop.GAConfig.generations. Either way, a KeyboardInterrupt
    stops the loop gracefully and still flushes the checkpoint/hash logs/working_writer before
    returning, so an interrupted continuous run doesn't lose buffered discoveries."""
    rng = rng if rng is not None else random.Random()
    next_id = itertools.count(1)
    working_hashes = (
        HashLog(working_hashes_path, hash_bytes=32, flush_interval_seconds=flush_interval_seconds)
        if working_hashes_path is not None
        else None
    )
    not_working_hashes = (
        HashLog(not_working_hashes_path, hash_bytes=8, flush_interval_seconds=flush_interval_seconds)
        if not_working_hashes_path is not None
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
                    task, contexts, actions, pool, next_id, working_writer, working_hashes, not_working_hashes
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
            if working_hashes is not None:
                working_hashes.flush()
            if not_working_hashes is not None:
                not_working_hashes.flush()
            if working_writer is not None:
                working_writer.flush()

    return policy
