from __future__ import annotations

import itertools
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from genetic_ml.compact_format import encode_candidate
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.failure_log import FailureLogger
from genetic_ml.hash_log import HashLog
from genetic_ml.mutation import mutate
from genetic_ml.population import Lineage, Population, canonical_hash
from genetic_ml.simulator_pool import SimulatorPool
from genetic_ml.working_folder import WorkingFolderWriter

Candidate = dict[str, Any]


@contextmanager
def _timed(section_seconds: dict[str, float], name: str) -> Iterator[None]:
    """Accumulates wall-clock time spent in a named section of the generation loop, so the
    periodic progress printout can show where time is actually going - reset every printout
    (see run_ga) so the breakdown reflects the current window, not a lifetime average, letting a
    section that's growing slower over the course of a long run stand out."""
    started = time.perf_counter()
    try:
        yield
    finally:
        section_seconds[name] = section_seconds.get(name, 0.0) + (time.perf_counter() - started)


@dataclass(frozen=True)
class GAConfig:
    population_capacity: int = 16
    offspring_per_lineage: int = 6
    # None means run until interrupted (Ctrl+C) instead of stopping after a fixed count.
    generations: int | None = 50
    mutation_ops_min: int = 1
    mutation_ops_max: int = 3
    # Probability a non-working mutant is kept around for a few more generations of
    # mutation anyway, since some working designs are only reachable through a
    # temporarily-broken intermediate. These never enter the population or archive.
    accept_broken_probability: float = 0.1
    exploration_capacity: int = 8
    exploration_ttl: int = 3
    seed: int | None = None
    # Minimum wall-clock gap between progress printouts. A run can find candidates in rapid
    # bursts, so printing on every discovery (or even every generation) gets noisy fast over a
    # long run - this caps it to a periodic status line instead, still printed immediately for
    # the very first generation so a run doesn't look stalled right after starting.
    progress_interval_seconds: float = 15.0


@dataclass
class _ExplorationEntry:
    candidate: Candidate
    ttl: int


@dataclass
class GAResult:
    population: Population
    working_hashes: HashLog
    not_working_hashes: HashLog
    generations_run: int
    total_simulated: int
    crashes_logged: int = 0
    hangs_logged: int = 0


def run_ga(
    simulator_config: SimulatorRunConfig,
    ga_config: GAConfig,
    seed_candidates: list[Candidate],
    working_hashes_path: str,
    not_working_hashes_path: str | None = None,
    working_dir: str | None = None,
    working_dir_prefix: str = "discovered",
    pool: SimulatorPool | None = None,
    working_writer: object | None = None,
    flush_interval_seconds: float = 1.0,
    crash_dir: str | None = None,
    hang_dir: str | None = None,
    stream_hub: Any = None,
) -> GAResult:
    """pool lets a caller supply an already-constructed SimulatorPool (or subclass) instead of
    having run_ga build a plain one. Defaults to None, which preserves the exact previous
    behavior (construct a plain SimulatorPool here) - crash_dir/hang_dir are ignored when pool
    is supplied directly, since the caller's pool already owns its own failure_logger.

    crash_dir/hang_dir, if both given, enable genetic_ml.failure_log.FailureLogger: every
    candidate that crashes or hangs the simulator gets written as its own JSON repro file (crash_
    or hung_-prefixed) into the matching directory, for later use hardening the C++ simulator.
    Optional; the pool still recovers and keeps going either way (see SimulatorPool), this only
    controls whether repro files get written to disk.

    working_writer similarly lets a caller supply an already-constructed writer - anything with
    a save(candidate) -> Path method, e.g. genetic_ml.compact_working_writer.CompactWorkingWriter
    - instead of having run_ga build a plain WorkingFolderWriter from working_dir/working_dir_prefix.
    Defaults to None, which preserves the exact previous behavior. It's the sole place the full
    candidate (blocks/trigger) for a working discovery is persisted - working_hashes_path only
    ever stores the hash.

    working_hashes_path/not_working_hashes_path are genetic_ml.hash_log.HashLog files: fixed-width
    binary, no candidate/result payload, just hash bytes back-to-back. working_hashes_path keeps
    the full 32-byte hash (exact, zero collision risk - a false "yes, this works" would corrupt
    downstream reward signals). not_working_hashes_path (defaults to "not_working_hashes.log" next
    to working_hashes_path) truncates to 8 bytes, since that side is the volume problem and a rare
    false positive there only ever costs a missed opportunity, not a wrong answer.

    stream_hub: anything with a publish(frame: bytes) method - e.g. genetic_ml.stream_hub.
    StreamHub. Optional; None skips this. Every newly-discovered working candidate found in a
    generation is encoded and published as a single batch frame once that generation's
    postprocessing finishes, so a connected flyer-web-visualizer /live dashboard sees the same
    discoveries resolved_working_writer just persisted.

    flush_interval_seconds is passed to both HashLogs: each buffers new records in memory and
    writes them to disk in one batch at most once per this many seconds, rather than opening/
    writing/closing the file on every single record - a long run can find (or rule out) candidates
    in rapid bursts, and a disk write+flush per record adds up to a lot of small writes over a long
    run. Raise it to cut disk writes further, at the cost of losing more buffered-but-unflushed
    records if the process is killed uncleanly (a graceful Ctrl+C or normal exit still flushes
    everything)."""
    if not seed_candidates:
        raise ValueError("run_ga requires at least one seed candidate")

    rng = random.Random(ga_config.seed)
    population = Population(capacity=ga_config.population_capacity)
    population.seed(seed_candidates)
    working_hashes = HashLog(working_hashes_path, hash_bytes=32, flush_interval_seconds=flush_interval_seconds)
    resolved_not_working_path = not_working_hashes_path or str(
        Path(working_hashes_path).with_name("not_working_hashes.log")
    )
    not_working_hashes = HashLog(resolved_not_working_path, hash_bytes=8, flush_interval_seconds=flush_interval_seconds)
    resolved_working_writer = (
        working_writer
        if working_writer is not None
        else (WorkingFolderWriter(working_dir, working_dir_prefix) if working_dir is not None else None)
    )
    failure_logger = FailureLogger(crash_dir, hang_dir) if crash_dir is not None and hang_dir is not None else None
    owned_pool = pool if pool is not None else SimulatorPool(simulator_config, failure_logger=failure_logger)

    next_id = itertools.count(max((c["id"] for c in seed_candidates), default=0) + 1)
    exploration: list[_ExplorationEntry] = []
    total_simulated = 0
    generations_completed = 0
    # HashLog.__init__ preloads every hash already on disk from a previous run, so
    # len(working_hashes) includes history this run never found - counting that against this
    # run's elapsed time would spoof the discovery rate (e.g. reporting thousands/min right at
    # startup just because a prior run had already found that many). Track only what THIS run
    # newly discovers.
    discoveries_this_run = 0

    generation_iter = (
        itertools.count(1) if ga_config.generations is None else range(1, ga_config.generations + 1)
    )
    generations_label = "inf" if ga_config.generations is None else str(ga_config.generations)

    run_started = time.monotonic()
    last_printed = 0.0  # 0 forces an immediate first printout on generation 1
    # Reset every printout (see below) so the reported breakdown/rate reflects the current
    # window instead of a lifetime average - a section that's growing slower over the course of
    # a long run should show up as a rising share of window_seconds, not get smoothed away.
    section_seconds: dict[str, float] = {}
    window_started = run_started
    simulated_this_window = 0

    with owned_pool as pool:
        try:
            for generation in generation_iter:
                sources: list[tuple[Candidate, str, str | None]] = [
                    (lineage.candidate, "lineage", lineage.hash) for lineage in population.lineages
                ]
                sources.extend((entry.candidate, "exploration", None) for entry in exploration)

                offspring: list[tuple[Candidate, str, str | None, str]] = []
                seen_this_batch: set[str] = set()
                with _timed(section_seconds, "mutate"):
                    for source_candidate, source_kind, source_hash in sources:
                        for _ in range(ga_config.offspring_per_lineage):
                            op_count = rng.randint(ga_config.mutation_ops_min, ga_config.mutation_ops_max)
                            child = mutate(source_candidate, rng, op_count=op_count)
                            child["id"] = next(next_id)
                            child_hash = canonical_hash(child)
                            if (
                                child_hash in seen_this_batch
                                or population.has_seen(child_hash)
                                or working_hashes.has(child_hash)
                                or not_working_hashes.has(child_hash)
                            ):
                                continue
                            seen_this_batch.add(child_hash)
                            offspring.append((child, source_kind, source_hash, child_hash))

                exploration = [_ExplorationEntry(e.candidate, e.ttl - 1) for e in exploration if e.ttl - 1 > 0]

                generations_completed = generation

                if not offspring:
                    for lineage in population.lineages:
                        lineage.stale_generations += 1
                    continue

                with _timed(section_seconds, "simulate"):
                    results = pool.run_all([child for child, _, _, _ in offspring])
                total_simulated += len(results)
                simulated_this_window += len(results)

                new_working: list[Candidate] = []
                with _timed(section_seconds, "postprocess"):
                    for (child, source_kind, source_hash, child_hash), result in zip(offspring, results, strict=True):
                        # validCycle (not the older working/cycles hash-based heuristic) is the
                        # simulator's ground-truth check: does the machine settle and end up an
                        # exact translated copy of its starting layout, not just "a repeat was
                        # detected."
                        working = result.get("validCycle") is True
                        if working:
                            lineage = Lineage(
                                candidate=child,
                                origin=source_kind,
                                generation_found=generation,
                                parent_hash=source_hash,
                            )
                            population.admit(lineage)
                            newly_recorded = working_hashes.record(child_hash)
                            if newly_recorded:
                                discoveries_this_run += 1
                                if resolved_working_writer is not None:
                                    resolved_working_writer.save(child)
                                new_working.append(child)
                        else:
                            not_working_hashes.record(child_hash)
                            if (
                                len(exploration) < ga_config.exploration_capacity
                                and rng.random() < ga_config.accept_broken_probability
                            ):
                                exploration.append(_ExplorationEntry(child, ga_config.exploration_ttl))

                    for lineage in population.lineages:
                        lineage.stale_generations += 1

                if stream_hub is not None and new_working:
                    stream_hub.publish(b"".join(encode_candidate(c) for c in new_working))

                now = time.monotonic()
                if now - last_printed >= ga_config.progress_interval_seconds:
                    last_printed = now
                    elapsed_minutes = (now - run_started) / 60.0
                    discovery_rate = discoveries_this_run / elapsed_minutes if elapsed_minutes > 0 else 0.0
                    sim_rate = total_simulated / elapsed_minutes if elapsed_minutes > 0 else 0.0

                    window_seconds = now - window_started
                    profile_suffix = ""
                    if window_seconds > 0:
                        parts = [
                            f"{name}={seconds / window_seconds * 100:.0f}%"
                            for name, seconds in sorted(section_seconds.items(), key=lambda kv: -kv[1])
                        ]
                        window_sim_rate = simulated_this_window / window_seconds * 60.0
                        profile_suffix = (
                            f" | window={window_seconds:.1f}s sim_rate_window={window_sim_rate:.1f}/min "
                            f"{' '.join(parts)}"
                        )
                    section_seconds = {}
                    window_started = now
                    simulated_this_window = 0

                    crash_count = getattr(pool, "crash_count", 0)
                    hang_count = getattr(pool, "hang_count", 0)
                    print(
                        f"gen {generation}/{generations_label}: "
                        f"found={discoveries_this_run} ({discovery_rate:.1f}/min) "
                        f"seeded={len(seed_candidates)} population={len(population)} "
                        f"exploring={len(exploration)} total_simulated={total_simulated} "
                        f"({sim_rate:.1f}/min) (C/H):({crash_count},{hang_count}){profile_suffix}"
                    )
        except KeyboardInterrupt:
            print(f"\nInterrupted after {generations_completed} generation(s) - stopping gracefully...")
        finally:
            # Buffered writers (HashLog x2, CompactWorkingWriter) hold records in memory between
            # disk flushes - always flush before returning, whether the run finished normally,
            # was interrupted, or hit an exception, so nothing buffered is lost.
            working_hashes.flush()
            not_working_hashes.flush()
            if resolved_working_writer is not None and hasattr(resolved_working_writer, "flush"):
                resolved_working_writer.flush()

    return GAResult(
        population=population,
        working_hashes=working_hashes,
        not_working_hashes=not_working_hashes,
        generations_run=generations_completed,
        total_simulated=total_simulated,
        crashes_logged=getattr(pool, "crash_count", 0),
        hangs_logged=getattr(pool, "hang_count", 0),
    )
