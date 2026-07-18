from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genetic_ml.archive import Archive
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.failure_log import FailureLogger
from genetic_ml.mutation import mutate
from genetic_ml.population import Lineage, Population, canonical_hash
from genetic_ml.simulator_pool import SimulatorPool
from genetic_ml.tried_log import TriedLog
from genetic_ml.working_folder import WorkingFolderWriter

Candidate = dict[str, Any]


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
    archive: Archive
    generations_run: int
    total_simulated: int
    crashes_logged: int = 0
    hangs_logged: int = 0


def run_ga(
    simulator_config: SimulatorRunConfig,
    ga_config: GAConfig,
    seed_candidates: list[Candidate],
    archive_path: str,
    crash_dir: str | None = None,
    hang_dir: str | None = None,
    working_dir: str | None = None,
    working_dir_prefix: str = "discovered",
    pool: SimulatorPool | None = None,
    working_writer: object | None = None,
    flush_interval_seconds: float = 1.0,
    tried_log_path: str | None = None,
) -> GAResult:
    """pool lets a caller supply an already-constructed SimulatorPool (or subclass) instead of
    having run_ga build a plain one. Defaults to None, which preserves the exact previous
    behavior (construct a plain SimulatorPool here).

    working_writer similarly lets a caller supply an already-constructed writer - anything with
    a save(candidate) -> Path method, e.g. genetic_ml.compact_working_writer.CompactWorkingWriter
    - instead of having run_ga build a plain WorkingFolderWriter from working_dir/working_dir_prefix.
    Defaults to None, which preserves the exact previous behavior.

    flush_interval_seconds is passed to both Archive and TriedLog (see genetic_ml.archive.Archive,
    genetic_ml.tried_log.TriedLog): each buffers new records in memory and writes them to disk in
    one batch at most once per this many seconds, rather than opening/writing/closing the file on
    every single discovery - a long run can find candidates in rapid bursts, and a disk write+flush
    per discovery adds up to a lot of small writes over a long run. Raise it to cut disk writes
    further, at the cost of losing more buffered-but-unflushed discoveries if the process is
    killed uncleanly (a graceful Ctrl+C or normal exit still flushes everything).

    tried_log_path defaults to a "tried.log" file next to archive_path - see
    genetic_ml.tried_log.TriedLog. Every simulated candidate's hash (working or not) is recorded
    there so a later generation never re-simulates the same (or structurally-identical) shape,
    including ones that failed and were never accepted into the transient exploration list."""
    if not seed_candidates:
        raise ValueError("run_ga requires at least one seed candidate")

    rng = random.Random(ga_config.seed)
    population = Population(capacity=ga_config.population_capacity)
    population.seed(seed_candidates)
    archive = Archive(archive_path, flush_interval_seconds=flush_interval_seconds)
    resolved_tried_log_path = tried_log_path or str(Path(archive_path).with_name("tried.log"))
    tried_log = TriedLog(resolved_tried_log_path, flush_interval_seconds=flush_interval_seconds)
    failure_logger = FailureLogger(crash_dir, hang_dir) if crash_dir is not None and hang_dir is not None else None
    resolved_working_writer = (
        working_writer
        if working_writer is not None
        else (WorkingFolderWriter(working_dir, working_dir_prefix) if working_dir is not None else None)
    )
    owned_pool = pool if pool is not None else SimulatorPool(simulator_config, failure_logger=failure_logger)

    next_id = itertools.count(max((c["id"] for c in seed_candidates), default=0) + 1)
    exploration: list[_ExplorationEntry] = []
    total_simulated = 0
    generations_completed = 0
    # Archive.__init__ preloads every hash already on disk from a previous run, so len(archive)
    # includes history this run never found - counting that against this run's elapsed time
    # would spoof the discovery rate (e.g. reporting thousands/min right at startup just because
    # a prior run had already found that many). Track only what THIS run newly discovers.
    discoveries_this_run = 0

    generation_iter = (
        itertools.count(1) if ga_config.generations is None else range(1, ga_config.generations + 1)
    )
    generations_label = "inf" if ga_config.generations is None else str(ga_config.generations)

    run_started = time.monotonic()
    last_printed = 0.0  # 0 forces an immediate first printout on generation 1

    with owned_pool as pool:
        try:
            for generation in generation_iter:
                sources: list[tuple[Candidate, str, str | None]] = [
                    (lineage.candidate, "lineage", lineage.hash) for lineage in population.lineages
                ]
                sources.extend((entry.candidate, "exploration", None) for entry in exploration)

                offspring: list[tuple[Candidate, str, str | None, str]] = []
                seen_this_batch: set[str] = set()
                for source_candidate, source_kind, source_hash in sources:
                    for _ in range(ga_config.offspring_per_lineage):
                        op_count = rng.randint(ga_config.mutation_ops_min, ga_config.mutation_ops_max)
                        child = mutate(source_candidate, rng, op_count=op_count)
                        child["id"] = next(next_id)
                        child_hash = canonical_hash(child)
                        if (
                            child_hash in seen_this_batch
                            or population.has_seen(child_hash)
                            or archive.has(child_hash)
                            or tried_log.has(child_hash)
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

                results = pool.run_all([child for child, _, _, _ in offspring])
                total_simulated += len(results)

                for (child, source_kind, source_hash, child_hash), result in zip(offspring, results, strict=True):
                    # validCycle (not the older working/cycles hash-based heuristic) is the
                    # simulator's ground-truth check: does the machine settle and end up an exact
                    # translated copy of its starting layout, not just "a repeat was detected."
                    working = result.get("validCycle") is True
                    tried_log.record(child_hash, working)
                    if working:
                        lineage = Lineage(
                            candidate=child,
                            origin=source_kind,
                            generation_found=generation,
                            parent_hash=source_hash,
                        )
                        admitted = population.admit(lineage)
                        newly_archived = archive.record(
                            child,
                            child_hash,
                            result,
                            generation,
                            origin=source_kind if admitted else f"{source_kind}-not-admitted",
                        )
                        if newly_archived:
                            discoveries_this_run += 1
                            if resolved_working_writer is not None:
                                resolved_working_writer.save(child)
                    elif len(exploration) < ga_config.exploration_capacity and rng.random() < ga_config.accept_broken_probability:
                        exploration.append(_ExplorationEntry(child, ga_config.exploration_ttl))

                for lineage in population.lineages:
                    lineage.stale_generations += 1

                now = time.monotonic()
                if now - last_printed >= ga_config.progress_interval_seconds:
                    last_printed = now
                    elapsed_minutes = (now - run_started) / 60.0
                    discovery_rate = discoveries_this_run / elapsed_minutes if elapsed_minutes > 0 else 0.0
                    failure_suffix = ""
                    if failure_logger is not None:
                        failure_suffix = (
                            f" crashes={failure_logger.crash_count} hangs={failure_logger.hang_count}"
                        )
                    print(
                        f"gen {generation}/{generations_label}: "
                        f"found={discoveries_this_run} ({discovery_rate:.1f}/min) "
                        f"seeded={len(seed_candidates)} population={len(population)} "
                        f"exploring={len(exploration)} total_simulated={total_simulated}{failure_suffix}"
                    )
        except KeyboardInterrupt:
            print(f"\nInterrupted after {generations_completed} generation(s) - stopping gracefully...")
        finally:
            # Buffered writers (Archive, TriedLog, CompactWorkingWriter) hold discoveries in
            # memory between disk flushes - always flush before returning, whether the run
            # finished normally, was interrupted, or hit an exception, so nothing buffered is
            # lost.
            archive.flush()
            tried_log.flush()
            if resolved_working_writer is not None and hasattr(resolved_working_writer, "flush"):
                resolved_working_writer.flush()

    return GAResult(
        population=population,
        archive=archive,
        generations_run=generations_completed,
        total_simulated=total_simulated,
        crashes_logged=failure_logger.crash_count if failure_logger is not None else 0,
        hangs_logged=failure_logger.hang_count if failure_logger is not None else 0,
    )
