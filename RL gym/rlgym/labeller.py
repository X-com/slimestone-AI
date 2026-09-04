"""Exhaustive k-block labelling - md/ALPHAZERO.md Part 8, Milestone 1.

At k=1 the model is useless for discovery: brute force gives the complete, exact answer in
seconds. What that buys instead is a perfect supervised curriculum - the true optimal policy
target, dense over every action, which no amount of search can produce (Part 4).

Running this as a script IS Milestone 1. Its headline number - what fraction of single-block
modifications leave the machine working - is currently unknown, several decisions rest on it,
and no further design can resolve it:

    ~40%    the imbalance machinery is over-built and the task may be too easy
    ~2%     as assumed; proceed to Phase B unchanged
    ~0.1%   the supervised bootstrap fails on one machine

Usage:
    py -m rlgym.labeller simple_observer_engine [--radius 1] [--workers 12] [--out labels.json]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlgym.blocks import PALETTE
from rlgym.game import (
    Candidate,
    GameState,
    Machine,
    Placement,
    canonical_hash,
    decode_action,
)
from rlgym.sim import SimConfig, simulate_all

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "flying machines" / "json"


def load_fixture(name: str) -> Candidate:
    path = Path(name)
    if not path.exists():
        path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No fixture named {name} in {FIXTURE_DIR}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Label:
    """One exhaustively-determined action outcome. The reward is a float from the start."""

    action: int
    cell: tuple[int, int, int]
    slot: int
    entry: str
    reward: float
    period: int
    shift: tuple[int, int, int]
    working: bool
    # Cheap half of OPEN.md's cargo detector: same period and same net shift as the base.
    # The full test also requires every original block to behave identically at every tick,
    # which needs the sim-log record and so arrives in Phase B. This half is already decisive
    # in practice and costs nothing, since both fields are in the verdict.
    maybe_cargo: bool = False
    duplicate_of: int | None = None


@dataclass
class LabelSet:
    machine: str
    radius: int
    k: int
    action_slots: int
    legal_actions: int
    simulated: int
    duplicates: int
    working: int
    maybe_cargo: int = 0
    labels: list[Label] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.working / self.legal_actions if self.legal_actions else 0.0

    @property
    def non_cargo(self) -> int:
        return self.working - self.maybe_cargo

    @property
    def non_cargo_rate(self) -> float:
        return self.non_cargo / self.legal_actions if self.legal_actions else 0.0


def enumerate_k1(machine: Machine) -> list[tuple[int, Placement]]:
    """Every legal single-block action, as (action index, placement)."""
    state = GameState(machine, k=1)
    mask = state.legal_mask()
    return [
        (index, decode_action(index, machine.cell_list))
        for index, legal in enumerate(mask)
        if legal
    ]


def label_k1(
    name: str,
    radius: int = 1,
    workers: int = 12,
    config: SimConfig | None = None,
) -> LabelSet:
    fixture = load_fixture(name)
    machine = Machine.from_candidate(fixture, radius=radius)
    actions = enumerate_k1(machine)

    # Deduplicate before simulating (point 23). Different actions can reach the same machine -
    # most obviously replacing a block with what a neighbouring action also produces - and a
    # duplicate is a free label rather than a wasted simulation, so the result is copied across
    # afterwards instead of being re-derived.
    by_hash: dict[str, int] = {}
    to_run: list[Candidate] = []
    plan: list[tuple[int, Placement, str, int | None]] = []
    base_hash = canonical_hash(machine.to_candidate())

    for action, placement in actions:
        candidate = machine.to_candidate(
            GameState(machine, (placement,), k=1).current_cells(), cid=action
        )
        digest = canonical_hash(candidate)
        first = by_hash.get(digest)
        if first is None and digest != base_hash:
            by_hash[digest] = action
            to_run.append(candidate)
            plan.append((action, placement, digest, None))
        else:
            plan.append((action, placement, digest, first if first is not None else -1))

    # The base machine is simulated too, and first: its period and shift are what the cargo
    # detector compares against, so they must come from the same build path as everything else
    # rather than being read off a fixture or assumed.
    base_candidate = machine.to_candidate(cid=-1)
    started = time.perf_counter()
    all_results = simulate_all([base_candidate, *to_run], workers=workers, config=config)
    wall = time.perf_counter() - started
    base_result, results = all_results[0], all_results[1:]
    if not base_result.get("validCycle"):
        raise ValueError(
            f"Base machine {name} does not itself have a valid cycle "
            f"(errorCode={base_result.get('errorCode')!r}). Every label would be measured "
            "against a broken reference."
        )

    by_action: dict[int, dict[str, Any]] = {r["id"]: r for r in results}

    labels: list[Label] = []
    working = 0
    cargo = 0
    disagreements = 0
    rejected = Counter()
    failures = Counter()
    periods = Counter()
    shifts = Counter()
    elapsed_ns = 0
    base_period = int(base_result.get("period", 0))
    base_shift_raw = base_result.get("finalShift") or {"x": 0, "y": 0, "z": 0}
    base_shift = (base_shift_raw["x"], base_shift_raw["y"], base_shift_raw["z"])

    for action, placement, digest, duplicate_of in plan:
        source = action if duplicate_of is None else duplicate_of
        result = by_action.get(source)
        if result is None:
            failures["missing_verdict"] += 1
            continue
        if not result.get("ok", False):
            # A refusal is a legitimate reward-0 outcome, not a crash. The common one here is
            # "structural-verify trigger must point at an observer or a piston", which fires
            # when the modification overwrites the trigger cell or lands an extended piston
            # head on it. Counted separately so a genuine crash cannot hide among them.
            rejected[str(result.get("error", result.get("errorCode", "unknown")))[:60]] += 1
        reward = 1.0 if result.get("validCycle") else 0.0
        period = int(result.get("period", 0))
        shift = result.get("finalShift") or {"x": 0, "y": 0, "z": 0}
        shift_t = (shift["x"], shift["y"], shift["z"])
        is_cargo = reward > 0.0 and period == base_period and shift_t == base_shift
        entry = PALETTE[placement.slot]
        labels.append(
            Label(
                action=action,
                cell=placement.cell,
                slot=placement.slot,
                entry=entry.name if entry else "reserved",
                reward=reward,
                period=period,
                shift=shift_t,
                working=bool(result.get("working")),
                maybe_cargo=is_cargo,
                duplicate_of=None if duplicate_of is None else source,
            )
        )
        if reward > 0.0:
            working += 1
            cargo += int(is_cargo)
            periods[period] += 1
            shifts[shift_t] += 1
        if bool(result.get("working")) != bool(result.get("validCycle")):
            disagreements += 1
        if duplicate_of is None:
            elapsed_ns += int(result.get("elapsedNs", 0))

    duplicates = sum(1 for _, _, _, d in plan if d is not None)
    return LabelSet(
        machine=name,
        radius=radius,
        k=1,
        action_slots=machine.action_count,
        legal_actions=len(actions),
        simulated=len(to_run),
        duplicates=duplicates,
        working=working,
        maybe_cargo=cargo,
        labels=labels,
        diagnostics={
            "candidate_cells": len(machine.cell_list),
            "base_blocks": len(machine.cells),
            "base_period": base_period,
            "base_shift": base_shift,
            "wall_seconds": round(wall, 3),
            "sims_per_second": round(len(to_run) / wall, 1) if wall > 0 else 0.0,
            # Mean of the C++ side's own per-candidate timer. Wall time is NOT comparable to
            # its sum once workers > 1 - the sum counts every worker's time while the clock
            # counts one - so overhead is deliberately not derived by subtracting them.
            "mean_elapsed_ms": round(elapsed_ns / 1e6 / max(1, len(to_run)), 4),
            "working_validcycle_disagreements": disagreements,
            "rejected_by_simulator": dict(rejected.most_common(5)),
            "simulate_failures": dict(failures),
            "working_periods": dict(periods.most_common(10)),
            "working_shifts": {str(k): v for k, v in shifts.most_common(10)},
        },
    )


def _report(result: LabelSet) -> str:
    d = result.diagnostics
    lines = [
        f"machine                {result.machine}  ({d['base_blocks']} blocks)",
        f"shell radius           R={result.radius}  ->  {d['candidate_cells']} candidate cells",
        f"action slots           {result.action_slots}  (cells x 48 + stop)",
        f"legal actions          {result.legal_actions}",
        f"simulated              {result.simulated}   duplicates skipped {result.duplicates}",
        f"base machine           period {d['base_period']}, shift {d['base_shift']}",
        "",
        f"WORKING                {result.working}   =  {result.success_rate * 100:.2f}% "
        "of legal actions",
        f"  of which maybe cargo {result.maybe_cargo}   (same period AND same shift as base)",
        f"NON-CARGO WORKING      {result.non_cargo}   =  {result.non_cargo_rate * 100:.2f}%",
        "",
        f"wall time              {d['wall_seconds']}s   ({d['sims_per_second']} sims/sec)",
        f"mean elapsedNs         {d['mean_elapsed_ms']} ms per candidate (C++ side timer)",
        f"working != validCycle  {d['working_validcycle_disagreements']}",
        f"rejected by simulator  {d['rejected_by_simulator'] or 'none'}",
        f"simulate failures      {d['simulate_failures'] or 'none'}",
        f"periods of workers     {d['working_periods']}",
        f"shifts of workers      {d['working_shifts']}",
    ]
    return "\n".join(lines)


def save(result: LabelSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["success_rate"] = result.success_rate
    payload["non_cargo_rate"] = result.non_cargo_rate
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load(path: Path) -> LabelSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = [
        Label(
            action=item["action"],
            cell=tuple(item["cell"]),
            slot=item["slot"],
            entry=item["entry"],
            reward=item["reward"],
            period=item["period"],
            shift=tuple(item["shift"]),
            working=item["working"],
            maybe_cargo=item["maybe_cargo"],
            duplicate_of=item["duplicate_of"],
        )
        for item in payload["labels"]
    ]
    return LabelSet(
        machine=payload["machine"],
        radius=payload["radius"],
        k=payload["k"],
        action_slots=payload["action_slots"],
        legal_actions=payload["legal_actions"],
        simulated=payload["simulated"],
        duplicates=payload["duplicates"],
        working=payload["working"],
        maybe_cargo=payload["maybe_cargo"],
        labels=labels,
        diagnostics=payload["diagnostics"],
    )


def corpus_names(max_blocks: int = 200) -> list[str]:
    """Every fixture small enough to label exhaustively, by block count.

    Machines without a valid cycle are NOT filtered here - label_k1 refuses them itself, with a
    reason, which is more useful than silently omitting them from a sweep.
    """
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if len(data.get("blocks") or []) <= max_blocks:
            out.append(path.stem)
    return out


def label_corpus(
    names: list[str] | None = None,
    radius: int = 1,
    workers: int = 12,
    out_dir: Path | None = None,
    max_blocks: int = 200,
) -> dict[str, LabelSet]:
    """Exhaustively label every usable fixture. ~9 minutes for the whole small corpus.

    This is what replaces DECISIONS.md point 17's "train on one machine": ~250,000 dense, exact
    labels over ~33 machines, which makes generalisation measurable instead of asserted.
    """
    names = names or corpus_names(max_blocks)
    results: dict[str, LabelSet] = {}
    skipped: list[tuple[str, str]] = []
    started = time.perf_counter()

    for index, name in enumerate(names, start=1):
        if out_dir is not None:
            cached = out_dir / f"{name}.json"
            if cached.exists():
                results[name] = load(cached)
                print(f"[{index}/{len(names)}] {name}: cached")
                continue
        try:
            result = label_k1(name, radius=radius, workers=workers)
        except ValueError as exc:
            skipped.append((name, str(exc).split("(")[0].strip()))
            print(f"[{index}/{len(names)}] {name}: SKIPPED - no valid cycle")
            continue
        results[name] = result
        if out_dir is not None:
            save(result, out_dir / f"{name}.json")
        print(
            f"[{index}/{len(names)}] {name}: {result.legal_actions} actions, "
            f"{result.working} working ({result.success_rate * 100:.1f}%), "
            f"{result.non_cargo} non-cargo"
        )

    wall = time.perf_counter() - started
    total_actions = sum(r.legal_actions for r in results.values())
    total_working = sum(r.working for r in results.values())
    total_non_cargo = sum(r.non_cargo for r in results.values())
    print()
    print(f"machines labelled     {len(results)}   skipped {len(skipped)} (no valid cycle)")
    print(f"total actions         {total_actions}")
    print(
        f"total working         {total_working} "
        f"({total_working / max(1, total_actions) * 100:.2f}%)"
    )
    print(
        f"total non-cargo       {total_non_cargo} "
        f"({total_non_cargo / max(1, total_actions) * 100:.3f}%)"
    )
    print(f"wall time             {wall:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("machine", nargs="?", default="simple_observer_engine")
    parser.add_argument("--all", action="store_true", help="sweep the whole small corpus")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-blocks", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.all:
        out_dir = args.out or Path("data/labels")
        label_corpus(
            radius=args.radius,
            workers=args.workers,
            out_dir=out_dir,
            max_blocks=args.max_blocks,
        )
        print(f"\nwrote {out_dir}")
        return

    result = label_k1(args.machine, radius=args.radius, workers=args.workers)
    print(_report(result))
    if args.out:
        save(result, args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
