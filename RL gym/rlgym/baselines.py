"""The benchmark ladder - what every model number is reported against.

Beating uniform random is not evidence of anything. Milestone 1 measured real signal in block
type alone (glass 42%, glazed terracotta 15%, extended pistons 8%), so a frequency table over
palette entries is already a strong ranker - and unlike a per-cell table it **transfers to a
machine it has never seen**, because cells are machine-specific coordinates and block types are
not.

    0  uniform          nothing
    1  per-entry        block identity          TRANSFERS - this is the bar
    2  per-cell         position                does not transfer
    3  per-cell-entry   both, memorised         does not transfer

2 and 3 are deliberately fitted and evaluated on the SAME machine. They are not competitors;
they are the memorisation ceiling, and the gap between them and the model says which failure you
have:

    model well below 2/3 on training            underfitting
    model matches 2/3 on training, below 1 held-out   memorising
    model matches 2/3 on training, above 1 held-out   it generalised

Usage:
    py -m rlgym.baselines [--labels data/labels] [--budget 100]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from rlgym.labeller import Label, LabelSet, load

Scores = dict[int, float]  # action index -> score

# Held-out machines, fixed once and never trained on. Chosen for structural spread across the
# size range rather than at random, and test_flyer_21/22 are kept on the same side because they
# share a canonical_hash - splitting an exact duplicate would leak the answer across the split.
HELD_OUT = (
    "simple_caterpillar",
    "test_flyer_13",
    "test_flyer_21",
    "test_flyer_22",
    "test_flyer_5",
    "test_flyer_23",
    "test_flyer_9",
)


@dataclass(frozen=True)
class Evaluation:
    name: str
    machine: str
    budget: int
    working_at_budget: int
    non_cargo_at_budget: int
    working_available: int
    non_cargo_available: int
    auc: float

    @property
    def precision(self) -> float:
        return self.working_at_budget / self.budget if self.budget else 0.0

    @property
    def recall(self) -> float:
        return (
            self.working_at_budget / self.working_available
            if self.working_available
            else 0.0
        )


def _rank(labels: Sequence[Label], scores: Scores, rng: random.Random) -> list[Label]:
    """Highest score first. Ties are broken randomly, not by action index - otherwise a ranker
    that scores everything equally would silently inherit the enumeration order, which is sorted
    by cell and would look far better than chance."""
    order = list(labels)
    rng.shuffle(order)
    return sorted(order, key=lambda label: -scores.get(label.action, 0.0))


def _auc(ranked: Sequence[Label]) -> float:
    """Probability that a randomly chosen working action outranks a failing one. 0.5 = chance."""
    positives = sum(1 for label in ranked if label.reward > 0.0)
    negatives = len(ranked) - positives
    if not positives or not negatives:
        return 0.5
    seen_negative = 0
    concordant = 0
    for label in reversed(ranked):  # worst first
        if label.reward > 0.0:
            concordant += seen_negative
        else:
            seen_negative += 1
    return concordant / (positives * negatives)


def evaluate(
    name: str,
    labelset: LabelSet,
    scores: Scores,
    budget: int = 100,
    seed: int = 0,
) -> Evaluation:
    labels = [label for label in labelset.labels if label.duplicate_of is None]
    ranked = _rank(labels, scores, random.Random(seed))
    top = ranked[:budget]
    return Evaluation(
        name=name,
        machine=labelset.machine,
        budget=min(budget, len(ranked)),
        working_at_budget=sum(1 for label in top if label.reward > 0.0),
        non_cargo_at_budget=sum(
            1 for label in top if label.reward > 0.0 and not label.maybe_cargo
        ),
        working_available=sum(1 for label in labels if label.reward > 0.0),
        non_cargo_available=sum(
            1 for label in labels if label.reward > 0.0 and not label.maybe_cargo
        ),
        auc=_auc(ranked),
    )


# --- the four rankers -----------------------------------------------------------------------


def uniform_scores(labelset: LabelSet) -> Scores:
    """Baseline 0. Every action equal, so ranking is pure tie-breaking noise."""
    return {label.action: 0.0 for label in labelset.labels}


def fit_entry_rates(training: Iterable[LabelSet]) -> dict[str, float]:
    """Baseline 1, fitted: success rate per palette entry, pooled over training machines.

    Pooled rather than averaged per machine, so a large machine contributes proportionally to
    how much evidence it actually carries.
    """
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for labelset in training:
        for label in labelset.labels:
            if label.duplicate_of is not None:
                continue
            total[label.entry] += 1
            if label.reward > 0.0:
                hits[label.entry] += 1
    return {entry: hits[entry] / count for entry, count in total.items() if count}


def entry_scores(labelset: LabelSet, rates: dict[str, float]) -> Scores:
    return {label.action: rates.get(label.entry, 0.0) for label in labelset.labels}


def cell_scores(labelset: LabelSet) -> Scores:
    """Baseline 2, fitted on this machine: the memorisation ceiling for position alone.

    Milestone 1 showed why this is strong - 11 of 26 cells accepted 66-92% of anything, the
    other 15 accepted almost nothing.
    """
    hits: dict[tuple, int] = defaultdict(int)
    total: dict[tuple, int] = defaultdict(int)
    for label in labelset.labels:
        total[label.cell] += 1
        if label.reward > 0.0:
            hits[label.cell] += 1
    rates = {cell: hits[cell] / count for cell, count in total.items()}
    return {label.action: rates[label.cell] for label in labelset.labels}


def cell_entry_scores(labelset: LabelSet) -> Scores:
    """Baseline 3: perfect memorisation of this machine. The ceiling nothing can beat here,
    and worth exactly nothing on any other machine."""
    return {label.action: label.reward for label in labelset.labels}


# --- reporting ------------------------------------------------------------------------------


def load_corpus(directory: Path) -> dict[str, LabelSet]:
    return {path.stem: load(path) for path in sorted(directory.glob("*.json"))}


def split(corpus: dict[str, LabelSet]) -> tuple[dict[str, LabelSet], dict[str, LabelSet]]:
    held = {k: v for k, v in corpus.items() if k in HELD_OUT}
    train = {k: v for k, v in corpus.items() if k not in HELD_OUT}
    return train, held


def run(directory: Path, budget: int = 100) -> dict[str, list[Evaluation]]:
    corpus = load_corpus(directory)
    if not corpus:
        raise SystemExit(f"no label files in {directory} - run: py -m rlgym.labeller --all")
    training, held = split(corpus)
    rates = fit_entry_rates(training.values())

    out: dict[str, list[Evaluation]] = defaultdict(list)
    for machine, labelset in corpus.items():
        out["0 uniform"].append(
            evaluate("0 uniform", labelset, uniform_scores(labelset), budget)
        )
        out["1 per-entry"].append(
            evaluate("1 per-entry", labelset, entry_scores(labelset, rates), budget)
        )
        out["2 per-cell"].append(
            evaluate("2 per-cell", labelset, cell_scores(labelset), budget)
        )
        out["3 per-cell-entry"].append(
            evaluate("3 per-cell-entry", labelset, cell_entry_scores(labelset), budget)
        )
    return out, training, held, rates


def _summarise(evals: Sequence[Evaluation], names: Iterable[str]) -> tuple[float, float, float]:
    chosen = [e for e in evals if e.machine in set(names)]
    if not chosen:
        return 0.0, 0.0, 0.0
    precision = sum(e.precision for e in chosen) / len(chosen)
    auc = sum(e.auc for e in chosen) / len(chosen)
    non_cargo = sum(e.non_cargo_at_budget for e in chosen)
    return precision, auc, non_cargo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/labels"))
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    results, training, held, rates = run(args.labels, args.budget)

    print(f"corpus: {len(training)} training machines, {len(held)} held out")
    print(f"budget: top {args.budget} actions by score\n")

    header = f"{'baseline':<18}{'TRAIN prec':>11}{'TRAIN auc':>10}   {'HELD prec':>10}{'HELD auc':>9}{'HELD nc':>9}"
    print(header)
    print("-" * len(header))
    for name in ("0 uniform", "1 per-entry", "2 per-cell", "3 per-cell-entry"):
        tp, ta, _ = _summarise(results[name], training)
        hp, ha, hn = _summarise(results[name], held)
        print(f"{name:<18}{tp:>10.1%}{ta:>10.3f}   {hp:>9.1%}{ha:>9.3f}{int(hn):>9}")

    print()
    print("HELD prec is the bar the model has to clear: baseline 1 is the only one that")
    print("transfers, since cells are machine-specific and block types are not.")
    print()
    print("per-entry success rates fitted on training machines (top and bottom 6):")
    ordered = sorted(rates.items(), key=lambda kv: -kv[1])
    for entry, rate in ordered[:6]:
        print(f"   {entry:<32}{rate:>7.1%}")
    print("   ...")
    for entry, rate in ordered[-6:]:
        print(f"   {entry:<32}{rate:>7.1%}")


if __name__ == "__main__":
    main()
