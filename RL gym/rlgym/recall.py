"""MILESTONE 3 - recall@B against exhaustive k=2 ground truth. md/ALPHAZERO.md Part 7.

> *Gate: recall@B against brute-forced 2-block ground truth, versus random ordering (point 32).*

**Why this measurement and not another.** Every other number in the project is relative: the
ladder compares the model against frequency tables, the loop compares it against a control. This
one has a **denominator**. Of the N working 2-block modifications that exist on this machine, how
many did B simulator calls actually find? Nothing else answers that, because nothing else knows
what there was to find.

That denominator costs about ten minutes of CPU once per machine, and it is why `labeller.py`
grew a `--k2` mode. It is a **test set**, never a strategy: at k=3 the same enumeration is 3.8
days and at k=5 it is twelve thousand years.

Two properties of the measurement, both deliberate:

    HASHES, NOT ROUTES   a discovery is a machine, not a path to it. Different action pairs
                         reach the same machine, and counting routes would let a search claim
                         one discovery twice.
    EQUAL BUDGET         both sides spend the same number of simulator calls. It is the only
                         currency the uninformed control can also spend, and the whole point of
                         point 32 is that the control is a genuine alternative use of the budget.

Usage:
    py -m rlgym.recall --truth data/k2/simple_machine2.json --checkpoint data/runs/stage0/best.pt
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from rlgym.config import Config
from rlgym.dataset import graph_for, machine_for
from rlgym.game import canonical_hash
from rlgym.labeller import GroundTruth, load_ground_truth
from rlgym.metrics import Metrics
from rlgym.search import Search, net_evaluator, uniform_evaluator
from rlgym.sim import SimConfig, SimulatorProcess


@dataclass
class Recall:
    name: str
    budget: int
    calls: int
    unique_seen: int
    working_found: int
    non_cargo_found: int
    working_available: int
    non_cargo_available: int

    @property
    def recall_working(self) -> float:
        return self.working_found / self.working_available if self.working_available else 0.0

    @property
    def recall_non_cargo(self) -> float:
        return (
            self.non_cargo_found / self.non_cargo_available
            if self.non_cargo_available
            else 0.0
        )

    @property
    def precision(self) -> float:
        """Working discoveries per simulator call actually spent. The efficiency half - recall
        alone would reward a search that simply spent more."""
        return self.working_found / self.calls if self.calls else 0.0


def measure(
    truth: GroundTruth,
    evaluate,
    budget: int,
    config: Config,
    sim: SimulatorProcess,
    seed: int = 0,
    episodes: int = 0,
) -> Recall:
    """Spend `budget` simulator calls under one policy and count what it found.

    The transposition cache is **per run**, not shared between the two sides. Sharing it would
    let whichever ran second inherit the other's answers for free, and the second side would look
    dramatically better for no reason other than ordering.
    """
    machine = machine_for(truth.machine, radius=config.radius)
    found_working: set[str] = set()
    found_non_cargo: set[str] = set()
    seen: set[str] = set()
    cache: dict[str, float] = {}
    calls = 0

    def oracle(candidate) -> float:
        nonlocal calls
        calls += 1
        return 1.0 if sim.simulate(candidate).get("validCycle") else 0.0

    graph = graph_for(machine, truth.machine, cache_dir=config.graphs_dir, tick_cap=config.tick_cap)
    rng = random.Random(seed)
    # Episodes are restarted until the budget runs out. A single search would exhaust its own
    # tree long before B calls on a machine this small, and would then measure the stall guard
    # rather than the policy.
    per_episode = max(1, budget // max(1, episodes or 20))
    # An episode that spends nothing is not proof the policy is finished: each new search draws
    # fresh Dirichlet noise at the root, so the next one may reach somewhere new. Giving up on
    # the first empty episode would understate a *peaked* policy specifically - which is the
    # model - and flatter the uniform control, so the tolerance matters for fairness, not just
    # for termination.
    empty = 0
    while calls < budget and empty < 5:
        search = Search(
            machine,
            graph,
            evaluate,
            oracle,
            config=config.search,
            rng=rng,
            cache=cache,
        )
        before = calls
        search.run(budget=min(per_episode, budget - calls))
        for _, candidate, reward in search.attempts:
            digest = canonical_hash(candidate)
            seen.add(digest)
            if digest in truth.working_hashes:
                found_working.add(digest)
            if digest in truth.non_cargo_hashes:
                found_non_cargo.add(digest)
        empty = 0 if calls > before else empty + 1

    return Recall(
        name=truth.machine,
        budget=budget,
        calls=calls,
        unique_seen=len(seen),
        working_found=len(found_working),
        non_cargo_found=len(found_non_cargo),
        working_available=truth.working,
        non_cargo_available=truth.non_cargo,
    )


def run(
    truth_path: Path,
    checkpoint: Path | None,
    budget: int,
    config: Config,
    seed: int = 0,
) -> dict:
    truth = load_ground_truth(truth_path)
    machine = machine_for(truth.machine, radius=config.radius)
    results: dict[str, Recall] = {}

    with SimulatorProcess(SimConfig()) as sim:
        results["control"] = measure(
            truth, uniform_evaluator(machine), budget, config, sim, seed=seed
        )
        if checkpoint is not None:
            from rlgym.train import load_checkpoint

            net, payload = load_checkpoint(checkpoint)
            net.eval()
            graph = graph_for(
                machine, truth.machine, cache_dir=config.graphs_dir, tick_cap=config.tick_cap
            )
            results["model"] = measure(
                truth, net_evaluator(net, graph), budget, config, sim, seed=seed
            )
    return {"truth": truth, "results": results}


def _report(truth: GroundTruth, results: dict[str, Recall], budget: int) -> None:
    print(f"machine              {truth.machine}   (R=1, k=2)")
    print(f"ground truth         {truth.unique} unique machines from {truth.pairs} action pairs")
    print(f"  working            {truth.working}   = {truth.working_rate * 100:.2f}%")
    print(f"  non-cargo          {truth.non_cargo}   = {truth.non_cargo_rate * 100:.3f}%")
    print(f"budget               {budget} simulator calls\n")

    header = (
        f"{'policy':<10}{'calls':>7}{'unique':>8}{'working':>9}{'recall':>9}"
        f"{'non-cargo':>11}{'nc recall':>11}{'work/call':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in ("control", "model"):
        r = results.get(name)
        if r is None:
            continue
        print(
            f"{name:<10}{r.calls:>7}{r.unique_seen:>8}{r.working_found:>9}"
            f"{r.recall_working:>8.2%}{r.non_cargo_found:>11}"
            f"{r.recall_non_cargo:>10.2%}{r.precision:>11.3f}"
        )

    if "model" in results and "control" in results:
        model, control = results["model"], results["control"]
        print()
        if control.recall_working > 0:
            print(
                f"MILESTONE 3: the model's recall is "
                f"{model.recall_working / control.recall_working:.2f}x the uninformed control's, "
                f"at equal budget."
            )
        else:
            print("MILESTONE 3: the control found nothing at this budget.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=Path("data/k2/simple_machine2.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/runs/stage0/best.pt"))
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/runs/milestone3"))
    args = parser.parse_args()

    config = Config.load(args.config)
    config.search.k = 2
    checkpoint = args.checkpoint if args.checkpoint and args.checkpoint.exists() else None
    if checkpoint is None:
        print(f"no checkpoint at {args.checkpoint}; measuring the control alone\n")

    payload = run(args.truth, checkpoint, args.budget, config, seed=args.seed)
    _report(payload["truth"], payload["results"], args.budget)

    metrics = Metrics(args.out / "metrics.jsonl", {"config": config.to_json()})
    metrics.log(
        "recall",
        machine=payload["truth"].machine,
        budget=args.budget,
        seed=args.seed,
        **{name: vars(r) for name, r in payload["results"].items()},
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
