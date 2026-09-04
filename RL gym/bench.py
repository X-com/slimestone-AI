"""Timing baseline for every stage of the loop - the build plan's `bench.py`.

**This never fails the build.** It reports numbers; whether a number is acceptable is a decision
for a person reading `md/SLOWDOWNS.md`, not an assertion. A benchmark that fails CI becomes a
benchmark that gets its threshold raised until it stops complaining.

The one number worth watching is the ratio at the bottom: simulation is ~0.26 ms and everything
else is measured against it, because the plan's known constraint is that torch is CPU-only here,
so training and simulation contend for the same cores rather than running on separate hardware
as point 22 assumed.

    py bench.py [--machine simple_observer_engine] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def timed(fn, repeats: int = 3):
    """Best of N, not mean. A slow run is contention with something else on the box; the fastest
    run is the closest thing to the cost of the work itself."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000)
    return min(times), statistics.median(times), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="simple_observer_engine")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from rlgym.dataset import machine_for
    from rlgym.game import GameState, Placement
    from rlgym.graph import apply_notes, build
    from rlgym.net import Net, make_batch
    from rlgym.simlog import record_for

    rows: list[dict] = []

    def report(name: str, best: float, median: float, note: str = "") -> None:
        rows.append({"step": name, "best_ms": round(best, 3), "median_ms": round(median, 3)})
        print(f"{name:<34}{best:>10.3f} ms{('   ' + note) if note else ''}")

    machine = machine_for(args.machine)
    print(f"machine: {args.machine}  ({len(machine.cells)} blocks, "
          f"{len(machine.cell_list)} candidate cells, {machine.action_count} actions)\n")

    # Two different simulator costs, and confusing them makes every ratio below wrong.
    # `record_for` spawns a process and writes a file; the loop pays neither, because it holds
    # one SimulatorProcess open and only reads a verdict. The verdict is the honest denominator.
    from rlgym.sim import SimulatorProcess

    with SimulatorProcess() as process:
        candidate = machine.to_candidate(cid=0)
        process.simulate(candidate)  # warm the process; the first call pays for startup
        best, median, _ = timed(lambda: process.simulate(candidate), args.repeats * 10)
    report("simulate, verdict only", best, median, "what the loop actually pays")

    best, median, record = timed(lambda: record_for(machine.to_candidate(cid=0)), args.repeats)
    report("simulate + write a record", best, median, "spawn + file; graph builds only")

    best, median, graph = timed(lambda: build(machine, record), args.repeats)
    report("graph build", best, median, f"{graph.n_items} items, {graph.n_edges} edges")

    cell = next(c for c in machine.cell_list if c not in machine.cells)
    placement = (Placement(cell, 30),)
    best, median, _ = timed(lambda: apply_notes(graph, placement), args.repeats * 10)
    report("note patch (per candidate)", best, median, "the reason one build serves all")

    net = Net()
    batch = make_batch([graph])
    with torch.no_grad():
        best, median, _ = timed(lambda: net(batch), args.repeats)
    report("network forward", best, median, f"{net.n_parameters():,} parameters")

    legal = GameState(machine, k=1).legal_mask()
    with torch.no_grad():
        best, median, _ = timed(lambda: net.priors(graph, legal), args.repeats)
    report("prior + value for one state", best, median)

    def backward():
        out = net(batch)
        (out.value.sum() + out.policy_logits.sum()).backward()
        net.zero_grad(set_to_none=True)

    best, median, _ = timed(backward, args.repeats)
    report("forward + backward", best, median)

    simulate = next(r for r in rows if r["step"] == "simulate, verdict only")["best_ms"]
    graph_ms = next(r for r in rows if r["step"] == "graph build")["best_ms"]
    forward = next(r for r in rows if r["step"] == "network forward")["best_ms"]
    print()
    print(f"graph build   / simulate  =  {graph_ms / max(simulate, 1e-9):>7.1f}x")
    print(f"net forward   / simulate  =  {forward / max(simulate, 1e-9):>7.1f}x")
    print("\nSLOWDOWNS.md #3 is the graph ratio. It is mitigated by the note patch above, not")
    print("by making the build faster: one build per machine serves every candidate on it.")
    if forward > simulate:
        print()
        print("The second ratio is the one that changes a design decision. ALPHAZERO.md Part 4")
        print("counts budget in simulator calls because the network runs on separate hardware at")
        print("~1 ms. Here it does not: torch is CPU-only, so the NETWORK is the bottleneck and")
        print("an MCTS iteration costs more than the simulator call it is trying to save.")
        print("Watch it - if it stays this way, 'simulator calls' stops being the honest unit.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"machine": args.machine, "rows": rows}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
