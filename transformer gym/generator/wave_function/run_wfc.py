"""Standalone CLI: generate + simulate WFC boards and report stats. Deliberately NOT wired into
corpus.py's generator mix (see the plan) - this is for inspecting the paradigm on its own terms:
does local-constraint placement produce a meaningfully different texture than gen_random.py's
uniform noise, and what's its zero-event rate in practice (expected to be high - WFC governs
local coherence, not causal depth; see this package's __init__.py docstring).

Usage (run from "transformer gym/"):
    py generator/wave_function/run_wfc.py --count 200 [--seed 1] [--out <shard dir>]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # "transformer gym" root
import generator  # noqa: F401,E402  (runs generator's own sys.path shim)

from runner import SimRunError, simulate  # noqa: E402
from transformer_gym.encode import TooManyNodesError, encode  # noqa: E402
from transformer_gym.simlog_reader import read_footer  # noqa: E402
from wave_function.gen_wfc import generate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="if given, write accepted .simlog files here")
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args(argv)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    gen = generate(rng)
    stats = {"attempted": 0, "accepted": 0, "zero_event": 0, "sim_error": 0, "too_big": 0}
    total_events = 0
    max_attempts = args.max_attempts if args.max_attempts is not None else args.count * 20

    while stats["accepted"] < args.count and stats["attempted"] < max_attempts:
        stats["attempted"] += 1
        candidate = next(gen)
        candidate["id"] = stats["attempted"]
        try:
            log = simulate(candidate)
        except SimRunError:
            stats["sim_error"] += 1
            continue

        footer = read_footer(log)
        if footer["eventCount"] == 0:
            stats["zero_event"] += 1
        total_events += footer["eventCount"]

        if args.out is not None:
            path = args.out / f"wfc_{stats['attempted']}.simlog"
            path.write_bytes(log)
            try:
                encode(path)
            except TooManyNodesError:
                stats["too_big"] += 1
                path.unlink()
                continue

        stats["accepted"] += 1

    attempted = max(1, stats["attempted"])
    accepted = max(1, stats["accepted"])
    print(f"attempted={stats['attempted']} accepted={stats['accepted']} "
          f"zero_event={stats['zero_event']} sim_error={stats['sim_error']} too_big={stats['too_big']}")
    print(f"avg events per accepted candidate: {total_events / accepted:.1f}")
    print(f"zero-event rate (of attempted):    {stats['zero_event'] / attempted:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
