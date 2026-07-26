"""CLI: py generate.py --out <dir> --count N [--seed S]
Generates a shard directory of .simlog files for transformer_gym.dataset.SimlogDirDataset, then
prints the corpus.build_corpus stats and the coverage matrix (doc §8's "run before scaling up").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import generator` resolves
import generator  # noqa: F401,E402
from corpus import build_corpus
from coverage import build_matrix, print_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    stats = build_corpus(args.out, args.count, seed=args.seed)
    print(f"generated {stats['accepted']}/{args.count} to {args.out} "
          f"(attempted={stats['attempted']} dup={stats['dup']} sim_error={stats['sim_error']} "
          f"dead={stats['dead']} too_big={stats['too_big']})")
    print()
    print_matrix(build_matrix(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
