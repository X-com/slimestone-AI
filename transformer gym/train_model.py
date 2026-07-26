"""Run the physics-transformer training curriculum and print the held-out results to console.

Usage (run from this folder, "transformer gym/"):
    py train_model.py                                        # hand fixtures only
    py train_model.py --shard-dir generator/generated         # + a synthetic held-out curve
    py train_model.py --save checkpoints/model.pt             # also save weights to reload later

Prints, in order: per-phase curriculum loss (movement -> event_grid -> structure), then held-out
accuracy against the fixtures never trained on (generator/holdout.py's fixed list), and - if
--shard-dir points at a generator/generate.py output folder - the same metrics against a held-out
slice of that synthetic data, for the real-vs-synthetic divergence check described in
generator-design-handoff.md §8.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformer_gym.eval import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help="a generator/generate.py output directory to add a synthetic held-out curve")
    parser.add_argument("--save", type=Path, default=None,
                        help="path to save the trained model's state_dict (e.g. checkpoints/model.pt)")
    args = parser.parse_args(argv)

    evaluate(synthetic_shard_dir=args.shard_dir, save_path=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
