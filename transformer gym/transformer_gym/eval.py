"""Held-out eval. Reports the real-fixture curve against generator/holdout.py's fixed list (never
trained/perturbed/spliced on by any generator - see that file), plus an optional synthetic
held-out curve from a generated shard directory. Watching both side by side is the doc's
divergence check (§8): if synthetic accuracy is high but real accuracy lags, the generator has
drifted from the deployment distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .dataset import SimlogDataset, SimlogDirDataset, collate, fixture_names
from .mechanics import binary_accuracy as _binary_accuracy
from .mechanics import mechanic_accuracy
from .train import train

_GENERATOR_DIR = Path(__file__).resolve().parents[1] / "generator"
if str(_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_DIR))
from holdout import HOLDOUT_FIXTURES  # noqa: E402


def _report(name: str, model: torch.nn.Module, batch: dict) -> None:
    with torch.no_grad():
        out = model(batch)
    mask = batch["mask"]
    print(f"-- {name} --")
    print(f"moves accuracy:          {_binary_accuracy(out['moves'], batch['y_moves'], mask):.3f}")
    print(f"stays_attached accuracy: {_binary_accuracy(out['stays_attached'], batch['y_stays_attached'], mask):.3f}")
    grid_mask = mask.unsqueeze(-1).expand_as(out["event_grid"])
    print(f"event_grid accuracy:     {_binary_accuracy(out['event_grid'], batch['y_event_grid'], grid_mask):.3f}")
    term_correct = (out["termination"].argmax(-1) == batch["y_termination"]).float().mean().item()
    print(f"termination accuracy:    {term_correct:.3f}")
    print(f"valid_cycle accuracy:    {((out['valid_cycle'] > 0).float() == batch['y_valid_cycle']).float().mean().item():.3f}")
    print(f"net_shift MAE:           {(out['net_shift'] - batch['y_net_shift']).abs().mean().item():.3f}")


def _report_mechanics(title: str, scores: dict[str, float]) -> None:
    if not scores:
        print(f"-- {title} -- (no mechanic-tagged samples present)")
        return
    print(f"-- {title} --")
    for tag, score in scores.items():
        print(f"  {tag:<15} {score:.3f}")


def evaluate(
    synthetic_shard_dir: Path | None = None,
    synthetic_train_frac: float = 0.8,
    save_path: Path | None = None,
    progress_log: Path | None = None,
) -> torch.nn.Module:
    names = fixture_names()
    train_names = [n for n in names if n not in HOLDOUT_FIXTURES]
    held_out = [n for n in names if n in HOLDOUT_FIXTURES]
    print(f"training on {len(train_names)} fixtures, holding out {len(held_out)} (generator/holdout.py)")

    model = train(train_names, progress_log=progress_log)
    model.eval()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"saved model weights to {save_path}")

    ds = SimlogDataset(held_out)
    if not ds.samples:
        print("no held-out fixtures small enough to encode (see MAX_NODES) - nothing to eval")
    else:
        _report("real held-out fixtures", model, collate(ds.samples))
        _report_mechanics("held-out mechanic scores (derived tags)", mechanic_accuracy(model, ds.samples))

    # Hand-tagged verification suite (design doc §2/§"Tag source"): the precise ground-truth
    # cross-check against the coarse derived-tag scores above - if a derived-tag slice looks fine
    # but its matching hand fixture here doesn't, the derived slice was being carried by a
    # co-occurring mechanic, not the one it's nominally tracking.
    from .mechanic_fixtures import build_suite

    suite = build_suite()
    _report_mechanics(
        "hand-tagged fixture scores (precise)",
        {name: mechanic_accuracy(model, [sample])[name] for name, sample, _ in suite},
    )

    if synthetic_shard_dir is not None:
        synth = SimlogDirDataset(synthetic_shard_dir)
        if not synth.samples:
            print(f"no samples in {synthetic_shard_dir} - nothing to eval")
            return model
        split = max(1, int(len(synth.samples) * synthetic_train_frac))
        held = synth.samples[split:]
        if not held:
            print("synthetic shard too small to hold anything out - skipping synthetic curve")
            return model
        _report("synthetic held-out", model, collate(held))
        _report_mechanics("synthetic held-out mechanic scores (derived tags)", mechanic_accuracy(model, held))

    return model


if __name__ == "__main__":
    evaluate()
