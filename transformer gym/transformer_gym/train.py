"""Curriculum training driver: two orthogonal curricula stacked on the same full-batch loop.

1. Loss-head curriculum (unchanged): movement -> event_grid -> structure, easiest signal first.
2. Mechanic curriculum (the training-paradigm design's "annealed data-mixture schedule"): rather
   than literally resampling which candidates get generated, each sample's contribution to the
   loss is weighted by how well its mechanic tags (mechanics.derive_tags) match the CURRENT
   training progress. Early on, push_order-tagged samples dominate; by the end, composed/observer
   samples do - but every mechanic keeps a floor weight throughout, so nothing is ever fully
   absent from training (this is what removes the need for a replay buffer: nothing is ever
   forgotten because nothing ever stops being shown, see mechanics.py's docstring).

Whole ~54-fixture corpus fits in one batch on CPU, so this is a plain full-batch loop, not a
DataLoader/epoch machine - add one only once the corpus is big enough that it matters.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .dataset import SimlogDataset, collate
from .mechanics import MECHANIC_ORDER, derive_tags, mechanic_accuracy
from .model import PhysicsTransformer

PHASES = [
    ("movement", ["moves", "stays_attached"], 150),
    ("event_grid", ["moves", "stays_attached", "event_grid"], 150),
    ("structure", ["moves", "stays_attached", "event_grid", "structure"], 150),
]
TOTAL_STEPS = sum(n for _, _, n in PHASES)

_STAGE_INDEX = {name: i for i, name in enumerate(MECHANIC_ORDER)}
_N_STAGES = len(MECHANIC_ORDER)
# Never fully absent - the floor is what makes forgetting a non-problem instead of something to
# mitigate with replay (design doc §1).
_FLOOR_WEIGHT = 0.15


def _mechanic_weight(mechanic: str, progress: float) -> float:
    """Triangular bump peaking at `mechanic`'s position in MECHANIC_ORDER, linearly interpolated
    across training progress in [0, 1]. Adjacent mechanics' bumps overlap exactly at the midpoint
    between them; only progress values a full stage-spacing away from a mechanic's peak reach
    the floor."""
    target = _STAGE_INDEX[mechanic] / (_N_STAGES - 1)
    dist = abs(progress - target)
    bump = max(0.0, 1.0 - dist * (_N_STAGES - 1))
    return _FLOOR_WEIGHT + (1.0 - _FLOOR_WEIGHT) * bump


def _sample_weights(tags_per_sample: list[set[str]], progress: float) -> torch.Tensor:
    weights = [
        max((_mechanic_weight(t, progress) for t in tags), default=1.0)
        for tags in tags_per_sample
    ]
    return torch.tensor(weights)


def _per_sample_losses(out: dict, batch: dict) -> dict[str, torch.Tensor]:
    """Same four loss terms as before, but each returned as a [B] per-sample vector instead of a
    scalar - needed so the mechanic curriculum can weight samples before averaging."""
    mask = batch["mask"]
    node_count = mask.sum(dim=-1).clamp(min=1)

    def masked_bce(pred, target):
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return (loss * mask).sum(dim=-1) / node_count

    grid_loss = nn.functional.binary_cross_entropy_with_logits(
        out["event_grid"], batch["y_event_grid"], reduction="none"
    )
    k = out["event_grid"].shape[-1]
    grid_loss = (grid_loss * mask.unsqueeze(-1)).sum(dim=(-2, -1)) / (node_count * k)

    structure_loss = (
        nn.functional.binary_cross_entropy_with_logits(out["valid_cycle"], batch["y_valid_cycle"], reduction="none")
        + nn.functional.cross_entropy(out["termination"], batch["y_termination"], reduction="none")
        + nn.functional.mse_loss(out["net_shift"], batch["y_net_shift"], reduction="none").mean(dim=-1)
    )

    return {
        "moves": masked_bce(out["moves"], batch["y_moves"]),
        "stays_attached": masked_bce(out["stays_attached"], batch["y_stays_attached"]),
        "event_grid": grid_loss,
        "structure": structure_loss,
    }


def train(
    names: list[str] | None = None,
    lr: float = 3e-4,
    progress_log: Path | None = None,
    log_every: int = 25,
) -> PhysicsTransformer:
    ds = SimlogDataset(names)
    if not ds.samples:
        raise ValueError("no fixtures small enough to encode (see dataset.MAX_NODES)")
    batch = collate(ds.samples)
    tags_per_sample = [derive_tags(s) for s in ds.samples]
    model = PhysicsTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        progress_log.write_text("")  # truncate: one JSONL file per training run, not appended-to forever

    global_step = 0
    for phase_name, active, n_steps in PHASES:
        for step in range(n_steps):
            progress = global_step / max(1, TOTAL_STEPS - 1)
            weights = _sample_weights(tags_per_sample, progress)

            out = model(batch)
            per_sample = _per_sample_losses(out, batch)
            total_per_sample = sum(per_sample[k] for k in active)
            loss = (total_per_sample * weights).sum() / weights.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 50 == 0 or step == n_steps - 1:
                detail = " ".join(f"{k}={per_sample[k].mean().item():.4f}" for k in active)
                print(f"[{phase_name}] step {step}/{n_steps} loss={loss.item():.4f} ({detail})")

            if progress_log is not None and (global_step % log_every == 0 or global_step == TOTAL_STEPS - 1):
                model.eval()
                scores = mechanic_accuracy(model, ds.samples)
                model.train()
                record = {
                    "step": global_step, "phase": phase_name, "progress": progress,
                    "loss": loss.item(), "mechanic_scores": scores,
                }
                with progress_log.open("a") as f:
                    f.write(json.dumps(record) + "\n")

            global_step += 1

    return model


if __name__ == "__main__":
    train()
