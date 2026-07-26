"""Curriculum training driver: movement -> event grid -> structure (easiest signal first, per the
transformer-gym plan). The `cause` head (pointer-over-tokens, hardest) is not implemented yet -
add it as a fourth phase once the first three are proven out on real (post-generator) data.

Whole ~54-fixture corpus fits in one batch on CPU, so this is a plain full-batch loop, not a
DataLoader/epoch machine - add one only once the corpus is big enough that it matters.
"""
from __future__ import annotations

import torch
from torch import nn

from .dataset import SimlogDataset, collate
from .model import PhysicsTransformer

PHASES = [
    ("movement", ["moves", "stays_attached"], 150),
    ("event_grid", ["moves", "stays_attached", "event_grid"], 150),
    ("structure", ["moves", "stays_attached", "event_grid", "structure"], 150),
]


def _losses(out: dict, batch: dict) -> dict:
    mask = batch["mask"]
    node_count = mask.sum().clamp(min=1)

    def masked_bce(pred, target):
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return (loss * mask).sum() / node_count

    grid_loss = nn.functional.binary_cross_entropy_with_logits(
        out["event_grid"], batch["y_event_grid"], reduction="none"
    )
    grid_loss = (grid_loss * mask.unsqueeze(-1)).sum() / (node_count * out["event_grid"].shape[-1])

    structure_loss = (
        nn.functional.binary_cross_entropy_with_logits(out["valid_cycle"], batch["y_valid_cycle"])
        + nn.functional.cross_entropy(out["termination"], batch["y_termination"])
        + nn.functional.mse_loss(out["net_shift"], batch["y_net_shift"])
    )

    return {
        "moves": masked_bce(out["moves"], batch["y_moves"]),
        "stays_attached": masked_bce(out["stays_attached"], batch["y_stays_attached"]),
        "event_grid": grid_loss,
        "structure": structure_loss,
    }


def train(names: list[str] | None = None, lr: float = 3e-4) -> PhysicsTransformer:
    ds = SimlogDataset(names)
    if not ds.samples:
        raise ValueError("no fixtures small enough to encode (see dataset.MAX_NODES)")
    batch = collate(ds.samples)
    model = PhysicsTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for phase_name, active, n_steps in PHASES:
        for step in range(n_steps):
            out = model(batch)
            losses = _losses(out, batch)
            loss = sum(losses[k] for k in active)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == n_steps - 1:
                detail = " ".join(f"{k}={losses[k].item():.4f}" for k in active)
                print(f"[{phase_name}] step {step}/{n_steps} loss={loss.item():.4f} ({detail})")

    return model


if __name__ == "__main__":
    train()
