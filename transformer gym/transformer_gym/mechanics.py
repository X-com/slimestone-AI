"""Per-mechanic tagging and scoring - the "which mechanics has the model actually learned" signal
the training-paradigm design calls for, and the annealed curriculum's stage ordering.

Tags are derived automatically from fields an encoded Sample already carries (y_event_grid,
stickiness, y_moves) - no extra simlog decoding needed, and every sample (real fixture or
generated corpus shard) gets tagged for free. This is deliberately coarse (a board can carry
several tags at once); precise ground truth instead comes from mechanic_fixtures.py's small
hand-built, hand-tagged suite, which mechanic_fixtures.verify_suite() cross-checks against these
same derived tags.
"""
from __future__ import annotations

import torch

from .encode import Sample
from .simlog_reader import KIND_NAMES

_KIND_TO_IDX = {name: idx for idx, name in KIND_NAMES.items()}
_PUSH_KINDS = ("PistonQueued", "PistonMoveExecuted", "BlockPushed", "PistonNeighborNotified")
_OBSERVER_KINDS = ("ObserverFired", "ObserverActivated", "ObserverSuppressed")

# Matches sim_event_log.h's STICKINESS_* constants (verify_simulation_data.STICKINESS_NAMES).
STICKINESS_ALL = 1
STICKINESS_NEVER = 3

# Ordered by causal dependency (design doc "Mechanics are ordered by causal dependency") - this
# order is what the annealed schedule in train.py ramps across, index 0 -> peaks earliest.
MECHANIC_ORDER: tuple[str, ...] = ("push_order", "sticky_drag", "non_stick", "observer_pulse", "composed")


def _any_kind(sample: Sample, kind_names: tuple[str, ...]) -> bool:
    idxs = [_KIND_TO_IDX[k] for k in kind_names]
    return bool(sample.y_event_grid[:, idxs].any())


def derive_tags(sample: Sample) -> set[str]:
    """Coarse, fully automatic mechanic tags for one sample."""
    tags: set[str] = set()
    if _any_kind(sample, _PUSH_KINDS):
        tags.add("push_order")
    if sample.stickiness.eq(STICKINESS_ALL).any() and sample.y_moves.gt(0).any():
        tags.add("sticky_drag")
    if sample.stickiness.eq(STICKINESS_NEVER).any():
        tags.add("non_stick")
    if _any_kind(sample, _OBSERVER_KINDS):
        tags.add("observer_pulse")
    core = tags & {"push_order", "sticky_drag", "non_stick", "observer_pulse"}
    if len(core) >= 2:
        tags.add("composed")
    return tags


def binary_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = (logits > 0).float()
    correct = ((pred == target).float() * mask).sum()
    return (correct / mask.sum().clamp(min=1)).item()


def mechanic_accuracy(model: torch.nn.Module, samples: list[Sample]) -> dict[str, float]:
    """Per-mechanic accuracy (average of the moves/event_grid head accuracies, the two heads
    tied directly to whether a mechanic's physical outcome was predicted correctly), scored on
    whichever samples are passed in - either the corpus (derived-tag slicing, coarse but free)
    or mechanic_fixtures.py's hand-tagged suite (precise ground truth)."""
    from .dataset import collate  # local import: avoids a dataset<->mechanics import cycle

    buckets: dict[str, list[Sample]] = {m: [] for m in MECHANIC_ORDER}
    for s in samples:
        for tag in derive_tags(s):
            buckets[tag].append(s)

    scores: dict[str, float] = {}
    with torch.no_grad():
        for tag, bucket in buckets.items():
            if not bucket:
                continue
            batch = collate(bucket)
            out = model(batch)
            mask = batch["mask"]
            moves_acc = binary_accuracy(out["moves"], batch["y_moves"], mask)
            grid_mask = mask.unsqueeze(-1).expand_as(out["event_grid"])
            grid_acc = binary_accuracy(out["event_grid"], batch["y_event_grid"], grid_mask)
            scores[tag] = (moves_acc + grid_acc) / 2
    return scores
