"""Losses, weighting and the training loop - md/ALPHAZERO.md Part 5.

    l = w_v . BCE(v, z) + w_p . CE(p, pi) + w_B . BCE(B, b) + w_D . CE(D, d) + weight decay

**BCE for value, not MSE.** AlphaZero's `z` is -1/0/+1 so squared error is natural; ours is a
probability in [0,1], where BCE is the proper scoring rule - and it accepts a soft target
unchanged, so making the reward graded later costs nothing here.

**Point 13's weighting crisis evaporates**, and it is worth seeing why rather than trusting it:
the policy is *one* cross-entropy over a 1,248-way softmax, not 1,248 independent predictions.
Averaged within each answer, all four terms are one number each, and the weights are an ordinary
tuning question rather than a structural imbalance.

**Point 14 still applies to the value head alone.** `pi` is a distribution summing to 1, so
cross-entropy against it is balanced by construction; `z` is not, so the positive term is scaled
by the rate actually observed in the batch rather than by a guessed constant.

Stage 0 supervises `p` on root states and `v` on terminals. It does **not** supervise `v` at a
non-terminal, because Part 5 shows that would be labelling the wrong question - `A(s1)=0` while
`v(s1)=0.58` for the same state - and no cheap label for the right one exists before search.
`B` and `D` are wired through the loss and left unsupervised here: their labels need a record
per candidate, which is 207,935 sim-logs for a signal AlphaZero does not have at all. The heads
exist so that turning them on later is a config change, not a shape change.

Usage:
    py -m rlgym.train --steps 400 --out data/runs/stage0
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlgym.baselines import HELD_OUT, evaluate, fit_entry_rates, entry_scores
from rlgym.config import Config
from rlgym.dataset import MachineData, load_corpus, noted_graph
from rlgym.graph import FEATURE_VERSION
from rlgym.metrics import Metrics
from rlgym.net import Batch, Net, Outputs, make_batch


@dataclass
class Targets:
    """What each graph in a batch is supervised on. `None` means "not this one"."""

    policy: list[np.ndarray | None]  # per machine, [A] summing to 1, or None
    value: torch.Tensor | None  # [M] in 0..1, NaN where unsupervised
    blocks: torch.Tensor | None = None  # [C]
    reason: torch.Tensor | None = None  # [M] class ids, -1 where unsupervised
    # Per-example policy weight. This is where `root_weight` and `policy_age_decay` land:
    # `pi` is an opinion and ages, and at k=2 half of all policy examples are the same root.
    weights: list[float] | None = None


def policy_loss(out: Outputs, batch: Batch, legal: torch.Tensor, targets: Targets):
    """Cross-entropy against `pi`, averaged **within** a machine before across machines.

    The per-machine average is what point 16 demands: without it a 3,000-action machine
    contributes 2.4x the gradient of a 1,250-action one purely because of its size, so a
    machine's weight in the loss would depend on who its batchmates were.
    """
    log_p = out.log_policy(legal)
    counts = batch.action_counts()
    pieces = torch.split(log_p, counts)
    terms = []
    weights = []
    for index, target in enumerate(targets.policy):
        if target is None:
            continue
        pi = torch.from_numpy(np.ascontiguousarray(target)).float()
        terms.append(-(pi * pieces[index]).sum())
        weights.append(1.0 if targets.weights is None else float(targets.weights[index]))
    if not terms:
        return torch.zeros((), requires_grad=True), 0
    stacked = torch.stack(terms)
    w = torch.tensor(weights, dtype=stacked.dtype)
    total = float(w.sum())
    if total <= 0:
        return torch.zeros((), requires_grad=True), 0
    return (stacked * w).sum() / total, len(terms)


def value_loss(out: Outputs, targets: Targets, pos_weight: float):
    """BCE, with the positive term scaled by the observed rate (point 14).

    `pos_weight <= 0` means "compute it from this batch", which is the honest default: the
    working rate is 43.6% corpus-wide but varies per machine, and a constant fitted once would
    drift silently as the library grows.
    """
    if targets.value is None:
        return torch.zeros(()), 0
    z = targets.value
    supervised = torch.isfinite(z)
    if not bool(supervised.any()):
        return torch.zeros(()), 0
    v = out.value[supervised].clamp(1e-6, 1 - 1e-6)
    z = z[supervised]
    if pos_weight <= 0:
        positives = float(z.sum())
        pos_weight = (len(z) - positives) / max(positives, 1.0)
    loss = -(pos_weight * z * v.log() + (1 - z) * (1 - v).log())
    return loss.mean(), int(supervised.sum())


def auxiliary_loss(out: Outputs, targets: Targets):
    """`B` and `D`. Unsupervised in Stage 0; the terms exist so enabling them is a config edit."""
    blocks = torch.zeros(())
    reason = torch.zeros(())
    if targets.blocks is not None:
        blocks = F.binary_cross_entropy_with_logits(out.blocks, targets.blocks)
    if targets.reason is not None:
        mask = targets.reason >= 0
        if bool(mask.any()):
            reason = F.cross_entropy(out.reason[mask], targets.reason[mask])
    return blocks, reason


def total_loss(out, batch, legal, targets, config) -> tuple[torch.Tensor, dict]:
    p, n_policy = policy_loss(out, batch, legal, targets)
    v, n_value = value_loss(out, targets, config.value_pos_weight)
    b, d = auxiliary_loss(out, targets)
    loss = (
        config.w_policy * p
        + config.w_value * v
        + config.w_blocks * b
        + config.w_reason * d
    )
    return loss, {
        "loss": float(loss.detach()),
        "loss_policy": float(p.detach()),
        "loss_value": float(v.detach()),
        "n_policy": n_policy,
        "n_value": n_value,
    }


# --- assembling one step ------------------------------------------------------------------


def build_step(
    corpus: list[MachineData], rng: random.Random, config, step: int = 0
) -> tuple[Batch, torch.Tensor, Targets]:
    """One step's batch. **Policy steps and value steps alternate.**

    A noted terminal is a full machine graph, and machines span 793 to 22,797 items - a 29x
    spread. Mixing roots and terminals in one batch means the largest machine can appear twice,
    which is 45,000 items, and an unbounded batch does not raise: the process segfaults.

    Alternating instead bounds every batch by the *largest single graph* rather than by twice
    it, and it removes the bias the mixed form had - where a machine bigger than the budget ate
    the whole step with its root and never contributed a value example, while every counter
    still looked healthy.

        even step   roots only      -> the dense policy target, the only dense signal there is
        odd step    terminals only  -> the value targets

    Each step therefore carries one loss term. That is ordinary multi-task alternation, and it
    is what `n_policy` and `n_value` in the metrics row report per step.
    """
    value_step = step % 2 == 1
    chosen = rng.sample(corpus, min(config.batch_machines, len(corpus)))
    graphs: list = []
    legal_parts: list = []
    policy: list[np.ndarray | None] = []
    values: list[float] = []
    used = 0

    def fits(data: MachineData) -> bool:
        # The first graph is always admitted, so a machine larger than the whole budget still
        # trains. Dropping it silently would restrict training to small machines.
        return not graphs or used + data.graph.n_items <= config.max_items_per_step

    def add(data: MachineData, graph, target, value: float) -> None:
        nonlocal used
        graphs.append(graph)
        legal_parts.append(data.legal)
        policy.append(target)
        values.append(value)
        used += data.graph.n_items

    if not value_step:
        for data in chosen:
            if not fits(data):
                break
            # Part 5: `v` at a non-terminal has no Stage 0 label, and inventing one would label
            # the wrong question. NaN means "not supervised" and value_loss drops it.
            target = data.policy_target if data.has_policy_target else None
            add(data, data.graph, target, math.nan)
    else:
        # Round-robin, so no machine is starved by whoever happens to be listed first.
        machines = [data for data in chosen if len(data.labelled_actions())]
        queues = [
            rng.choices(
                [int(a) for a in data.labelled_actions()], k=config.value_states_per_machine
            )
            for data in machines
        ]
        for round_index in range(config.value_states_per_machine):
            if not any(fits(data) for data in machines):
                break
            for data, queue in zip(machines, queues):
                if not fits(data):
                    continue
                action = queue[round_index]
                add(data, noted_graph(data, action), None, float(data.rewards[action]))

    if not graphs:  # every chosen machine had no labelled action at all
        return build_step(corpus, rng, config, step=0)

    batch = make_batch(graphs)
    # A terminal state has no legal action of its own; its policy row is never read, and the
    # finite masking constant keeps an all-masked softmax from producing NaN. Reusing the base
    # machine's mask keeps every segment the right length.
    legal = torch.from_numpy(np.concatenate(legal_parts))
    if legal.shape[0] != sum(batch.action_counts()):
        raise AssertionError("legal mask length disagrees with the batch's action count")
    return batch, legal, Targets(policy=policy, value=torch.tensor(values, dtype=torch.float32))


# --- evaluation ---------------------------------------------------------------------------


@torch.no_grad()
def model_scores(net: Net, data: MachineData) -> dict[int, float]:
    """The model's prior over one machine's actions, in `baselines.Scores` form.

    Deliberately the same shape the four baselines produce, so the model goes through the exact
    same `evaluate` and the ladder is a like-for-like comparison rather than two measurements
    that merely sound similar.
    """
    net.eval()
    prior, _ = net.priors(data.graph, list(data.legal))
    return {int(action): float(prior[action]) for action in data.labelled_actions()}


@torch.no_grad()
def value_calibration(net: Net, data: MachineData, rng: random.Random, n: int = 128, bins: int = 5):
    """Point 33's table: bucket `v` and compare each bucket's mean prediction to its outcome.

    The Stage 0 gate is that these two columns track each other. A model that ranks perfectly
    but predicts 0.02 everywhere passes an AUC check and fails this one, which is exactly the
    failure the ladder cannot see.
    """
    actions = data.labelled_actions()
    if not len(actions):
        return []
    sample = rng.sample(list(actions), min(n, len(actions)))
    graphs = [noted_graph(data, int(a)) for a in sample]
    net.eval()
    predictions = []
    # Chunked by items, not by count, for the same reason build_step is - 16 copies of a
    # 12,000-item machine is 192,000 items in one batch, which does not fit.
    per_chunk = max(1, 8000 // max(1, data.graph.n_items))
    for start in range(0, len(graphs), per_chunk):
        chunk = graphs[start : start + per_chunk]
        predictions.extend(net(make_batch(chunk)).value.tolist())
    outcomes = [float(data.rewards[a]) for a in sample]
    table = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        picked = [
            (p, o)
            for p, o in zip(predictions, outcomes)
            if (low <= p < high) or (index == bins - 1 and p >= high)
        ]
        if picked:
            table.append(
                {
                    "bin": f"{low:.1f}-{high:.1f}",
                    "n": len(picked),
                    "predicted": round(sum(p for p, _ in picked) / len(picked), 3),
                    "actual": round(sum(o for _, o in picked) / len(picked), 3),
                }
            )
    return table


def evaluate_split(net, corpus: dict[str, MachineData], names, budget: int, seed: int = 0):
    aucs, precisions, non_cargo = [], [], 0
    for name in names:
        data = corpus[name]
        result = evaluate("model", data.labelset, model_scores(net, data), budget, seed)
        aucs.append(result.auc)
        precisions.append(result.precision)
        non_cargo += result.non_cargo_at_budget
    if not aucs:
        return {"auc": 0.0, "precision": 0.0, "non_cargo": 0}
    return {
        "auc": sum(aucs) / len(aucs),
        "precision": sum(precisions) / len(precisions),
        "non_cargo": non_cargo,
    }


def baseline_reference(corpus: dict[str, MachineData], training, held, budget: int):
    """Baseline 1 on the same split, so every model number has its bar printed beside it."""
    rates = fit_entry_rates(corpus[name].labelset for name in training)
    out = {}
    for label, names in (("train", training), ("held", held)):
        aucs, precisions = [], []
        for name in names:
            data = corpus[name]
            result = evaluate(
                "1 per-entry", data.labelset, entry_scores(data.labelset, rates), budget
            )
            aucs.append(result.auc)
            precisions.append(result.precision)
        out[label] = {
            "auc": sum(aucs) / max(1, len(aucs)),
            "precision": sum(precisions) / max(1, len(precisions)),
        }
    return out


# --- checkpoints --------------------------------------------------------------------------


def save_checkpoint(path: Path, net: Net, config: Config, step: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_version": FEATURE_VERSION,
            "net_config": vars(net.config),
            "config": config.to_json(),
            "step": step,
            "best_held_auc": best,
            "state_dict": net.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[Net, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint was trained on features {payload['feature_version']!r} but graph.py "
            f"now produces {FEATURE_VERSION!r}. The scalar columns have moved; retrain."
        )
    from rlgym.config import NetConfig

    net = Net(NetConfig(**payload["net_config"]))
    net.load_state_dict(payload["state_dict"])
    return net, payload


# --- the loop -----------------------------------------------------------------------------


def train(config: Config, out_dir: Path, corpus=None, quiet: bool = False) -> dict:
    torch.manual_seed(config.train.seed)
    rng = random.Random(config.train.seed)

    if corpus is None:
        corpus = load_corpus(
            config.labels_dir,
            radius=config.radius,
            cache_dir=config.graphs_dir,
            tick_cap=config.tick_cap,
            reward_cargo=config.train.reward_cargo,
            verbose=not quiet,
        )
    held = [name for name in corpus if name in HELD_OUT]
    training = [name for name in corpus if name not in HELD_OUT]
    if not training:
        raise SystemExit("no training machines - the whole corpus is in HELD_OUT")

    # Evaluation forwards a whole machine per call, so the training side is sampled while the
    # held-out side never is. Held-out is the number that decides anything.
    eval_train = training[:: max(1, len(training) // 6)][:6]
    train_data = [corpus[name] for name in training]

    net = Net(config.net)
    optimiser = torch.optim.AdamW(
        net.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    metrics = Metrics(
        out_dir / "metrics.jsonl", {"config": config.to_json(), "params": net.n_parameters()}
    )
    bars = baseline_reference(corpus, training, held, config.train.eval_budget)
    metrics.log("baseline", **bars)

    if not quiet:
        print(f"machines: {len(training)} training, {len(held)} held out")
        print(f"parameters: {net.n_parameters():,}")
        print(
            f"baseline 1  train auc {bars['train']['auc']:.3f}   "
            f"held auc {bars['held']['auc']:.3f}  <- the bar\n"
        )

    best = -1.0
    # Steps alternate between policy and value, so the term the step did not train is exactly
    # zero. Carrying the last real value of each keeps an eval row from reading as "the policy
    # loss collapsed to zero" on every other row, which is a reporting artefact and not a fact.
    recent = {"loss_policy": float("nan"), "loss_value": float("nan")}
    started = time.perf_counter()
    for step in range(1, config.train.steps + 1):
        net.train()
        batch, legal, targets = build_step(train_data, rng, config.train, step)
        out = net(batch)
        loss, parts = total_loss(out, batch, legal, targets, config.train)

        lr = _schedule(step, config.train)
        for group in optimiser.param_groups:
            group["lr"] = lr
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.train.grad_clip)
        optimiser.step()
        if parts["n_policy"]:
            recent["loss_policy"] = parts["loss_policy"]
        if parts["n_value"]:
            recent["loss_value"] = parts["loss_value"]
        metrics.log("step", step=step, lr=lr, **parts)

        if step % config.train.eval_every == 0 or step == config.train.steps:
            on_train = evaluate_split(net, corpus, eval_train, config.train.eval_budget)
            on_held = evaluate_split(net, corpus, held, config.train.eval_budget)
            calibration = (
                value_calibration(net, corpus[held[0]], rng) if held else []
            )
            metrics.log(
                "eval",
                step=step,
                train=on_train,
                held=on_held,
                calibration=calibration,
                **recent,
                n_policy=parts["n_policy"],
                n_value=parts["n_value"],
            )
            if not quiet:
                print(
                    f"step {step:>5}  p {recent['loss_policy']:.4f}  "
                    f"v {recent['loss_value']:.4f}   "
                    f"train auc {on_train['auc']:.3f}   held auc {on_held['auc']:.3f}"
                )
            save_checkpoint(out_dir / "last.pt", net, config, step, best)
            # Point 18: keep going past the first downturn and remember the best point. The
            # held-out measure is noisy at 7 machines and a dip is usually a wobble.
            if on_held["auc"] > best:
                best = on_held["auc"]
                save_checkpoint(out_dir / "best.pt", net, config, step, best)

    wall = time.perf_counter() - started
    final = {
        "steps": config.train.steps,
        "wall_seconds": round(wall, 1),
        "best_held_auc": best,
        "baseline_held_auc": bars["held"]["auc"],
        "beats_baseline_1": best > bars["held"]["auc"],
        "parameters": net.n_parameters(),
    }
    metrics.log("final", **final)
    if not quiet:
        print(f"\n{'MILESTONE 2':<14}{'model':>10}{'baseline 1':>12}")
        print(f"{'held-out AUC':<14}{best:>10.3f}{bars['held']['auc']:>12.3f}")
        print(
            "\n"
            + (
                "the model beats the only baseline that transfers."
                if final["beats_baseline_1"]
                else "the model does NOT beat baseline 1 yet."
            )
        )
        print(f"wrote {out_dir}")
    return final


def _schedule(step: int, config) -> float:
    """Linear warmup then cosine decay - AdamW's usual pairing, and the warmup matters here
    because the first steps of a tied trunk are the least stable ones."""
    if step <= config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--reward-cargo", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/runs/stage0"))
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.steps is not None:
        config.train.steps = args.steps
    if args.reward_cargo is not None:
        config.train.reward_cargo = args.reward_cargo
    train(config, args.out)


if __name__ == "__main__":
    main()
