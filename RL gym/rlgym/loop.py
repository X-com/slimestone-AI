"""The outer loop - md/ALPHAZERO.md Part 6. Stage 1: MCTS with ground-truth leaves.

    episode   k placements on one base machine       ~200 simulator calls
    round     pick a machine, run N episodes         minutes
    lifetime  library grows, model improves          days

**The structural weakness this file is built around.** AlphaZero's curriculum is self-balancing
because the opponent is always exactly your strength. We have no opponent, so the model's biases
shape its own future training data with nothing outside the loop to correct them:

    model chooses what to try  ->  what works enters the library
            ^                                    |
            |                                    v
       model trains on  <---------------  the library IS the training data

Glass is inert, pushable and safe, so the model learns glass is low-risk, proposes it often, it
often works, the library fills with glass, and it stops proposing observers - forever. **This
looks healthy from every angle**: high success rate, growing library, falling losses.

The 5% uninformed share is the only defence, and it has three jobs at once - the measurement
baseline (point 32), the unbiased calibration sample (point 33), and **the only unbiased
training data the system produces**. It must stay genuinely uninformed: the model-sampled share
still uses model scores and cannot double as the control.

Usage:
    py -m rlgym.loop --rounds 5 --out data/runs/stage1 [--checkpoint data/runs/stage0/best.pt]
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlgym.config import Config
from rlgym.dataset import machine_for
from rlgym.game import Candidate, GameState, Machine, Placement, canonical_hash, decode_action
from rlgym.graph import GraphTooLarge, apply_notes, build
from rlgym.labeller import corpus_names
from rlgym.metrics import Metrics
from rlgym.net import Net, make_batch
from rlgym.search import Search, net_evaluator, uniform_evaluator
from rlgym.sim import SimConfig, SimulatorProcess
from rlgym.simlog import record_for
from rlgym.store import Attempt, Entry, Store
from rlgym.train import Targets, load_checkpoint, save_checkpoint, total_loss

SOURCES = ("top", "sampled", "uninformed")


@dataclass
class Example:
    """One training example. `z` never ages; `pi` does (Part 5's distinction)."""

    machine_key: str
    placements: tuple[Placement, ...]
    pi: np.ndarray | None
    z: float
    round_index: int
    is_root: bool


@dataclass
class Replay:
    """Point 19's "keep everything", corrected: three of the four label types never age.

    `z`, `B` and `D` are simulator facts - a modification that worked in round 5 works in round
    5,000. `pi` is an opinion from an older, weaker search, and is the only one AlphaZero's
    replay window exists to expire. Decay is a knob, default flat, so it is measured rather than
    assumed.
    """

    examples: list[Example] = field(default_factory=list)

    def add(self, example: Example) -> None:
        self.examples.append(example)

    def policy_weight(self, example: Example, now: int, decay: float, root_weight: float) -> float:
        weight = root_weight if example.is_root else 1.0
        if decay > 0:
            weight *= (1.0 - decay) ** max(0, now - example.round_index)
        return weight

    @staticmethod
    def sample_from(pool: list[Example], rng: random.Random, n: int) -> list[Example]:
        if len(pool) <= n:
            return list(pool)
        return rng.sample(pool, n)


class Loop:
    def __init__(self, config: Config, out_dir: Path, net: Net | None = None):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.out_dir / "library")
        self.replay = Replay()
        self.rng = random.Random(config.loop.seed)
        self.net = net if net is not None else Net(config.net)
        self.optimiser = torch.optim.AdamW(
            self.net.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.metrics = Metrics(self.out_dir / "metrics.jsonl", {"config": config.to_json()})
        self.machines: dict[str, Machine] = {}
        self.graphs: dict[str, Any] = {}
        self.verdicts: dict[str, dict] = {}
        self.transposition: dict[str, float] = {}

    # --- machines ---------------------------------------------------------------------

    def seed_library(self, names: list[str], sim: SimulatorProcess) -> None:
        for name in names:
            try:
                machine = machine_for(name, radius=self.config.radius)
            except FileNotFoundError:
                continue
            candidate = machine.to_candidate(cid=0)
            verdict = sim.simulate(candidate)
            if not verdict.get("validCycle"):
                continue
            shift = verdict.get("finalShift") or {"x": 0, "y": 0, "z": 0}
            entry = self.store.seed(
                name, candidate, int(verdict.get("period", 0)), (shift["x"], shift["y"], shift["z"])
            )
            self.machines[entry.digest] = machine
        self.store.save()

    def machine_of(self, entry: Entry) -> Machine:
        if entry.digest not in self.machines:
            self.machines[entry.digest] = Machine.from_candidate(
                entry.candidate, radius=self.config.radius
            )
        return self.machines[entry.digest]

    def graph_of(self, entry: Entry):
        """One build per machine, then every candidate on it is a note patch."""
        if entry.digest not in self.graphs:
            machine = self.machine_of(entry)
            record = record_for(machine.to_candidate(cid=0))
            self.graphs[entry.digest] = build(machine, record, tick_cap=self.config.tick_cap)
        return self.graphs[entry.digest]

    def pick_base(self) -> Entry:
        """Fewest descendants first, jittered.

        Directly against the self-selection failure above: a machine the loop has already mined
        heavily is the one whose children are most likely to be near-siblings, and near-siblings
        grow library *count* without growing library *variety*.
        """
        entries = list(self.store)
        weights = [1.0 / (1.0 + entry.descendants) for entry in entries]
        return self.rng.choices(entries, weights=weights, k=1)[0]

    # --- one episode ------------------------------------------------------------------

    def episode(
        self, entry: Entry, source: str, round_index: int, sim: SimulatorProcess
    ) -> dict[str, Any]:
        machine = self.machine_of(entry)
        graph = self.graph_of(entry)
        evaluate = (
            uniform_evaluator(machine)
            if source == "uninformed"
            else net_evaluator(self.net, graph)
        )

        def oracle(candidate: Candidate) -> float:
            verdict = sim.simulate(candidate)
            self.verdicts[canonical_hash(candidate)] = verdict
            return 1.0 if verdict.get("validCycle") else 0.0

        search = Search(
            machine,
            graph,
            evaluate,
            oracle,
            config=self.config.search,
            rng=self.rng,
            cache=self.transposition,
        )
        per_move = max(1, self.config.search.simulations // max(1, self.config.search.k))

        state = GameState(machine, k=self.config.search.k)
        root = search.expand(state, add_noise=True)
        played: list[Placement] = []
        for depth in range(self.config.search.k):
            result = search.run(root, budget=per_move)
            if result.pi.sum() <= 0:
                break
            # Point 20 refined: the exploration knob sits on visit counts, not raw scores -
            # strictly better information. `top` takes the argmax; `sampled` samples at tau.
            action = (
                int(np.argmax(result.pi))
                if source == "top"
                else int(self.rng.choices(range(len(result.pi)), weights=result.pi.tolist(), k=1)[0])
            )
            self.replay.add(
                Example(
                    machine_key=entry.digest,
                    placements=tuple(played),
                    pi=result.pi.copy(),
                    z=math.nan,
                    round_index=round_index,
                    is_root=(depth == 0),
                )
            )
            played.append(decode_action(action, machine.cell_list))
            child = search.reuse(root, action)
            state = state.step(played[-1])
            root = child if child is not None else search.expand(state)

        return self._harvest(entry, search, source, round_index, played)

    def _harvest(self, entry, search, source, round_index, played) -> dict[str, Any]:
        """Everything the search paid for, recorded.

        A search spends its whole budget and ends on one move; the other outcomes are true
        labels that cost nothing extra. All working non-cargo ones enter the library - nothing
        paid for is discarded.
        """
        working = non_cargo = 0
        machine = self.machine_of(entry)
        for first_action, candidate, reward in search.attempts:
            digest = canonical_hash(candidate)
            verdict = self.verdicts.get(digest, {})
            period = int(verdict.get("period", 0))
            raw = verdict.get("finalShift") or {"x": 0, "y": 0, "z": 0}
            shift = (raw["x"], raw["y"], raw["z"])
            cargo = reward > 0 and period == entry.period and shift == tuple(entry.shift)
            self.store.record(
                Attempt(
                    digest=digest,
                    parent=entry.digest,
                    round_index=round_index,
                    source=source,
                    reward=reward,
                    working=reward > 0,
                    cargo=bool(cargo),
                    period=period,
                    shift=shift,
                    blocks=len(candidate["blocks"]),
                )
            )
            if reward > 0:
                working += 1
                if not cargo:
                    non_cargo += 1
                    self.store.admit(
                        candidate,
                        entry.digest,
                        round_index,
                        period,
                        shift,
                        cargo=False,
                        allow_cargo=self.config.loop.cargo_may_enter_library,
                    )
            # Every terminal the search simulated is a value example - which is why `v` sees
            # ~100x more data than `p` and will look like it is learning while `p` plateaus.
            # That is the data rates, not a broken policy head.
            self.replay.add(
                Example(
                    machine_key=entry.digest,
                    placements=self._placements_of(machine, candidate, entry),
                    pi=None,
                    z=reward,
                    round_index=round_index,
                    is_root=False,
                )
            )
        return {
            "source": source,
            "calls": search.calls,
            "working": working,
            "non_cargo": non_cargo,
            "value_predictions": search.value_predictions,
        }

    @staticmethod
    def _placements_of(machine: Machine, candidate: Candidate, entry: Entry):
        """Recover the placements from a finished candidate by differencing against the base.

        Cheaper and more robust than threading them through the search: the search's job is to
        find candidates, and a candidate plus its base determines its placements exactly.
        """
        from rlgym.blocks import PALETTE

        base = machine.cells
        now = {(b["x"], b["y"], b["z"]): b["state"] for b in candidate["blocks"]}
        out = []
        for cell, state in now.items():
            if base.get(cell) == state:
                continue
            for slot, palette_entry in enumerate(PALETTE):
                if palette_entry is not None and palette_entry.state == state:
                    out.append(Placement(cell, slot))
                    break
        # A cell the base had and the candidate does not was set to air. Differencing only the
        # blocks that are present would miss every removal, and removals are a third of the
        # palette's effect - the example would carry no note at all and train `v` on the
        # unmodified machine while claiming the modified machine's reward.
        air = next(
            slot
            for slot, palette_entry in enumerate(PALETTE)
            if palette_entry is not None and palette_entry.is_air
        )
        out += [Placement(cell, air) for cell in base if cell not in now]
        return tuple(out)

    # --- training on what the loop produced -------------------------------------------

    def train_round(self, round_index: int, steps: int) -> dict[str, Any]:
        if not self.replay.examples:
            return {"steps": 0}
        last = {}
        for _ in range(steps):
            batch, legal, targets = self._batch(round_index)
            if batch is None:
                break
            self.net.train()
            out = self.net(batch)
            loss, parts = total_loss(out, batch, legal, targets, self.config.train)
            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.train.grad_clip)
            self.optimiser.step()
            last = parts
        return last

    def _batch(self, round_index: int):
        policy_examples = [e for e in self.replay.examples if e.pi is not None]
        value_examples = [e for e in self.replay.examples if e.pi is None]
        chosen: list[Example] = []
        chosen += self.replay.sample_from(policy_examples, self.rng, self.config.train.batch_machines)
        chosen += self.replay.sample_from(
            value_examples, self.rng, self.config.train.value_states_per_machine
        )
        if not chosen:
            return None, None, None

        graphs, legal_parts, policy, values, weights = [], [], [], [], []
        used = 0
        for example in chosen:
            entry = self.store.entries.get(example.machine_key)
            if entry is None:
                continue
            base = self.graphs.get(example.machine_key)
            if base is None:
                continue
            if used + base.n_items > self.config.train.max_items_per_step and graphs:
                break
            machine = self.machine_of(entry)
            state = GameState(machine, example.placements, k=self.config.search.k)
            graphs.append(apply_notes(base, example.placements))
            legal_parts.append(np.asarray(GameState(machine, (), k=1).legal_mask(), dtype=bool))
            policy.append(example.pi)
            values.append(example.z if example.pi is None else math.nan)
            weights.append(
                self.replay.policy_weight(
                    example,
                    round_index,
                    self.config.train.policy_age_decay,
                    self.config.train.root_weight,
                )
            )
            used += base.n_items
        if not graphs:
            return None, None, None
        batch = make_batch(graphs)
        legal = torch.from_numpy(np.concatenate(legal_parts))
        targets = Targets(
            policy=policy,
            value=torch.tensor(values, dtype=torch.float32),
            weights=weights,
        )
        return batch, legal, targets

    # --- the round --------------------------------------------------------------------

    def run(self, names: list[str] | None = None) -> dict[str, Any]:
        loop = self.config.loop
        shares = loop.shares()
        plan = self._share_plan(loop.episodes_per_round, shares)

        with SimulatorProcess(SimConfig()) as sim:
            self.seed_library(names or corpus_names(max_blocks=60)[:6], sim)
            if not len(self.store):
                raise SystemExit("no seed machine has a valid cycle - nothing to build on")

            for round_index in range(1, loop.rounds + 1):
                started = time.perf_counter()
                totals = {source: {"calls": 0, "working": 0, "non_cargo": 0} for source in SOURCES}
                predictions: list[tuple[float, float]] = []
                for source in plan:
                    entry = self.pick_base()
                    try:
                        result = self.episode(entry, source, round_index, sim)
                    except GraphTooLarge:
                        continue
                    for key in ("calls", "working", "non_cargo"):
                        totals[source][key] += result[key]
                    predictions += result["value_predictions"]

                training = self.train_round(round_index, loop.train_steps_per_round)
                self.store.save()
                row = self.metrics.log(
                    "round",
                    round_index=round_index,
                    seconds=round(time.perf_counter() - started, 1),
                    per_source=totals,
                    headline=self._headline(totals),
                    library=self.store.variety(),
                    replay=len(self.replay.examples),
                    value_head=self._value_report(predictions),
                    **training,
                )
                self._print(row)
                save_checkpoint(
                    self.out_dir / "last.pt", self.net, self.config, round_index, 0.0
                )
        return self.metrics.of_kind("round")[-1] if self.metrics.of_kind("round") else {}

    def _share_plan(self, episodes: int, shares) -> list[str]:
        """Which share pays for each episode. The uninformed share is rounded **up** to at least
        one whenever any episodes run: at 5% of 8 episodes it would otherwise round to zero and
        the control group would silently disappear."""
        top, sampled, uninformed = shares
        plan = ["uninformed"] * max(1, round(episodes * uninformed))
        plan += ["top"] * max(0, round(episodes * top))
        while len(plan) < episodes:
            plan.append("sampled")
        return plan[:episodes]

    @staticmethod
    def _headline(totals) -> dict[str, float]:
        """**Non-cargo discoveries per 1,000 simulator calls**, model versus the control.

        Everything else in the row is diagnostic. At 0.348% non-cargo a few hundred calls
        contain about one by chance, so a single round's number is noise - it is logged per
        round so it can be pooled, not so it can be read alone.
        """
        out = {}
        for source, counts in totals.items():
            calls = counts["calls"]
            out[source] = round(1000 * counts["non_cargo"] / calls, 2) if calls else 0.0
        model_calls = totals["top"]["calls"] + totals["sampled"]["calls"]
        model_hits = totals["top"]["non_cargo"] + totals["sampled"]["non_cargo"]
        out["model"] = round(1000 * model_hits / model_calls, 2) if model_calls else 0.0
        return out

    @staticmethod
    def _value_report(predictions) -> dict[str, Any]:
        """`v` measured without being given authority - point 31 applied inside the tree.

        Its prediction at every node is logged against the exact terminal the search then found.
        In Stages 0-1 only its *ranking* matters; this is the evidence for whether it can be
        trusted in Stage 2, where its *value* is backed up the tree.
        """
        if not predictions:
            return {"n": 0}
        errors = [abs(p - z) for p, z in predictions]
        return {
            "n": len(predictions),
            "mean_abs_error": round(sum(errors) / len(errors), 4),
            "mean_predicted": round(sum(p for p, _ in predictions) / len(predictions), 4),
            "mean_actual": round(sum(z for _, z in predictions) / len(predictions), 4),
        }

    @staticmethod
    def _print(row) -> None:
        head = row["headline"]
        print(
            f"round {row['round_index']:>3}  {row['seconds']:>6.1f}s   "
            f"non-cargo/1k: model {head['model']:>6.2f}  control {head['uninformed']:>6.2f}   "
            f"library {row['library']['size']:>4}   replay {row['replay']:>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--machines", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/runs/stage1"))
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.rounds is not None:
        config.loop.rounds = args.rounds
    net = None
    if args.checkpoint is not None:
        net, payload = load_checkpoint(args.checkpoint)
        print(f"loaded {args.checkpoint} (step {payload['step']})")

    loop = Loop(config, args.out, net=net)
    print(
        f"budget split: {config.loop.share_top:.0%} top / {config.loop.share_sampled:.0%} "
        f"sampled / {config.loop.share_uninformed:.0%} uninformed\n"
    )
    loop.run(args.machines)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
