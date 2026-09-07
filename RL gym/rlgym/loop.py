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
from rlgym.function import added_cells, graded_reward, trim
from rlgym.game import (
    Candidate,
    GameState,
    Machine,
    Placement,
    blocks_to_map,
    canonical_hash,
    decode_action,
    is_stop,
)
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
# Summed per source into the round row. `grading_calls` is deliberately NOT part of `calls`:
# see the note in `episode`.
TALLIED = (
    "calls",
    "working",
    "load_bearing",
    "redundant_stripped",
    "grading_calls",
    "graded",
    "stopped",
    "depth",
    "episodes",
)


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
        self.records: dict[str, Any] = {}
        self.verdicts: dict[str, dict] = {}
        # digest -> Trim, filled by the oracle when functional_reward grades a candidate and
        # read back by _harvest. Without it a graded run would trim every working candidate
        # twice, once to price it and once to strip it, at ~6 simulator calls a time.
        self.trims: dict[str, Any] = {}
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
        """One build per machine, then every candidate on it is a note patch.

        The record is kept, not discarded: `function.trim` needs the base machine's push groups
        to work out which groups a modification changed, and it was already paid for here.
        """
        if entry.digest not in self.graphs:
            machine = self.machine_of(entry)
            record = record_for(machine.to_candidate(cid=0))
            self.records[entry.digest] = record
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
        self,
        entry: Entry,
        source: str,
        round_index: int,
        sim: SimulatorProcess,
        search_config=None,
    ) -> dict[str, Any]:
        machine = self.machine_of(entry)
        graph = self.graph_of(entry)
        evaluate = (
            uniform_evaluator(machine)
            if source == "uninformed"
            else net_evaluator(self.net, graph)
        )

        base_record = self.records.get(entry.digest)
        weight = self.config.loop.functional_reward
        # Per episode, not per run. A Trim is only valid against the base it was computed from,
        # and the same digest can be reached from two different parents on two rounds - so
        # keeping these would eventually hand _harvest another machine's answer. It also stops
        # the dict growing a full cell map per working candidate for the life of the loop.
        self.trims.clear()
        # Calls spent PRICING candidates rather than finding them. Reported separately, never
        # folded into search.calls: the headline is discoveries per simulator call, and quietly
        # charging grading to the search would make a graded run look worse at finding things
        # when all that changed was what it paid to know.
        graded = {"calls": 0, "n": 0}

        def oracle(candidate: Candidate) -> float:
            verdict = sim.simulate(candidate)
            digest = canonical_hash(candidate)
            self.verdicts[digest] = verdict
            if not verdict.get("validCycle"):
                return 0.0
            if weight <= 0.0 or base_record is None:
                return 1.0
            cells = blocks_to_map(candidate["blocks"])
            added = added_cells(machine.cells, cells)
            result = trim(machine, base_record, cells, added)
            self.trims[digest] = result
            graded["calls"] += result.simulator_calls
            graded["n"] += 1
            return graded_reward(result, added, weight)

        search_config = search_config or self.config.search
        search = Search(
            machine,
            graph,
            evaluate,
            oracle,
            config=search_config,
            rng=self.rng,
            cache=self.transposition,
        )
        per_move = max(1, search_config.simulations // max(1, search_config.k))

        state = GameState(
            machine, k=search_config.k, allow_stop=search_config.allow_stop
        )
        root = search.expand(state, add_noise=True)
        played: list[Placement] = []
        stopped = False
        for depth in range(search_config.k):
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
            # The example is recorded BEFORE the action is applied, so `pi` describes the state
            # the search ran from - which is what puts trainable mass on the stop logit. Break
            # after recording, never before, or choosing to stop would teach nothing.
            if is_stop(action, machine.cell_list):
                stopped = True
                break
            played.append(decode_action(action, machine.cell_list))
            child = search.reuse(root, action)
            state = state.advance(action)
            root = child if child is not None else search.expand(state)

        outcome = self._harvest(entry, search, source, round_index, played)
        outcome["grading_calls"] = graded["calls"]
        outcome["graded"] = graded["n"]
        outcome["stopped"] = int(stopped)
        outcome["depth"] = len(played)
        outcome["episodes"] = 1
        return outcome

    def _harvest(self, entry, search, source, round_index, played) -> dict[str, Any]:
        """Everything the search paid for, recorded.

        A search spends its whole budget and ends on one move; the other outcomes are true
        labels that cost nothing extra. Every working one enters the library, **stripped of its
        redundant blocks** - nothing paid for is discarded, and nothing useless is kept.
        """
        working = load_bearing = stripped = 0
        machine = self.machine_of(entry)
        base_record = self.records.get(entry.digest)
        for first_action, candidate, reward in search.attempts:
            digest = canonical_hash(candidate)
            verdict = self.verdicts.get(digest, {})
            period = int(verdict.get("period", 0))
            raw = verdict.get("finalShift") or {"x": 0, "y": 0, "z": 0}
            shift = (raw["x"], raw["y"], raw["z"])
            # Whether the machine FLIES, which is not the same as whether its reward is above
            # zero once `functional_reward` is on: a machine whose every addition was redundant
            # scores exactly 0.0 at w=1.0. Reading "working" off the reward collapsed that case
            # with "does not fly" - measured on the first graded run, where 12 of 19 flying
            # machines were logged as failures and never reached the library. The reward is a
            # training signal; the verdict is the fact.
            flies = bool(verdict.get("validCycle"))
            result = None
            if flies and base_record is not None:
                working += 1
                cells = blocks_to_map(candidate["blocks"])
                # Already trimmed if functional_reward graded this candidate; trimming again
                # would pay ~6 simulator calls for an answer we hold.
                result = self.trims.get(digest)
                if result is None:
                    result = trim(
                        machine, base_record, cells, added_cells(machine.cells, cells)
                    )
                stripped += result.redundant_removed
                trimmed = machine.to_candidate(result.cells, cid=0)
                # A discovery counts when what SURVIVES trimming is not the machine we started
                # from. If every block the search added was redundant, the trimmed machine is
                # the parent again and nothing was found - which is the honest answer, and it
                # is what the old cargo test was groping at without being able to say it.
                if canonical_hash(trimmed) != entry.digest:
                    load_bearing += 1
                self.store.admit(
                    trimmed,
                    entry.digest,
                    round_index,
                    period,
                    shift,
                    redundant_removed=result.redundant_removed,
                    added_is_load_bearing=result.added_is_load_bearing,
                )
            elif flies:
                working += 1
            self.store.record(
                Attempt(
                    digest=digest,
                    parent=entry.digest,
                    round_index=round_index,
                    source=source,
                    reward=reward,
                    working=flies,
                    period=period,
                    shift=shift,
                    blocks=len(candidate["blocks"]),
                    redundant_removed=result.redundant_removed if result else 0,
                    added_is_load_bearing=bool(result.added_is_load_bearing) if result else True,
                )
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
            "load_bearing": load_bearing,
            "redundant_stripped": stripped,
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
        # Steps alternate, so the term a step did not train is exactly zero. Carrying the last
        # real value of each keeps a round from reading as "the policy loss collapsed" whenever
        # the final step happened to be a value step - a reporting artefact, not a fact.
        last: dict[str, Any] = {"loss_policy": float("nan"), "loss_value": float("nan"),
                                "n_policy": 0, "n_value": 0}
        for step in range(steps):
            batch, legal, targets = self._batch(round_index, step)
            if batch is None:
                break
            self.net.train()
            out = self.net(batch)
            loss, parts = total_loss(out, batch, legal, targets, self.config.train)
            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.train.grad_clip)
            self.optimiser.step()
            if parts["n_policy"]:
                last["loss_policy"] = parts["loss_policy"]
                last["n_policy"] = parts["n_policy"]
            if parts["n_value"]:
                last["loss_value"] = parts["loss_value"]
                last["n_value"] = parts["n_value"]
        return last

    def _batch(self, round_index: int, step: int = 0):
        policy_examples = [e for e in self.replay.examples if e.pi is not None]
        value_examples = [e for e in self.replay.examples if e.pi is None]
        # Alternate, for the same reason train.build_step does: mixing both kinds in one batch
        # means the item budget is shared, and a library machine near the cap on its own then
        # leaves room for almost nothing. Measured on the first Stage 1 run: 2 policy and 1-3
        # value examples per step, against Stage 0's much larger ones. The loop trained, but
        # thinly, and every counter still looked healthy.
        pool = policy_examples if step % 2 == 0 else value_examples
        if not pool:
            pool = value_examples if step % 2 == 0 else policy_examples
        want = (
            self.config.train.batch_machines
            if step % 2 == 0
            else self.config.train.value_states_per_machine
        )
        chosen = self.replay.sample_from(pool, self.rng, max(want, 8))
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
            # allow_stop must match what produced `pi`. A target with mass on the stop action
            # against a mask that forbids it puts probability on a -1e9 logit, which is a huge
            # finite loss that trains the model away from a move it was never allowed to make.
            legal_parts.append(
                np.asarray(
                    GameState(
                        machine, (), k=1, allow_stop=self.config.search.allow_stop
                    ).legal_mask(),
                    dtype=bool,
                )
            )
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
                totals = {source: dict.fromkeys(TALLIED, 0) for source in SOURCES}
                predictions: list[tuple[float, float]] = []
                # Exploration decays across the run rather than being a constant.
                search_config = self.config.search.at_round(round_index, loop.rounds)
                for source in plan:
                    entry = self.pick_base()
                    try:
                        result = self.episode(
                            entry, source, round_index, sim, search_config=search_config
                        )
                    except GraphTooLarge:
                        continue
                    for key in TALLIED:
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
                    temperature=round(search_config.temperature, 4),
                    dirichlet_weight=round(search_config.dirichlet_weight, 4),
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
        """**Functional discoveries per 1,000 simulator calls**, model versus the control.

        A discovery counts when, after every redundant block has been stripped, what remains is
        **not the machine we started from**. Every block in it serves a function: removing any
        one would move a piston to another tick, reorder two pistons within a tick, or stop the
        machine working.

        This replaced "non-cargo per 1,000", which asked whether the flight changed - a question
        that cannot separate a useless block from one placed for looks, as a floor, or for any
        other purpose in the game.

        A single round's number is still noise and is logged per round so it can be pooled.
        """
        out = {}
        for source, counts in totals.items():
            calls = counts["calls"]
            out[source] = round(1000 * counts["load_bearing"] / calls, 2) if calls else 0.0
        model_calls = totals["top"]["calls"] + totals["sampled"]["calls"]
        model_hits = totals["top"]["load_bearing"] + totals["sampled"]["load_bearing"]
        out["model"] = round(1000 * model_hits / model_calls, 2) if model_calls else 0.0
        out["redundant_stripped"] = sum(c["redundant_stripped"] for c in totals.values())
        out["grading_calls"] = sum(c["grading_calls"] for c in totals.values())
        # Where the stop action shows up, and the only thing that says whether it learned to
        # use it or just learned that doing nothing is safe. `mean_depth` near 0 with
        # `stopped` near `episodes` is the degenerate policy config.py warns about.
        episodes = sum(c["episodes"] for c in totals.values())
        out["stopped"] = sum(c["stopped"] for c in totals.values())
        out["mean_depth"] = (
            round(sum(c["depth"] for c in totals.values()) / episodes, 2) if episodes else 0.0
        )
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
        # Only printed when the stop action is actually in play, so a default run's line is
        # unchanged and a probe run does not need the JSONL opened to see the one number it
        # exists to produce.
        extra = (
            f"   stop {head['stopped']:>2}  depth {head['mean_depth']:>4.2f}"
            if head.get("stopped")
            else ""
        )
        print(
            f"round {row['round_index']:>3}  {row['seconds']:>6.1f}s   "
            f"functional/1k: model {head['model']:>7.2f}  control {head['uninformed']:>6.2f}   "
            f"stripped {head['redundant_stripped']:>4}   "
            f"library {row['library']['size']:>4}   replay {row['replay']:>6}{extra}"
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
