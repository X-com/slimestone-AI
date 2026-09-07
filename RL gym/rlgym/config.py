"""Every knob, in one place - md/ALPHAZERO.md and the build plan's deferred-decision table.

The point of this file is that **no deferred decision is settled in code**. Each one that the
plan lists as "decide it by measuring the net effect" appears here as a field with the
default that reproduces today's behaviour, so turning it on is a config edit and turning it
off again is a config edit, and both are recorded in the metrics row.

    open problem                     knob                  default
    binary reward is 313:1 cargo     reward_cargo          1.0   (= today: cargo counts fully)
    policy-target ageing             policy_age_decay      0.0   (= flat, no ageing)
    root oversampling                root_weight           1.0   (= no down-weight)
    top-M search restriction         search_top_m          0     (= unrestricted)
    when to stop editing             allow_stop            False (= k is an exact length)
    what a working machine is worth  functional_reward     0.0   (= 1.0 for anything working)
    graph build cost                 tick_cap              32
    d_model / T / heads              d_model, n_rounds...  128 / 8 / 4

Load with `Config.load(path)`; a JSON file need only carry the fields it overrides.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class NetConfig:
    """Part 3. **The invariant: no field here may depend on machine size, cells or ticks.**"""

    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_rounds: int = 8  # T - trunk applications, all sharing one block's weights (point 10)
    tie_trunk: bool = True
    dropout: float = 0.0
    # Recompute each trunk round in the backward pass instead of storing it. Attention gathers
    # q, k and v per EDGE, so one round of a 430,000-edge batch holds ~660 MB and eight rounds
    # exhausted memory (measured: the process segfaults, it does not raise). Costs ~33% more
    # compute for 8x less memory, which is the right trade on CPU where memory is the wall.
    checkpoint_trunk: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must divide into {self.n_heads} heads"
            )


@dataclass
class TrainConfig:
    """Part 5. The loss is `w_v.BCE(v,z) + w_p.CE(p,pi) + w_B.BCE(B,b) + w_D.CE(D,d)`."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 400
    warmup_steps: int = 20
    batch_machines: int = 2  # root graphs per step
    value_states_per_machine: int = 8  # noted terminals sampled per machine per step
    # Memory for the backward pass scales with edges x n_rounds, and machines span 519 to
    # 15,647 items - so an unbounded step segfaults, measured, on a draw of two large machines.
    # ONE budget covering roots and terminals alike; the first root is always admitted.
    max_items_per_step: int = 14000
    grad_clip: float = 1.0

    w_policy: float = 1.0
    w_value: float = 1.0
    w_blocks: float = 0.25  # B - auxiliary, deliberately small
    w_reason: float = 0.25  # D - auxiliary

    # Point 14 stands for the value head alone: z is unbalanced, so the positive term is
    # scaled by the OBSERVED rate rather than by a guessed constant. 0.0 disables it.
    value_pos_weight: float = 0.0  # 0 = compute from the batch

    # --- the deferred decisions, as knobs ---------------------------------------------
    reward_cargo: float = 1.0
    root_weight: float = 1.0
    policy_age_decay: float = 0.0

    eval_every: int = 25
    eval_budget: int = 100
    seed: int = 0


@dataclass
class SearchConfig:
    """Part 4. Budget is counted in **simulator calls**, never iterations."""

    c_puct: float = 1.25
    simulations: int = 200  # simulator calls per episode, the currency of every comparison
    dirichlet_alpha: float = 0.0  # 0 = derive as 10/actions, per AlphaZero's own scaling
    dirichlet_weight: float = 0.25
    temperature: float = 1.0
    search_top_m: int = 0  # 0 = unrestricted. The measurement is recall@B with it on and off
    k: int = 2
    reuse_subtree: bool = True

    # Unmask the stop action, so `k` becomes a CEILING the model may stop short of rather than an
    # exact edit length. Off by default, and the default is not timidity: under a binary
    # does-it-work reward, stopping at depth 0 returns the base machine, which works, so a
    # perfect 1.0 is available for zero risk and doing nothing is the optimal policy. Turning
    # this on without `loop.functional_reward` above 0 measures exactly that degenerate policy.
    # See md/OPEN.md, "Ranking which working changes are useful".
    allow_stop: bool = False

    # Both exploration knobs start high and decay - README.md: "Both start high and decay.
    # Nothing switches over; it is one loop throughout." They were fixed constants until this
    # was added, so the loop explored exactly as hard on its last round as on its first.
    #
    # Decayed together on purpose. They are the same idea applied at two points: Dirichlet noise
    # widens what the search LOOKS at, temperature widens what it PLAYS from what it found.
    # Decaying one without the other gives a search that explores broadly and then commits at
    # random, or one that looks only where it already believes and then agonises over the choice.
    temperature_final: float = 0.25
    dirichlet_weight_final: float = 0.05
    decay_rounds: int = 0  # 0 = decay across the loop's own round count

    def at_round(self, round_index: int, total_rounds: int) -> "SearchConfig":
        """This round's exploration settings. Linear, because it is predictable and because
        nothing here justifies a shape more specific than that yet."""
        span = self.decay_rounds or max(1, total_rounds)
        # A one-round run explores at FULL strength, not at the final value. Collapsing to the
        # end of the schedule when there is nowhere to decay to would silently make every short
        # run - which is every debugging run - the least exploratory one.
        progress = min(1.0, max(0.0, (round_index - 1) / (span - 1))) if span > 1 else 0.0
        from dataclasses import replace

        return replace(
            self,
            temperature=self.temperature
            + (self.temperature_final - self.temperature) * progress,
            dirichlet_weight=self.dirichlet_weight
            + (self.dirichlet_weight_final - self.dirichlet_weight) * progress,
        )


@dataclass
class LoopConfig:
    """Part 6. The three-way split, and the 5% that must stay genuinely uninformed."""

    rounds: int = 10
    episodes_per_round: int = 8
    share_top: float = 0.60
    share_sampled: float = 0.35
    share_uninformed: float = 0.05
    train_steps_per_round: int = 50
    # Redundant blocks are stripped before admission rather than the machine being refused, so
    # there is no longer an admission filter to switch off. See rlgym/function.py.
    trim_discoveries: bool = True

    # Reward shaping, and the only thing that makes stopping early cost anything.
    #
    #     0.0   today: every working candidate scores 1.0, whatever it is made of
    #     w     a working candidate scores (1 - w) + w * (added blocks that SURVIVED trimming
    #           / added blocks), so a machine whose every addition was redundant - including
    #           the machine that stopped at depth 0 and added nothing - scores (1 - w)
    #
    # At w = 1.0 the reward IS the surviving fraction. Redundancy is the one objective
    # usefulness signal that exists (rlgym/function.py); this is where it becomes a reward
    # rather than only a filter.
    #
    # It is not free: grading a candidate needs `trim`, which is ~6 simulator calls. Only
    # WORKING candidates are graded and the working rate is a few percent, so the overhead is
    # small - but it is real, it is counted, and `grading_calls` reports it in the round row.
    functional_reward: float = 0.0
    seed: int = 0

    def shares(self) -> tuple[float, float, float]:
        total = self.share_top + self.share_sampled + self.share_uninformed
        if total <= 0:
            raise ValueError("budget shares must sum to something positive")
        return (
            self.share_top / total,
            self.share_sampled / total,
            self.share_uninformed / total,
        )


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    radius: int = 1  # the 2-cell shell is a hyperparameter; 1 until everything is verified
    tick_cap: int = 32
    labels_dir: Path = Path("data/labels")
    graphs_dir: Path = Path("data/graphs")
    runs_dir: Path = Path("data/runs")

    @classmethod
    def load(cls, path: Path | str | None) -> "Config":
        config = cls()
        if path is None:
            return config
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return config.merged(payload)

    def merged(self, payload: dict) -> "Config":
        """Overlay a partial dict. Unknown keys are refused rather than ignored - a typo in a
        config is otherwise a silent no-op that looks like the knob having no effect."""
        out = Config(
            net=NetConfig(**{**asdict(self.net), **payload.get("net", {})}),
            train=TrainConfig(**{**asdict(self.train), **payload.get("train", {})}),
            search=SearchConfig(**{**asdict(self.search), **payload.get("search", {})}),
            loop=LoopConfig(**{**asdict(self.loop), **payload.get("loop", {})}),
        )
        known = {f.name for f in fields(Config)}
        for key, value in payload.items():
            if key in ("net", "train", "search", "loop") or key.startswith("_"):
                continue  # "_" keys are comments; JSON has none of its own
            if key not in known:
                raise ValueError(f"unknown config key {key!r}")
            setattr(out, key, Path(value) if key.endswith("_dir") else value)
        return out

    def to_json(self) -> dict:
        payload = asdict(self)
        for key in ("labels_dir", "graphs_dir", "runs_dir"):
            payload[key] = str(payload[key])
        return payload
