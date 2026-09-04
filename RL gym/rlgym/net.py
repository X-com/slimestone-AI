"""The network - md/ALPHAZERO.md Part 3. One pass, four answers.

    p   how promising each placement is    ~1,250 at R=1     policy
    v   expected final reward              1                 value
    B   which blocks start behaving differently               auxiliary
    D   which of the 7 failureReason values                   auxiliary

**The invariant, enforced by test_parameters_do_not_depend_on_the_machine:**

> No model dimension may depend on machine size, cell count, or tick count.

The policy head is `Linear(d_model -> 48)` applied *per cell*. 32 cells or 3,000, the weights are
the same object; cell count is a data dimension. Everything here is built so that changing R,
the machine, or the cycle length changes no parameter at all.

Batching is by concatenation, never padding (point 16). Attention is edge-restricted and no edge
crosses between machines, so two machines in one batch genuinely cannot see each other and no
mask is needed to keep them apart. Every item carries `machine_id` so a loss can average within
a machine before averaging across them - otherwise a machine's weight in the loss depends on who
its batchmates happen to be.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from rlgym.blocks import PALETTE_SLOTS
from rlgym.config import NetConfig
from rlgym.graph import (
    FEATURE_VERSION,
    ITEM_CELL,
    ITEM_EVENT,
    ITEM_SUMMARY,
    N_ITEM_TYPES,
    N_RELATION_SLOTS,
    N_SCALARS,
    Graph,
)
from rlgym.record import N_FAILURE_REASONS, N_KINDS

N_BLOCK_IDS = 256
N_FACINGS = 7  # 0..5 plus "no facing"
N_NOTE_SLOTS = 49  # palette slot + 1, 0 meaning "not noted"
N_NOTE_FACINGS = 8
N_EVENT_KINDS = N_KINDS + 1  # plus "not an event"
NEG_INF = -1e9  # masking constant; -inf would make the softmax produce NaN for an all-masked row


# --- batching -------------------------------------------------------------------------------


@dataclass
class Batch:
    """Several graphs concatenated into one flat item list."""

    item_type: torch.Tensor  # [N]
    block_type: torch.Tensor
    facing: torch.Tensor
    note_type: torch.Tensor
    note_facing: torch.Tensor
    event_kind: torch.Tensor
    failure_reason: torch.Tensor
    scalars: torch.Tensor  # [N, N_SCALARS]
    machine_id: torch.Tensor  # [N]
    rel: torch.Tensor  # [E]
    src: torch.Tensor
    dst: torch.Tensor
    summary_items: torch.Tensor  # [M] - one per machine, NEVER shared
    pool_item: torch.Tensor  # [P] item index of one (policy cell, tick)
    pool_row: torch.Tensor  # [P] which pooled cell row it belongs to
    cell_machine: torch.Tensor  # [C] machine of each pooled cell row
    cells_per_machine: list[int]
    n_items: int
    n_machines: int

    @property
    def n_cells(self) -> int:
        return int(self.cell_machine.shape[0])

    def action_counts(self) -> list[int]:
        return [count * PALETTE_SLOTS + 1 for count in self.cells_per_machine]

    def policy_size(self) -> int:
        return sum(self.action_counts())


def make_batch(graphs: list[Graph]) -> Batch:
    """Concatenate. Item indices are offset per graph; edges follow the same offset."""
    if not graphs:
        raise ValueError("cannot batch zero graphs")
    for graph in graphs:
        if graph.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"graph built by {graph.feature_version!r} but this network speaks "
                f"{FEATURE_VERSION!r} - the scalar columns have shifted, so a checkpoint "
                "trained on the old ones would be silently wrong. Rebuild the graph cache."
            )

    offsets: list[int] = []
    total = 0
    for graph in graphs:
        offsets.append(total)
        total += graph.n_items

    def cat(name: str) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([getattr(g, name) for g in graphs]))

    machine_id = torch.from_numpy(
        np.concatenate([np.full(g.n_items, i, dtype=np.int64) for i, g in enumerate(graphs)])
    )
    rel = torch.from_numpy(np.concatenate([g.edges[0] for g in graphs]))
    src = torch.from_numpy(
        np.concatenate([g.edges[1] + off for g, off in zip(graphs, offsets)])
    )
    dst = torch.from_numpy(
        np.concatenate([g.edges[2] + off for g, off in zip(graphs, offsets)])
    )

    pool_item: list[int] = []
    pool_row: list[int] = []
    cell_machine: list[int] = []
    row = 0
    for index, (graph, off) in enumerate(zip(graphs, offsets)):
        for items in graph.cell_items:
            for item in items:
                pool_item.append(item + off)
                pool_row.append(row)
            cell_machine.append(index)
            row += 1

    return Batch(
        item_type=cat("item_type"),
        block_type=cat("block_type"),
        facing=cat("facing"),
        note_type=cat("note_type"),
        note_facing=cat("note_facing"),
        event_kind=cat("event_kind"),
        failure_reason=cat("failure_reason"),
        scalars=torch.from_numpy(np.concatenate([g.scalars for g in graphs])).float(),
        machine_id=machine_id,
        rel=rel,
        src=src,
        dst=dst,
        summary_items=torch.tensor(
            [g.summary_item + off for g, off in zip(graphs, offsets)], dtype=torch.long
        ),
        pool_item=torch.tensor(pool_item, dtype=torch.long),
        pool_row=torch.tensor(pool_row, dtype=torch.long),
        cell_machine=torch.tensor(cell_machine, dtype=torch.long),
        cells_per_machine=[len(g.cell_items) for g in graphs],
        n_items=total,
        n_machines=len(graphs),
    )


# --- segment reductions ---------------------------------------------------------------------


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Softmax within each segment. `scores` is [E, H]; `index` says which segment each row is.

    Max-subtracted before exponentiating, which is not decoration: attention scores routinely
    reach tens once d_model is 128, and exp of that overflows float32 in a way that shows up as
    a NaN loss twenty steps in rather than as an error here.
    """
    highest = torch.full(
        (n_segments, scores.shape[1]), NEG_INF, dtype=scores.dtype, device=scores.device
    )
    highest = highest.scatter_reduce(
        0, index.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True
    )
    weights = (scores - highest[index]).exp()
    denominator = torch.zeros_like(highest).index_add_(0, index, weights)
    return weights / denominator[index].clamp_min(1e-12)


def segment_log_softmax(
    scores: torch.Tensor, index: torch.Tensor, n_segments: int
) -> torch.Tensor:
    """Log-softmax within each segment, for the masked policy. [E] in, [E] out."""
    flat = scores.unsqueeze(-1)
    highest = torch.full((n_segments, 1), NEG_INF, dtype=flat.dtype, device=flat.device)
    highest = highest.scatter_reduce(
        0, index.unsqueeze(-1), flat, reduce="amax", include_self=True
    )
    shifted = flat - highest[index]
    denominator = torch.zeros_like(highest).index_add_(0, index, shifted.exp())
    return (shifted - denominator[index].clamp_min(1e-12).log()).squeeze(-1)


# --- the trunk ------------------------------------------------------------------------------


class RelationalAttention(nn.Module):
    """Point 6's Choice C: `score = query(dst) . key(src) + relation_bias[type]`.

    The dot product makes the weight depend on what the neighbour currently *is* - a moving
    slime and a stationary stone produce different keys - while the bias covers the *kind* of
    connection. Neither alone is enough: bias-only cannot tell two slimes apart, dot-only cannot
    tell "touching" from "powers".
    """

    def __init__(self, config: NetConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        # 64 slots whether or not a machine uses them, so relation ids stay stable across
        # models. Appending a relation is safe; renumbering one silently changes meaning.
        self.relation_bias = nn.Parameter(torch.zeros(N_RELATION_SLOTS, config.n_heads))

    def forward(self, x: torch.Tensor, rel, src, dst, n_items: int) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (-1, self.n_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        scores = (q[dst] * k[src]).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.relation_bias[rel]
        alpha = segment_softmax(scores, dst, n_items)
        gathered = torch.zeros(
            n_items, self.n_heads, self.head_dim, dtype=x.dtype, device=x.device
        )
        gathered.index_add_(0, dst, alpha.unsqueeze(-1) * v[src])
        return self.out(gathered.reshape(n_items, -1))


class TrunkBlock(nn.Module):
    """Pre-norm residual. Attention moves facts between items; the FFN combines them within one.

    The FFN is where *"recorded push group of 11, my note is adjacent, so really 12, which is
    exactly the limit"* happens. Attention cannot do that - it can only average, and the average
    of 12 things looks like the average of 11.
    """

    def __init__(self, config: NetConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = RelationalAttention(config)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rel, src, dst, n_items):
        x = x + self.dropout(self.attention(self.norm_attention(x), rel, src, dst, n_items))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


# --- outputs --------------------------------------------------------------------------------


@dataclass
class Outputs:
    policy_logits: torch.Tensor  # [A] flat over machines, stop last within each
    action_machine: torch.Tensor  # [A]
    value: torch.Tensor  # [M] in 0..1
    blocks: torch.Tensor  # [C] per policy cell, a logit
    reason: torch.Tensor  # [M, 7]

    def log_policy(self, legal: torch.Tensor) -> torch.Tensor:
        """Masked log-softmax. Illegal logits go to -inf **before** the softmax; zeroing
        afterwards would leave the rest mis-normalised."""
        masked = self.policy_logits.masked_fill(~legal, NEG_INF)
        n_machines = int(self.action_machine.max().item()) + 1
        return segment_log_softmax(masked, self.action_machine, n_machines)

    def policy(self, legal: torch.Tensor) -> torch.Tensor:
        return self.log_policy(legal).exp()

    def split(self, counts: list[int]) -> list[torch.Tensor]:
        return list(torch.split(self.policy_logits, counts))


class Net(nn.Module):
    def __init__(self, config: NetConfig | None = None):
        super().__init__()
        self.config = config or NetConfig()
        d = self.config.d_model

        # Categorical -> learned lookup table. Type and facing are separate (point 2) so one
        # table of 256 rows and one of 7 replace 60+ combined rows each learned from scratch.
        self.type_emb = nn.Embedding(N_ITEM_TYPES, d)
        self.block_emb = nn.Embedding(N_BLOCK_IDS, d)
        self.facing_emb = nn.Embedding(N_FACINGS, d)
        self.note_emb = nn.Embedding(N_NOTE_SLOTS, d)
        self.note_facing_emb = nn.Embedding(N_NOTE_FACINGS, d)
        self.event_emb = nn.Embedding(N_EVENT_KINDS, d)
        self.reason_emb = nn.Embedding(N_FAILURE_REASONS, d)
        # One projection per item type: an event carries no block type, so a single shared
        # layer over zero-filled columns would teach the model that events are made of air.
        self.scalar_proj = nn.ModuleList(
            [nn.Linear(N_SCALARS, d) for _ in range(N_ITEM_TYPES)]
        )

        blocks = 1 if self.config.tie_trunk else self.config.n_rounds
        self.trunk = nn.ModuleList([TrunkBlock(self.config) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(d)

        self.pool_query = nn.Parameter(torch.zeros(d))
        # A second learned query, pooling over EVERY item of a machine rather than one cell's
        # ticks. It exists because of a measurement, not a hunch: reading `v` off the summary
        # item alone made it a constant. A note moves the noted cell's own vector by ~6.0 and
        # the summary vector by ~0.004 - a thousandfold attenuation - so `v` was structurally
        # blind to the placement it was being asked to judge, and predicted the machine's base
        # rate for every action (spread 0.0002 across 60 actions on a held-out machine).
        #
        # Part 3 predicted the cause and did not connect it to this head: "the summary item
        # makes everything 2 steps apart, but it is a bottleneck - 421 items squeezed through
        # one 128-number vector - so fine detail cannot route that way." Which cell was noted
        # is exactly fine detail. A softmax pool CAN concentrate on one item; the summary's
        # own aggregation cannot be relied on to.
        # Randomly initialised, unlike pool_query: pooling over thousands of items from a
        # zero query starts perfectly uniform, which gives one noted cell a weight of
        # 1/n and a correspondingly tiny gradient to escape from.
        self.value_query = nn.Parameter(torch.randn(d) * d**-0.5)
        self.head_policy = nn.Linear(d, PALETTE_SLOTS)
        self.head_stop = nn.Linear(d, 1)
        # Two outputs, one used. Output 1 is reserved and unlabelled so that adding a second
        # value later is a config change rather than a shape change that makes every existing
        # checkpoint unloadable at exactly the moment you want to compare against them.
        self.head_value = nn.Linear(3 * d, 2)  # summary + attention pool + max pool
        self.head_blocks = nn.Linear(d, 1)
        self.head_reason = nn.Linear(d, N_FAILURE_REASONS)

    # --- steps of the forward pass ----------------------------------------------------

    def encode(self, batch: Batch) -> torch.Tensor:
        """All terms **added, not concatenated** - which is what keeps point 1's "the feature
        list will grow later" cheap: a new group is one more term and d_model never changes."""
        x = (
            self.type_emb(batch.item_type)
            + self.block_emb(batch.block_type)
            + self.facing_emb(batch.facing)
            + self.note_emb(batch.note_type)
            + self.note_facing_emb(batch.note_facing)
            + self.event_emb(batch.event_kind)
            + self.reason_emb(batch.failure_reason)
        )
        scalars = torch.zeros_like(x)
        for kind in (ITEM_CELL, ITEM_EVENT, ITEM_SUMMARY):
            rows = (batch.item_type == kind).nonzero(as_tuple=True)[0]
            if rows.numel():
                scalars = scalars.index_copy(
                    0, rows, self.scalar_proj[kind](batch.scalars[rows])
                )
        return x + scalars

    def pool_cells(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        """One vector per policy cell, by learned attention over that cell's tick items.

        Point 12: averaging divides a cell's one informative tick by the cycle length, and a
        maximum cannot represent a cell that needs *two* ticks together - pushed at tick 3, hit
        again at tick 8. A learned query can put all its weight on tick 4 or split between 3
        and 8, which neither fixed reduction can.
        """
        n_rows = batch.n_cells
        scores = (x[batch.pool_item] @ self.pool_query).unsqueeze(-1)
        alpha = segment_softmax(scores, batch.pool_row, n_rows)
        pooled = torch.zeros(n_rows, x.shape[1], dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch.pool_row, alpha * x[batch.pool_item])
        return pooled

    def pool_machine(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        """One vector per machine, by learned attention over **all** of its items.

        The route the summary bottleneck cannot provide. A softmax over every item can put its
        weight on the one noted cell; the summary item's own aggregation demonstrably does not.
        Machine-size independent, like everything else here - the query is one vector of
        `d_model` numbers whatever the machine.
        """
        scores = (x @ self.value_query).unsqueeze(-1)
        alpha = segment_softmax(scores, batch.machine_id, batch.n_machines)
        pooled = torch.zeros(batch.n_machines, x.shape[1], dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch.machine_id, alpha * x)

        # ...and a max, because attention alone was measured to be not enough. A softmax over
        # ~1,500 items with O(1) scores is nearly uniform: to put real weight on one item the
        # query has to grow until scores span log(n) ~ 7, and it does not get there in a few
        # hundred steps. The first fix moved `v`'s spread from 0.0002 to 0.0006 - still a
        # constant.
        #
        # A max concentrates by construction and needs to learn nothing to do it, and the noted
        # cell IS the outlier: its vector moves by ~6.0 when everything else moves by ~0.004.
        # Point 12 rejected max for pooling a cell's TICKS, because a cell can need two ticks
        # together; that argument does not apply to "is there something unusual here", which is
        # what `v` needs. The two are concatenated rather than chosen between.
        peaks = torch.full_like(pooled, float("-inf"))
        peaks = peaks.scatter_reduce(
            0,
            batch.machine_id.unsqueeze(-1).expand_as(x),
            x,
            reduce="amax",
            include_self=True,
        )
        return torch.cat([pooled, peaks], dim=-1)

    def forward(self, batch: Batch) -> Outputs:
        x = self.encode(batch)
        checkpointing = self.config.checkpoint_trunk and self.training and torch.is_grad_enabled()
        for round_index in range(self.config.n_rounds):
            block = self.trunk[0 if self.config.tie_trunk else round_index]
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, batch.rel, batch.src, batch.dst, batch.n_items,
                    use_reentrant=False,
                )
            else:
                x = block(x, batch.rel, batch.src, batch.dst, batch.n_items)
        x = self.final_norm(x)

        summary = x[batch.summary_items]
        pooled = self.pool_cells(x, batch)
        focus = self.pool_machine(x, batch)

        per_cell = self.head_policy(pooled)  # [C, 48]
        stop = self.head_stop(summary)  # [M, 1]
        logits: list[torch.Tensor] = []
        machine_of: list[torch.Tensor] = []
        start = 0
        for index, count in enumerate(batch.cells_per_machine):
            block_logits = per_cell[start : start + count].reshape(-1)
            logits.append(torch.cat([block_logits, stop[index]]))
            machine_of.append(
                torch.full((block_logits.numel() + 1,), index, dtype=torch.long)
            )
            start += count

        return Outputs(
            policy_logits=torch.cat(logits),
            action_machine=torch.cat(machine_of),
            value=torch.sigmoid(
                self.head_value(torch.cat([summary, focus], dim=-1))[:, 0]
            ),
            blocks=self.head_blocks(pooled).squeeze(-1),
            reason=self.head_reason(summary),
        )

    # --- convenience ------------------------------------------------------------------

    @torch.no_grad()
    def priors(self, graph: Graph, legal: list[bool]) -> tuple[np.ndarray, float]:
        """One machine: the masked prior over its actions, and `v`. What MCTS calls at expand."""
        self.eval()
        batch = make_batch([graph])
        out = self(batch)
        mask = torch.tensor(legal, dtype=torch.bool)
        if mask.shape[0] != out.policy_logits.shape[0]:
            raise ValueError(
                f"legal mask has {mask.shape[0]} entries but the graph implies "
                f"{out.policy_logits.shape[0]} actions - the mask and the graph disagree "
                "about the machine"
            )
        return out.policy(mask).numpy(), float(out.value[0])

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
