"""Small graph-transformer: per-node tokens, typed-relation edge-restricted attention (each node
attends only over its RELATION_TYPES neighbours - see encode.py's "self" relation for why every
node always has at least one). Bidirectional within that edge set (no causal mask). Sized for
CPU-only training on a ~54-fixture corpus - see the transformer-gym plan: 4-6 layers, d_model
128-192, don't scale up until real (post-generator) data shows the small model underfitting.

Was dense [B,H,N,N] relation-biased global attention through SDL9-era encode.py; switched to
edge-restricted attention because the dense form does not scale (measured 71 GB for one relation
tensor on the largest fixture in the corpus - see the tick-chunking plan). This is a real semantic
change, not just a memory optimization: a node can no longer attend anywhere, only along its
typed-relation edges. Must be validated (comparable accuracy on the <=300-node fixtures the old
dense form could still handle) rather than assumed harmless - see the plan's verification gates.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .encode import BLOCK_VOCAB, N_KINDS, NO_FACING, RELATION_TYPES

N_TERMINATION = 6  # len(TERMINATION_NAMES)
N_MOVABILITY = 3
N_STICKINESS = 4  # none, sticks-all (slime), sticks-all-except-slime (honey, unused), never-sticks (glazed terracotta)


class RelationAttention(nn.Module):
    """Edge-restricted multi-head attention. x is flat over the whole batch ([M, D], M = B*N_max
    from collate's padding scheme); rel_edges indexes into that flat dimension directly, so edges
    never cross between different samples in the batch (collate offsets each sample's own edges by
    its slot's base index, i*N_max) without needing an explicit batch-id argument here."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rel_edges: torch.Tensor, rel_weight: torch.Tensor) -> torch.Tensor:
        m, d = x.shape
        h, dh = self.n_heads, self.d_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(m, h, dh)
        k = k.view(m, h, dh)
        v = v.view(m, h, dh)

        etype, src, dst = rel_edges[0], rel_edges[1], rel_edges[2]
        out = torch.zeros(m, h, dh, device=x.device, dtype=x.dtype)
        if src.numel() == 0:
            return self.out(out.reshape(m, d))

        qd = q[dst]                                    # [E, H, Dh]
        ks = k[src]                                     # [E, H, Dh]
        logits = (qd * ks).sum(-1) / math.sqrt(dh)       # [E, H]
        logits = logits + rel_weight[etype]              # [E, H]

        # Numerically-stable scatter softmax, grouped by destination node (each node's incoming
        # edge set is its attention key set - the edge-restricted equivalent of a row-softmax over
        # a dense [N,N] logits matrix).
        dst_h = dst.unsqueeze(-1).expand(-1, h)
        logits_max = torch.full((m, h), float("-inf"), device=x.device, dtype=logits.dtype)
        logits_max = logits_max.scatter_reduce(0, dst_h, logits, reduce="amax", include_self=True)
        logits_max = torch.nan_to_num(logits_max, neginf=0.0)  # nodes with no in-edges: never indexed via dst anyway
        exp = (logits - logits_max[dst]).exp()           # [E, H]
        denom = torch.zeros(m, h, device=x.device, dtype=exp.dtype).scatter_add_(0, dst_h, exp)
        attn = exp / denom[dst].clamp(min=1e-9)           # [E, H]

        weighted = attn.unsqueeze(-1) * v[src]            # [E, H, Dh]
        dst_hd = dst.view(-1, 1, 1).expand(-1, h, dh)
        out = out.scatter_add(0, dst_hd, weighted)

        return self.out(out.reshape(m, d))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelationAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, rel_edges: torch.Tensor, rel_weight: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel_edges, rel_weight)
        x = x + self.ffn(self.norm2(x))
        return x


class PhysicsTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.block_type_emb = nn.Embedding(BLOCK_VOCAB, d_model)
        self.facing_emb = nn.Embedding(NO_FACING + 1, d_model)
        self.movability_emb = nn.Embedding(N_MOVABILITY, d_model)
        self.stickiness_emb = nn.Embedding(N_STICKINESS, d_model)
        self.scalar_proj = nn.Linear(3 + 1 + 1 + 3, d_model)  # flags, is_trigger, is_air, rel_pos

        self.rel_weight = nn.Parameter(torch.zeros(len(RELATION_TYPES), n_heads))
        self.n_heads = n_heads

        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)

        self.moves_head = nn.Linear(d_model, 1)
        self.stays_head = nn.Linear(d_model, 1)
        self.event_grid_head = nn.Linear(d_model, N_KINDS)
        self.valid_cycle_head = nn.Linear(d_model, 1)
        self.termination_head = nn.Linear(d_model, N_TERMINATION)
        self.net_shift_head = nn.Linear(d_model, 3)

    def forward(self, batch: dict) -> dict:
        b, n_max = batch["mask"].shape
        x = (
            self.block_type_emb(batch["block_type"])
            + self.facing_emb(batch["facing"])
            + self.movability_emb(batch["movability"])
            + self.stickiness_emb(batch["stickiness"])
            + self.scalar_proj(
                torch.cat(
                    [batch["flags"], batch["is_trigger"].unsqueeze(-1),
                     batch["is_air"].unsqueeze(-1), batch["rel_pos"]],
                    dim=-1,
                )
            )
        )  # [B, N_max, D]

        d = x.shape[-1]
        x_flat = x.reshape(b * n_max, d)
        for block in self.blocks:
            x_flat = block(x_flat, batch["rel_edges"], self.rel_weight)
        x_flat = self.norm_out(x_flat)
        x = x_flat.view(b, n_max, d)

        mask = batch["mask"]
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        return dict(
            moves=self.moves_head(x).squeeze(-1),
            stays_attached=self.stays_head(x).squeeze(-1),
            event_grid=self.event_grid_head(x),
            valid_cycle=self.valid_cycle_head(pooled).squeeze(-1),
            termination=self.termination_head(pooled),
            net_shift=self.net_shift_head(pooled),
        )
