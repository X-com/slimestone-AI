"""Small graph-transformer: per-node tokens, typed-relation attention bias (shared across layers,
learned once per relation/head pair), bidirectional (no causal mask). Sized for CPU-only training
on a ~54-fixture corpus - see the transformer-gym plan: 4-6 layers, d_model 128-192, don't scale up
until real (post-generator) data shows the small model underfitting.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .encode import BLOCK_VOCAB, N_KINDS, NO_FACING, RELATION_TYPES

N_TERMINATION = 6  # len(TERMINATION_NAMES)
N_MOVABILITY = 3
N_STICKINESS = 3


class RelationAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,N,Dh]
        k = k.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [B,H,N,N]
        logits = logits + rel_bias
        pad = (key_mask == 0).view(b, 1, 1, n)
        logits = logits.masked_fill(pad, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = attn @ v  # [B,H,N,Dh]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelationAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel_bias, key_mask)
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
        )

        rel_bias = torch.einsum("brij,rh->bhij", batch["relations"], self.rel_weight)
        mask = batch["mask"]
        for block in self.blocks:
            x = block(x, rel_bias, mask)
        x = self.norm_out(x)

        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        return dict(
            moves=self.moves_head(x).squeeze(-1),
            stays_attached=self.stays_head(x).squeeze(-1),
            event_grid=self.event_grid_head(x),
            valid_cycle=self.valid_cycle_head(pooled).squeeze(-1),
            termination=self.termination_head(pooled),
            net_shift=self.net_shift_head(pooled),
        )
