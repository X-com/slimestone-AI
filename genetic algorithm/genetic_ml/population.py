from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

Candidate = dict[str, Any]


def canonical_hash(candidate: Candidate) -> str:
    """Hash a candidate's shape independent of block order, id, and absolute position.
    Two candidates that describe the same structure (same relative block layout and
    trigger offset) hash identically, so the archive and population can dedupe."""

    blocks = candidate["blocks"]
    if blocks:
        min_x = min(block["x"] for block in blocks)
        min_y = min(block["y"] for block in blocks)
        min_z = min(block["z"] for block in blocks)
    else:
        min_x = min_y = min_z = 0

    normalized_blocks = sorted(
        (block["x"] - min_x, block["y"] - min_y, block["z"] - min_z, block["state"])
        for block in blocks
    )
    trigger = candidate["trigger"]
    normalized_trigger = (
        trigger["x"] - min_x,
        trigger["y"] - min_y,
        trigger["z"] - min_z,
    )

    payload = json.dumps([normalized_trigger, normalized_blocks], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Lineage:
    """One slot in the live population. Tracks a single working candidate plus a
    little bookkeeping so the GA loop can evict stale/oversized lineages."""

    candidate: Candidate
    origin: str
    generation_found: int
    parent_hash: str | None = None
    stale_generations: int = 0
    _hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self._hash = canonical_hash(self.candidate)

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def block_count(self) -> int:
        return len(self.candidate["blocks"])


class Population:
    """Fixed-capacity pool of working lineages. New discoveries are admitted until the
    pool is full, then they replace the "worst" existing lineage (largest block count,
    tie-broken by staleness) so the pool stays biased toward small, actively-mutating
    designs instead of growing without bound."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.lineages: list[Lineage] = []
        self._known_hashes: set[str] = set()

    def __len__(self) -> int:
        return len(self.lineages)

    def has_seen(self, candidate_hash: str) -> bool:
        return candidate_hash in self._known_hashes

    def seed(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            if len(self.lineages) >= self.capacity:
                break
            lineage = Lineage(candidate=candidate, origin="seed", generation_found=0)
            if lineage.hash in self._known_hashes:
                continue
            self.lineages.append(lineage)
            self._known_hashes.add(lineage.hash)

    def admit(self, lineage: Lineage) -> bool:
        """Try to add a newly-discovered working lineage. Returns True if it entered
        the live population (whether by filling a free slot or evicting a worse one)."""

        if lineage.hash in self._known_hashes:
            return False

        self._known_hashes.add(lineage.hash)
        if len(self.lineages) < self.capacity:
            self.lineages.append(lineage)
            return True

        worst_index = max(
            range(len(self.lineages)),
            key=lambda i: (self.lineages[i].block_count, self.lineages[i].stale_generations),
        )
        worst = self.lineages[worst_index]
        if lineage.block_count <= worst.block_count:
            self.lineages[worst_index] = lineage
            return True
        return False
