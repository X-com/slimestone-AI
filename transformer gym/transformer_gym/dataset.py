"""Batch-runs the existing hand-built JSON fixtures through the C++ simulator and encodes each
into a Sample. Structured so a future contraption generator only has to replace `fixture_names()`
/ how `.dat` files are produced - `encode.py` and everything downstream is unaffected.

No SimulatorPool here (see the transformer-gym plan, Part A gap #6): that's a generator-era
throughput concern. At today's ~54-fixture scale, one subprocess per fixture is plenty.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .encode import RELATION_TYPES, N_KINDS, Sample, TooManyNodesError, encode
from .simlog_reader import run_fixture

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "flying machines" / "json"

# ponytail: relations are a dense [N,N] tensor per sample - fine for the ~50-block hand fixtures
# this pipeline was built for, but some fixtures in this folder are full schematic-scale machines
# (thousands of blocks) that would blow up to gigabytes of dense adjacency. Skip those for now;
# upgrade to sparse relation edges (edge-index lists, not dense NxN) if/when large fixtures need
# to be trained on too.
MAX_NODES = 300


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


class SimlogDataset(Dataset):
    """Runs+encodes every fixture once at construction time (small corpus, cheap to keep in RAM)."""

    def __init__(self, names: list[str] | None = None, max_nodes: int = MAX_NODES):
        candidates = names if names is not None else fixture_names()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.samples: list[Sample] = []
        self.names: list[str] = []
        workdir = Path(self._tmpdir.name)
        for name in candidates:
            log = run_fixture(name, workdir)
            try:
                sample = encode(log, max_nodes=max_nodes)
            except TooManyNodesError as e:
                print(f"skipping {e}")
                continue
            self.samples.append(sample)
            self.names.append(name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class SimlogDirDataset(Dataset):
    """Encodes every .simlog already sitting in a directory - the generator's output shard, or
    any other pre-simulated batch. No simulator subprocess at train time, unlike SimlogDataset:
    generation (transformer gym/generator/) and training are fully decoupled, so a shard can be
    produced once and reused across many training runs."""

    def __init__(self, shard_dir: Path, max_nodes: int = MAX_NODES):
        self.samples: list[Sample] = []
        self.names: list[str] = []
        for path in sorted(shard_dir.glob("*.simlog")):
            try:
                sample = encode(path, max_nodes=max_nodes)
            except TooManyNodesError as e:
                print(f"skipping {e}")
                continue
            self.samples.append(sample)
            self.names.append(path.stem)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate(samples: list[Sample]) -> dict:
    """Pads the variable node count N to the batch max and returns a dict of batched tensors
    plus a `mask` [B, N] (1 = real node, 0 = padding)."""
    n_max = max(s.block_type.shape[0] for s in samples)
    b = len(samples)

    block_type = torch.zeros(b, n_max, dtype=torch.long)
    facing = torch.full((b, n_max), 6, dtype=torch.long)
    flags = torch.zeros(b, n_max, 3)
    movability = torch.zeros(b, n_max, dtype=torch.long)
    stickiness = torch.zeros(b, n_max, dtype=torch.long)
    is_trigger = torch.zeros(b, n_max)
    is_air = torch.zeros(b, n_max)
    rel_pos = torch.zeros(b, n_max, 3)
    relations = torch.zeros(b, len(RELATION_TYPES), n_max, n_max)
    mask = torch.zeros(b, n_max)
    y_moves = torch.zeros(b, n_max)
    y_stays_attached = torch.zeros(b, n_max)
    y_event_grid = torch.zeros(b, n_max, N_KINDS)
    y_net_shift = torch.zeros(b, 3)
    y_valid_cycle = torch.zeros(b)
    y_termination = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(samples):
        n = s.block_type.shape[0]
        block_type[i, :n] = s.block_type
        facing[i, :n] = s.facing
        flags[i, :n] = s.flags
        movability[i, :n] = s.movability
        stickiness[i, :n] = s.stickiness
        is_trigger[i, :n] = s.is_trigger
        is_air[i, :n] = s.is_air
        rel_pos[i, :n] = s.rel_pos
        relations[i, :, :n, :n] = s.relations
        mask[i, :n] = 1.0
        y_moves[i, :n] = s.y_moves
        y_stays_attached[i, :n] = s.y_stays_attached
        y_event_grid[i, :n] = s.y_event_grid
        y_net_shift[i] = s.y_net_shift
        y_valid_cycle[i] = s.y_valid_cycle
        y_termination[i] = s.y_termination

    return dict(
        names=[s.name for s in samples], block_type=block_type, facing=facing, flags=flags,
        movability=movability, stickiness=stickiness, is_trigger=is_trigger, is_air=is_air,
        rel_pos=rel_pos, relations=relations, mask=mask, y_moves=y_moves,
        y_stays_attached=y_stays_attached, y_event_grid=y_event_grid, y_net_shift=y_net_shift,
        y_valid_cycle=y_valid_cycle, y_termination=y_termination,
    )
