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

from .encode import N_KINDS, Sample, TooManyNodesError, encode
from .simlog_reader import iter_ticks, read_footer, run_fixture
from .state import apply_tick, initial_state

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "flying machines" / "json"

# Relations are now a sparse edge list (encode.py's rel_edges), not a dense [N,N] tensor - see the
# tick-chunking plan. This is a safety valve against pathological inputs, not the real capacity
# ceiling MAX_NODES=300 used to be: measured edge counts are 549-535,733x smaller than the dense
# form would have needed, and the largest fixture in the corpus (1797 onepointfive, N=59,729) now
# encodes fine.
MAX_NODES = 200_000


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
    plus a `mask` [B, N] (1 = real node, 0 = padding). Relation edges are concatenated into one
    flat edge list spanning the whole padded batch (each sample's own src/dst offset by its slot's
    base index i*n_max) rather than a per-sample dense [N,N] block - samples never share edges
    with each other since every edge index stays within its own [i*n_max, (i+1)*n_max) range, and
    model.py's attention operates on the batch as one flat [B*n_max, D] graph."""
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
    mask = torch.zeros(b, n_max)
    y_moves = torch.zeros(b, n_max)
    y_stays_attached = torch.zeros(b, n_max)
    y_event_grid = torch.zeros(b, n_max, N_KINDS)
    y_net_shift = torch.zeros(b, 3)
    y_valid_cycle = torch.zeros(b)
    y_termination = torch.zeros(b, dtype=torch.long)
    edge_chunks = []

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
        mask[i, :n] = 1.0
        y_moves[i, :n] = s.y_moves
        y_stays_attached[i, :n] = s.y_stays_attached
        y_event_grid[i, :n] = s.y_event_grid
        y_net_shift[i] = s.y_net_shift
        y_valid_cycle[i] = s.y_valid_cycle
        y_termination[i] = s.y_termination

        offset = i * n_max
        et, es, ed = s.rel_edges
        edge_chunks.append(torch.stack([et, es + offset, ed + offset]))

    rel_edges = torch.cat(edge_chunks, dim=1) if edge_chunks else torch.zeros(3, 0, dtype=torch.long)

    return dict(
        names=[s.name for s in samples], block_type=block_type, facing=facing, flags=flags,
        movability=movability, stickiness=stickiness, is_trigger=is_trigger, is_air=is_air,
        rel_pos=rel_pos, rel_edges=rel_edges, mask=mask, y_moves=y_moves,
        y_stays_attached=y_stays_attached, y_event_grid=y_event_grid, y_net_shift=y_net_shift,
        y_valid_cycle=y_valid_cycle, y_termination=y_termination,
    )


class TickChunkDataset(Dataset):
    """Per-tick recurrent training data (the tick-chunking plan's Phase 5): each item is one
    fixture's full tick sequence, `[(SimState(T), events(T)), ...]`, rather than one aggregate
    Sample. Still data-layer only - no model/training code depends on this yet. `SimState(T)` is
    built incrementally via state.apply_tick, verified against the replay/resume harness in
    util tools/check_state_builder.py, NOT against a model."""

    def __init__(self, names: list[str] | None = None, max_nodes: int = MAX_NODES):
        candidates = names if names is not None else fixture_names()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.sequences: list[list] = []   # per fixture: [(SimState, events), ...]
        self.names: list[str] = []
        workdir = Path(self._tmpdir.name)
        for name in candidates:
            log = run_fixture(name, workdir)
            data = log.read_bytes()
            footer = read_footer(data)
            try:
                state = initial_state(data, footer, max_nodes=max_nodes)
            except TooManyNodesError as e:
                print(f"skipping {e}")
                continue
            seq = []
            for tick, events in iter_ticks(data, footer):
                seq.append((state, events))
                state = apply_tick(state, tick, events)
            seq.append((state, []))  # final state, no further events
            self.sequences.append(seq)
            self.names.append(name)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list:
        return self.sequences[idx]
