"""Graph cache and the Stage 0 supervised dataset - md/ALPHAZERO.md Parts 5 and 8.

Two jobs, both about not paying twice.

**The graph cache.** Building a graph costs 157 ms and needs a simulator run first; a machine's
graph never changes, so it is built once and stored as one `.npz`. `data/graphs/` is a cache in
the strict sense - deleting it costs time and nothing else.

**The note patch.** All 207,935 training examples on 33 machines are 33 graph builds, because a
placement changes features and never structure. `apply_notes` is the whole reason Stage 0 fits in
minutes instead of nine hours.

Stage 0 is **ordinary supervised learning**, not reinforcement learning: `pi` is the exact
distribution over working moves and `z` the exact outcome, both from brute force. RL begins at
Stage 1, when `pi` starts coming from search.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rlgym.game import GameState, Machine, decode_action
from rlgym.graph import Graph, apply_notes, build
from rlgym.labeller import Label, LabelSet, load_fixture
from rlgym.simlog import record_for

CACHE_VERSION = 2  # bumped when the graph cell set changed; stale caches must fail loudly


# --- the graph cache --------------------------------------------------------------------


def _save(graph: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index_cells = sorted(graph.cell_index)
    np.savez_compressed(
        path,
        version=np.int64(CACHE_VERSION),
        feature_version=np.array(graph.feature_version),
        item_type=graph.item_type,
        block_type=graph.block_type,
        facing=graph.facing,
        note_type=graph.note_type,
        note_facing=graph.note_facing,
        event_kind=graph.event_kind,
        failure_reason=graph.failure_reason,
        scalars=graph.scalars,
        edges=graph.edges,
        summary_item=np.int64(graph.summary_item),
        n_ticks=np.int64(graph.n_ticks),
        policy_cells=np.asarray(graph.policy_cells, dtype=np.int64).reshape(-1, 3),
        cell_items=np.asarray(graph.cell_items, dtype=np.int64).reshape(
            len(graph.cell_items), -1
        ),
        index_cells=np.asarray(index_cells, dtype=np.int64).reshape(-1, 3),
        index_items=np.asarray(
            [graph.cell_index[cell] for cell in index_cells], dtype=np.int64
        ).reshape(len(index_cells), -1),
    )


def _load(path: Path) -> Graph:
    data = np.load(path, allow_pickle=False)
    if int(data["version"]) != CACHE_VERSION:
        raise ValueError(f"{path} was written by cache version {int(data['version'])}")
    index_cells = [tuple(int(v) for v in row) for row in data["index_cells"]]
    return Graph(
        n_items=int(data["item_type"].shape[0]),
        item_type=data["item_type"],
        block_type=data["block_type"],
        facing=data["facing"],
        note_type=data["note_type"],
        note_facing=data["note_facing"],
        event_kind=data["event_kind"],
        failure_reason=data["failure_reason"],
        scalars=data["scalars"],
        edges=data["edges"],
        summary_item=int(data["summary_item"]),
        policy_cells=[tuple(int(v) for v in row) for row in data["policy_cells"]],
        cell_items=[[int(v) for v in row] for row in data["cell_items"]],
        n_ticks=int(data["n_ticks"]),
        feature_version=str(data["feature_version"]),
        cell_index={
            cell: [int(v) for v in row]
            for cell, row in zip(index_cells, data["index_items"])
        },
    )


def graph_for(
    machine: Machine,
    name: str,
    cache_dir: Path | None = None,
    tick_cap: int = 32,
) -> Graph:
    """The base graph for a machine, from cache when possible.

    The cache key is the machine's name, not its content - so a cache directory must be cleared
    when `graph.py` changes. `FEATURE_VERSION` is stored and checked by `make_batch`, which
    turns the dangerous half of that (a checkpoint reading shifted columns) into a loud error.
    """
    path = None if cache_dir is None else Path(cache_dir) / f"{name}.npz"
    if path is not None and path.exists():
        return _load(path)
    graph = build(machine, record_for(machine.to_candidate(cid=0)), tick_cap=tick_cap)
    if path is not None:
        _save(graph, path)
    return graph


def machine_for(name: str, radius: int = 1) -> Machine:
    return Machine.from_candidate(load_fixture(name), radius=radius)


# --- the Stage 0 dataset ----------------------------------------------------------------


@dataclass
class MachineData:
    """One machine's complete supervised signal: a dense policy target and every terminal."""

    name: str
    machine: Machine
    graph: Graph
    labelset: LabelSet
    legal: np.ndarray  # [A] bool
    policy_target: np.ndarray  # [A] sums to 1, or all-zero when nothing works
    rewards: np.ndarray  # [A] the reward used, cargo weighting applied
    labelled: np.ndarray  # [A] bool - which actions have a label at all

    @property
    def has_policy_target(self) -> bool:
        """Part 5's edge case: if no move works, `pi` is undefined. Skip the policy term for
        this machine and keep the value term, rather than training toward a uniform target that
        says every move is equally good when in fact none is."""
        return bool(self.policy_target.sum() > 0)

    def labelled_actions(self) -> np.ndarray:
        return np.flatnonzero(self.labelled)


def reward_of(label: Label, reward_cargo: float) -> float:
    """The reward the model is trained toward.

    `reward_cargo` is the knob the plan hangs the 313:1 cargo problem on. At 1.0 a block that
    merely rides along counts as a full success, which is today's behaviour and the thing under
    suspicion. Lower it and both `pi` and `z` shift together, because they are two views of one
    reward function and letting them disagree would be a subtle, invisible bug.
    """
    if label.reward <= 0.0:
        return 0.0
    return reward_cargo if label.maybe_cargo else 1.0


def machine_data(
    name: str,
    labelset: LabelSet,
    radius: int = 1,
    cache_dir: Path | None = None,
    tick_cap: int = 32,
    reward_cargo: float = 1.0,
) -> MachineData:
    machine = machine_for(name, radius=radius)
    if len(machine.cell_list) * 48 + 1 != labelset.action_slots:
        raise ValueError(
            f"{name}: labels describe {labelset.action_slots} action slots but the machine at "
            f"radius {radius} has {machine.action_count}. The labels were made at a different "
            "radius; relabel or change the radius."
        )
    graph = graph_for(machine, name, cache_dir=cache_dir, tick_cap=tick_cap)

    size = machine.action_count
    legal = np.zeros(size, dtype=bool)
    rewards = np.zeros(size, dtype=np.float32)
    labelled = np.zeros(size, dtype=bool)
    for label in labelset.labels:
        legal[label.action] = True
        labelled[label.action] = True
        rewards[label.action] = reward_of(label, reward_cargo)

    target = rewards * legal
    total = float(target.sum())
    policy_target = (target / total) if total > 0 else np.zeros(size, dtype=np.float32)
    return MachineData(
        name=name,
        machine=machine,
        graph=graph,
        labelset=labelset,
        legal=legal,
        policy_target=policy_target.astype(np.float32),
        rewards=rewards,
        labelled=labelled,
    )


def noted_graph(data: MachineData, action: int) -> Graph:
    """The graph of the terminal state reached by one action - a note patch, not a rebuild."""
    placement = decode_action(action, data.machine.cell_list)
    return apply_notes(data.graph, (placement,))


def terminal_state(data: MachineData, action: int) -> GameState:
    return GameState(data.machine, (decode_action(action, data.machine.cell_list),), k=1)


def load_corpus(
    labels_dir: Path,
    radius: int = 1,
    cache_dir: Path | None = None,
    tick_cap: int = 32,
    reward_cargo: float = 1.0,
    names: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, MachineData]:
    from rlgym.labeller import load as load_labels

    paths = sorted(Path(labels_dir).glob("*.json"))
    if names is not None:
        wanted = set(names)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit(
            f"no label files in {labels_dir} - run: py -m rlgym.labeller --all --out {labels_dir}"
        )
    out: dict[str, MachineData] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        try:
            out[path.stem] = machine_data(
                path.stem,
                load_labels(path),
                radius=radius,
                cache_dir=cache_dir,
                tick_cap=tick_cap,
                reward_cargo=reward_cargo,
            )
        except Exception as exc:  # a machine over the tick cap, or a stale label file
            skipped[path.stem] = f"{type(exc).__name__}: {str(exc)[:120]}"
            if verbose:
                print(f"  skipped {path.stem}: {skipped[path.stem]}")
            continue
    if not out:
        raise SystemExit("every machine was skipped - see the messages above")
    load_corpus.skipped = skipped
    return out
