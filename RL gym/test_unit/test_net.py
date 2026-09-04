"""The network - md/VERIFICATION.md's weighted tests for Part 3.

Three claims carry most of the weight here, and each one fails silently if untested:

  BATCH INDEPENDENCE  A alone must equal A batched with B. Concatenation without padding is
                      only safe because no edge crosses between machines; if one ever did, or
                      if a summary item were shared, training would look *better* than it is
                      and then fail on a machine evaluated alone. Nothing else catches that.
  THE INVARIANT       parameter count must not move when the machine, R, or the cycle length
                      does. A head that is secretly per-cell passes every shape check and only
                      shows up as a machine that cannot be loaded.
  IT CAN LEARN        one example driven to near-zero loss. Separates "the model is wrong" from
                      "the plumbing is wrong" before either is expensive to diagnose.

Graphs here are synthetic and hand-built, not simulator output, so these run in the fast suite
and a failure points at the network rather than at the record decoder.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from rlgym.blocks import PALETTE_SLOTS
from rlgym.config import NetConfig
from rlgym.graph import (
    ITEM_EVENT,
    ITEM_SUMMARY,
    N_SCALARS,
    REL_SELF,
    REL_TOUCHING,
    Graph,
)
from rlgym.net import Net, make_batch, segment_softmax


def toy_graph(n_cells: int = 3, n_ticks: int = 2, n_events: int = 2, seed: int = 0) -> Graph:
    """A graph with the right shape and nothing real in it.

    Deliberately hand-assembled rather than taken from graph.py, following conftest's rule:
    a bug in the builder must not be able to hide behind a network test that used it.
    """
    rng = np.random.default_rng(seed)
    n_cell_items = n_cells * n_ticks
    n_items = n_cell_items + n_events + 1
    summary = n_items - 1

    item_type = np.zeros(n_items, dtype=np.int64)
    item_type[n_cell_items : n_cell_items + n_events] = ITEM_EVENT
    item_type[summary] = ITEM_SUMMARY

    edges = [(REL_SELF, i, i) for i in range(n_items)]
    for cell in range(n_cells - 1):
        for tick in range(n_ticks):
            a = cell * n_ticks + tick
            b = (cell + 1) * n_ticks + tick
            edges += [(REL_TOUCHING, a, b), (REL_TOUCHING, b, a)]

    return Graph(
        n_items=n_items,
        item_type=item_type,
        block_type=rng.integers(0, 40, n_items),
        facing=rng.integers(0, 7, n_items),
        note_type=np.zeros(n_items, dtype=np.int64),
        note_facing=np.zeros(n_items, dtype=np.int64),
        event_kind=np.zeros(n_items, dtype=np.int64),
        failure_reason=np.zeros(n_items, dtype=np.int64),
        scalars=rng.random((n_items, N_SCALARS)).astype(np.float32),
        edges=np.asarray(edges, dtype=np.int64).T,
        summary_item=summary,
        policy_cells=[(x, 0, 0) for x in range(n_cells)],
        cell_items=[
            [cell * n_ticks + tick for tick in range(n_ticks)] for cell in range(n_cells)
        ],
        n_ticks=n_ticks,
    )


@pytest.fixture(scope="module")
def net() -> Net:
    torch.manual_seed(0)
    model = Net(NetConfig(d_model=32, n_heads=2, d_ff=64, n_rounds=2))
    model.eval()
    return model


# --- batch independence ---------------------------------------------------------------


def test_a_alone_equals_a_batched(net):
    """The load-bearing test for point 16's no-padding batching."""
    a, b = toy_graph(seed=1), toy_graph(n_cells=5, n_ticks=3, seed=2)
    with torch.no_grad():
        alone = net(make_batch([a]))
        together = net(make_batch([a, b]))
    n_actions = len(a.cell_items) * PALETTE_SLOTS + 1
    assert torch.allclose(
        alone.policy_logits, together.policy_logits[:n_actions], atol=1e-5
    )
    assert torch.allclose(alone.value, together.value[:1], atol=1e-5)
    assert torch.allclose(alone.blocks, together.blocks[: len(a.cell_items)], atol=1e-5)


def test_order_in_the_batch_does_not_matter(net):
    a, b = toy_graph(seed=1), toy_graph(n_cells=5, n_ticks=3, seed=2)
    with torch.no_grad():
        forwards = net(make_batch([a, b]))
        backwards = net(make_batch([b, a]))
    assert torch.allclose(forwards.value[0], backwards.value[1], atol=1e-5)
    assert torch.allclose(forwards.value[1], backwards.value[0], atol=1e-5)


def test_each_machine_has_its_own_summary_item(net):
    """A shared summary would leak between machines in two steps - and would make training look
    better than it is before failing silently on a machine evaluated alone."""
    batch = make_batch([toy_graph(seed=1), toy_graph(seed=2)])
    assert batch.summary_items.tolist() == sorted(set(batch.summary_items.tolist()))
    assert len(batch.summary_items) == 2


def test_no_edge_crosses_between_machines():
    """The premise the whole batching scheme rests on, checked directly rather than assumed."""
    batch = make_batch([toy_graph(seed=1), toy_graph(n_cells=5, seed=2)])
    assert (batch.machine_id[batch.src] == batch.machine_id[batch.dst]).all()


# --- the invariant --------------------------------------------------------------------


def test_parameters_do_not_depend_on_the_machine():
    """`ALPHAZERO.md` Part 3: no model dimension may depend on machine size, cell count or tick
    count. Cell count is a data dimension, not a weight dimension."""
    small = Net(NetConfig(d_model=32, n_heads=2, d_ff=64, n_rounds=2))
    assert small.n_parameters() == Net(NetConfig(d_model=32, n_heads=2, d_ff=64, n_rounds=2)).n_parameters()
    with torch.no_grad():
        small(make_batch([toy_graph(n_cells=3, n_ticks=2)]))
        before = small.n_parameters()
        small(make_batch([toy_graph(n_cells=40, n_ticks=17)]))
    assert small.n_parameters() == before


def test_tying_the_trunk_is_what_makes_it_small():
    """The reason point 10 survives: at Stage 0's data size, tying is regularisation, and the
    numbers are 8x apart rather than marginally different."""
    config = NetConfig(n_rounds=8)
    tied = Net(config).n_parameters()
    untied = Net(NetConfig(n_rounds=8, tie_trunk=False)).n_parameters()
    assert untied > 5 * tied


def test_the_default_network_is_about_the_documented_size():
    """~252,000 in Part 3's table. A large drift means a head or a table changed shape."""
    assert 200_000 < Net().n_parameters() < 320_000


def test_policy_head_width_is_the_palette(net):
    batch = make_batch([toy_graph(n_cells=7)])
    with torch.no_grad():
        out = net(batch)
    assert out.policy_logits.shape[0] == 7 * PALETTE_SLOTS + 1  # + the reserved stop slot


# --- masking --------------------------------------------------------------------------


def test_illegal_actions_get_no_probability(net):
    """Masked to -inf **before** the softmax. Zeroing afterwards would leave the survivors
    mis-normalised, which is invisible until the numbers are used as a distribution."""
    graph = toy_graph(n_cells=3)
    batch = make_batch([graph])
    legal = torch.zeros(3 * PALETTE_SLOTS + 1, dtype=torch.bool)
    legal[5] = legal[100] = True
    with torch.no_grad():
        probabilities = net(batch).policy(legal)
    assert probabilities[~legal].max() < 1e-6
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_probabilities_sum_to_one_per_machine(net):
    a, b = toy_graph(n_cells=3), toy_graph(n_cells=5)
    batch = make_batch([a, b])
    legal = torch.ones(batch.policy_size(), dtype=torch.bool)
    with torch.no_grad():
        probabilities = net(batch).policy(legal)
    counts = batch.action_counts()
    for piece in torch.split(probabilities, counts):
        assert piece.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_segment_softmax_normalises_within_each_segment():
    """Derived by hand: two segments, [1,1] and [0,ln 3]. The first is 0.5/0.5; the second is
    1/(1+3) and 3/(1+3)."""
    scores = torch.tensor([[1.0], [1.0], [0.0], [float(np.log(3))]])
    index = torch.tensor([0, 0, 1, 1])
    out = segment_softmax(scores, index, 2).squeeze(-1)
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.25, 0.75], abs=1e-6)


# --- it can actually learn -------------------------------------------------------------


def test_one_example_can_be_driven_to_near_zero_loss():
    """Overfit-one-example. It proves the gradient reaches every head through the encoder, the
    trunk and the pooling - which is exactly what a plumbing bug breaks and a modelling mistake
    does not."""
    from rlgym.config import TrainConfig
    from rlgym.net import make_batch as batch_of
    from rlgym.train import Targets, total_loss

    torch.manual_seed(0)
    model = Net(NetConfig(d_model=32, n_heads=2, d_ff=64, n_rounds=2))
    graph = toy_graph(n_cells=3)
    batch = batch_of([graph])
    size = 3 * PALETTE_SLOTS + 1
    legal = torch.ones(size, dtype=torch.bool)
    target = np.zeros(size, dtype=np.float32)
    target[7] = 1.0
    targets = Targets(policy=[target], value=torch.tensor([1.0]))
    config = TrainConfig(value_pos_weight=1.0)

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        loss, parts = total_loss(model(batch), batch, legal, targets, config)
        first = first if first is not None else parts["loss"]
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert parts["loss"] < 0.05 * first
    with torch.no_grad():
        assert int(model(batch).policy(legal).argmax()) == 7


def test_the_value_head_can_learn_to_see_a_single_noted_cell():
    """Regression for a measured failure, not a hypothetical one.

    Reading `v` off the summary item alone made it a constant: a note moved the noted cell's own
    vector by ~6.0 and the summary vector by ~0.004, so `v` predicted the machine's base rate for
    every action - spread 0.0002 across 60 actions on a held-out machine. It ranked nothing,
    while the policy head's AUC looked healthy the whole time.

    So the test is not "does `v` twitch at initialisation" - a uniform pool over 483 items gives
    any single note a weight of 1/483 whatever the architecture. It is **can `v` be trained to
    separate two machines that differ only in which cell is noted**, which is the thing the
    summary bottleneck made impossible.
    """
    from rlgym.config import TrainConfig
    from rlgym.train import Targets, total_loss

    torch.manual_seed(0)
    model = Net(NetConfig(d_model=32, n_heads=2, d_ff=64, n_rounds=3))
    graph = toy_graph(n_cells=60, n_ticks=8, seed=3)

    def with_note(item: int) -> Graph:
        note_type = graph.note_type.copy()
        note_type[item] = 31
        return dataclasses.replace(graph, note_type=note_type)

    # Four machines identical but for the noted cell; the answer depends only on which one.
    items = (0, 120, 300, 400)
    targets_z = [1.0, 0.0, 1.0, 0.0]
    batch = make_batch([with_note(i) for i in items])
    size = batch.policy_size()
    legal = torch.ones(size, dtype=torch.bool)
    targets = Targets(
        policy=[None] * len(items), value=torch.tensor(targets_z, dtype=torch.float32)
    )
    config = TrainConfig(value_pos_weight=1.0, w_policy=0.0)

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = total_loss(model(batch), batch, legal, targets, config)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        v = model(batch).value
    assert v[0] > 0.7 and v[2] > 0.7, f"v failed to reach the positives: {v.tolist()}"
    assert v[1] < 0.3 and v[3] < 0.3, f"v failed to reach the negatives: {v.tolist()}"
