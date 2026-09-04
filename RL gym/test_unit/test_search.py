"""PUCT and backup - md/VERIFICATION.md's weighted tests for Part 4.

The oracle and the evaluator are fakes here, so these run in the fast suite and a failure points
at the search rather than at the simulator. What is being checked:

  THE WORKED EXAMPLE  ALPHAZERO.md Part 4's table, asserted as literal numbers. It was computed
                      by hand in the design document; if the implementation disagrees, one of
                      the two is wrong and the test says which line.
  CONSERVATION        visits and totals must account for exactly what happened. A backup bug is
                      otherwise invisible: the search still runs, still returns a move, and is
                      merely worse.
  NO SIGN FLIP        single-player. Two-player MCTS negates the value at each level and the
                      habit is easy to import; here it would invert every preference.
  THE BANNED LEAF     a leaf must be evaluated by descending to terminal, never by simulating
                      the partial machine. That mistake looks authoritative - the number really
                      is exact - while systematically pruning the extensions being hunted.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from rlgym.config import SearchConfig
from rlgym.game import GameState, Machine, canonical_hash
from rlgym.search import Node, Search, uniform_evaluator

SLIME = 30
STONE = 31


@pytest.fixture
def machine() -> Machine:
    """Two stone blocks in a row. Small enough that the action space is countable by hand."""
    return Machine(cells={(0, 0, 0): 1, (1, 0, 0): 1}, trigger=(0, 0, 0), radius=1)


def uniform_over(size: int, legal: np.ndarray) -> np.ndarray:
    return legal / legal.sum()


# --- PUCT ------------------------------------------------------------------------------


def test_puct_reproduces_the_worked_example():
    """ALPHAZERO.md Part 4, root after 20 iterations, c_puct = 1.25, sqrt(20) = 4.472.

        action                  P     N    W     Q      bonus   PUCT
        piston (1,1,1) west   0.30   12   7.0   0.583   0.129   0.712
        slime (0,1,2)         0.25    5   0.0   0.000   0.233   0.233
        stone (1,1,1)         0.10    2   0.0   0.000   0.186   0.186
        observer (0,1,2) N    0.05    1   0.0   0.000   0.140   0.140

    Written out as literals rather than recomputed, so this test disagrees with a rearranged
    formula instead of agreeing with it.
    """
    node = Node(
        state=None,
        prior=np.array([0.30, 0.25, 0.10, 0.05]),
        legal=np.array([True, True, True, True]),
        visits=np.array([12, 5, 2, 1]),
        totals=np.array([7.0, 0.0, 0.0, 0.0]),
    )
    assert int(node.visits.sum()) == 20
    assert node.q().tolist() == pytest.approx([0.5833, 0.0, 0.0, 0.0], abs=1e-4)
    assert node.puct(1.25).tolist() == pytest.approx([0.712, 0.233, 0.186, 0.140], abs=1e-3)
    # The piston wins despite slime having the LARGER exploration bonus - evidence displacing
    # opinion, which is the entire mechanism.
    assert node.puct(1.25)[1] > 1.25 * 0.30 * math.sqrt(20) / 13
    assert node.best(1.25) == 0


def test_the_first_iteration_follows_the_prior_exactly():
    """With every N and Q at zero, PUCT reduces to a positive multiple of P."""
    node = Node(
        state=None,
        prior=np.array([0.1, 0.5, 0.4]),
        legal=np.array([True, True, True]),
    )
    assert node.best(1.25) == 1
    assert np.argsort(-node.puct(1.25)).tolist() == np.argsort(-node.prior).tolist()


def test_illegal_actions_are_never_selected():
    node = Node(
        state=None,
        prior=np.array([0.9, 0.05, 0.05]),
        legal=np.array([False, True, True]),
    )
    assert node.best(1.25) != 0
    assert node.puct(1.25)[0] == -np.inf


def test_a_failing_action_is_overtaken_without_anything_being_scheduled():
    """If the piston starts failing, Q falls and slime overtakes automatically."""
    prior = np.array([0.30, 0.25])
    winning = Node(state=None, prior=prior, legal=np.array([True, True]),
                   visits=np.array([12, 5]), totals=np.array([7.0, 0.0]))
    failing = Node(state=None, prior=prior, legal=np.array([True, True]),
                   visits=np.array([12, 5]), totals=np.array([0.0, 0.0]))
    assert winning.best(1.25) == 0
    assert failing.best(1.25) == 1


# --- pi --------------------------------------------------------------------------------


def test_pi_is_the_visit_distribution():
    """Part 4's second table: 140/20/15/25 out of 200 becomes 0.70/0.10/0.075/0.125."""
    node = Node(
        state=None,
        prior=np.array([0.30, 0.25, 0.10, 0.35]),
        legal=np.ones(4, dtype=bool),
        visits=np.array([140, 20, 15, 25]),
        totals=np.zeros(4),
    )
    assert node.pi(1.0).tolist() == pytest.approx([0.70, 0.10, 0.075, 0.125])
    assert node.pi(1.0).sum() == pytest.approx(1.0)


def test_temperature_zero_is_argmax():
    node = Node(
        state=None,
        prior=np.ones(3) / 3,
        legal=np.ones(3, dtype=bool),
        visits=np.array([5, 9, 1]),
        totals=np.zeros(3),
    )
    assert node.pi(0.0).tolist() == [0.0, 1.0, 0.0]


def test_pi_of_an_unvisited_node_is_empty_rather_than_uniform():
    """An unsearched node must not produce a confident-looking flat target - `pi` undefined has
    to stay distinguishable from `pi` uniform, or Part 5's edge case cannot be detected."""
    node = Node(state=None, prior=np.ones(4) / 4, legal=np.ones(4, dtype=bool))
    assert node.pi(1.0).sum() == 0.0


# --- backup ----------------------------------------------------------------------------


def _search(machine, reward_of, k=1, simulations=25, seed=0):
    calls = {"n": 0}

    def oracle(candidate):
        calls["n"] += 1
        return reward_of(candidate)

    search = Search(
        machine,
        graph=None,
        evaluate=uniform_evaluator(machine),
        oracle=oracle,
        config=SearchConfig(k=k, simulations=simulations, dirichlet_weight=0.0),
        rng=random.Random(seed),
    )
    return search, calls


def test_backup_conserves_reward_per_edge(machine):
    """At k=1 each action leads to exactly one terminal, so `W(a)` must be `N(a) x reward(a)`
    exactly - not approximately, and not off by the value of some other branch.

    A backup bug is otherwise invisible: the search still runs, still returns a move, and is
    merely worse. This is the check that makes it visible.
    """
    # Reward depends only on the resulting machine, so each action has one true value.
    search, _ = _search(machine, lambda c: float(len(c["blocks"]) % 2), simulations=40)
    result = search.run()
    root = result.root
    truth = {action: reward for action, _, reward in search.attempts}
    assert truth, "the search never called the oracle"
    for action, reward in truth.items():
        assert float(root.totals[action]) == pytest.approx(
            reward * int(root.visits[action])
        ), f"action {action} backed up {root.totals[action]} over {root.visits[action]} visits"
    assert int(root.visits.sum()) >= result.simulator_calls


def test_no_sign_flip_in_backup(machine):
    """Single-player: a reward of 1 must arrive as +1 at every level. Two-player MCTS negates,
    and importing that habit here would invert every preference the search forms."""
    search, _ = _search(machine, lambda c: 1.0, k=2, simulations=12)
    result = search.run()
    assert float(result.root.totals.min()) >= 0.0
    assert float(result.root.q().max()) == pytest.approx(1.0)


def test_reward_reaches_the_root_from_depth_two(machine):
    """The whole point of k > 1: a discovery two placements deep must credit the first one."""
    search, _ = _search(machine, lambda c: 1.0, k=2, simulations=8)
    result = search.run()
    assert int(result.root.visits.sum()) >= 1
    assert float(result.root.totals.sum()) > 0.0


# --- budget and the cache --------------------------------------------------------------


def test_the_budget_is_counted_in_simulator_calls(machine):
    """Not iterations. It is the only currency the uninformed control can also spend, and it is
    the actual bottleneck."""
    search, calls = _search(machine, lambda c: 0.0, simulations=17)
    result = search.run()
    assert result.simulator_calls == 17
    assert calls["n"] == 17


def test_a_duplicate_terminal_costs_a_lookup_not_a_simulation(machine):
    """Point 23's global cache. Deliberately NOT counted against the budget: the budget measures
    what was actually spent, and a cache hit spent nothing."""
    search, calls = _search(machine, lambda c: 0.0, simulations=30)
    search.run()
    repeated = next(iter(search.cache))
    before, before_calls = search.calls, calls["n"]
    # Re-ask for a state already in the cache. Neither the oracle nor the budget may move.
    state = next(
        s
        for s in [GameState(machine, (p,), k=1) for p in _all_placements(machine)]
        if canonical_hash(s.to_candidate()) in search.cache
    )
    search.reward_of(state, first_action=0)
    assert calls["n"] == before_calls
    assert search.calls == before
    assert repeated in search.cache


def _all_placements(machine):
    from rlgym.game import decode_action

    state = GameState(machine, k=1)
    return [
        decode_action(index, machine.cell_list)
        for index, legal in enumerate(state.legal_mask())
        if legal
    ]


def test_every_attempt_is_recorded(machine):
    """Part 6: a search spends its budget and ends on one move; the other outcomes are true
    labels that cost nothing extra. Nothing paid for may be discarded."""
    search, calls = _search(machine, lambda c: 0.0, simulations=20)
    search.run()
    assert len(search.attempts) == calls["n"]


# --- the banned leaf evaluator ---------------------------------------------------------


def test_leaves_are_complete_machines_not_partial_ones(machine):
    """Every candidate handed to the oracle must have all k placements written in.

    Simulating a partial machine is cheap and exact and answers the wrong question - a piston
    alone always fails, so it would systematically kill every multi-block extension while
    looking authoritative.
    """
    sizes = []

    def oracle(candidate):
        sizes.append(len(candidate["blocks"]))
        return 0.0

    search = Search(
        machine,
        graph=None,
        evaluate=uniform_evaluator(machine),
        oracle=oracle,
        config=SearchConfig(k=2, simulations=10, dirichlet_weight=0.0),
        rng=random.Random(0),
    )
    search.run()
    assert sizes

    # The direct form of the claim: the state handed to the oracle is always terminal.
    for _ in range(20):
        state = search.rollout_to_terminal(GameState(machine, k=2), None)
        assert state.is_terminal or not any(state.legal_mask())

    # And no candidate is ever the untouched base machine, which is what a partial-machine
    # evaluation at depth 0 would produce.
    from rlgym.game import canonical_hash

    base = canonical_hash(machine.to_candidate())
    assert all(canonical_hash(c) != base for _, c, _ in search.attempts)


def test_dirichlet_noise_only_touches_legal_actions(machine):
    search = Search(
        machine,
        graph=None,
        evaluate=uniform_evaluator(machine),
        oracle=lambda c: 0.0,
        config=SearchConfig(k=1, dirichlet_weight=0.25),
        rng=random.Random(0),
    )
    root = search.expand(GameState(machine, k=1), add_noise=True)
    assert float(root.prior[~root.legal].sum()) == 0.0
    assert float(root.prior.sum()) == pytest.approx(1.0, abs=1e-6)
    # Noise must actually move the prior, or point 20's failure returns silently.
    plain = search.expand(GameState(machine, k=1), add_noise=False)
    assert not np.allclose(root.prior, plain.prior)


def test_top_m_restriction_keeps_exactly_m_actions(machine):
    search = Search(
        machine,
        graph=None,
        evaluate=uniform_evaluator(machine),
        oracle=lambda c: 0.0,
        config=SearchConfig(k=1, search_top_m=5, dirichlet_weight=0.0),
        rng=random.Random(0),
    )
    root = search.expand(GameState(machine, k=1))
    assert int((root.prior > 0).sum()) == 5
    assert int(root.legal.sum()) == 5


def test_the_rollout_actually_samples(machine):
    """Regression: one shared uniform makes `u ** (1/w)` monotone in `w`, so the exponential
    race collapses to argmax and every iteration reaches the same leaf.

    The search would still run, still return a move, and learn nothing beyond its own prior -
    a failure with no symptom other than a suspiciously flat set of attempts.
    """
    search, _ = _search(machine, lambda c: 0.0, k=1, simulations=1)
    reached = {
        canonical_hash(search.rollout_to_terminal(GameState(machine, k=1), None).to_candidate())
        for _ in range(30)
    }
    assert len(reached) > 1, "every rollout reached the same leaf - the sampling collapsed"


def test_a_search_that_exhausts_its_space_terminates(machine):
    """A tiny machine can run out of reachable candidates and then hit the transposition cache
    forever, spending no budget. Without a stall guard `run` never returns."""
    search, _ = _search(machine, lambda c: 0.0, k=1, simulations=100_000)
    result = search.run(budget=5_000)
    assert result.simulator_calls < 5_000
