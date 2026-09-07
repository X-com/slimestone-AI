"""PUCT with simulator leaves - md/ALPHAZERO.md Part 4.

    PUCT(a) = Q(a) + c_puct . P(a) . sqrt(sum N) / (1 + N(a))
              ----   -------------------------------------
              what    how much I still want to look here
              I found

At the start every `N` and `Q` is zero, so ranking is purely `P` and the first iterations follow
the network's opinion exactly. As visits accumulate `Q` dominates and evidence displaces opinion;
`c_puct` sets how long opinion holds out. Nothing needs scheduling.

**Single-player, so no sign flipping in backup.** Two-player MCTS negates the value at each level;
this does not, which removes a whole class of sign bug rather than managing it.

**Evaluating a leaf by simulating the partial machine is banned.** It is cheap and exact and it
answers the wrong question: a piston placed alone is dead weight and fails, while `piston + slime`
is exactly the extension being hunted. Mid-edit validity systematically prunes away the
modifications we want, and it looks authoritative because the number really is exact. Leaves are
evaluated by **descending to terminal and simulating that**.

**Budget is counted in simulator calls, never iterations.** They coincide at k=2, and they will
not later; the unit decides what the headline result means, and simulator calls are the only
currency the uninformed control can also spend.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rlgym.config import SearchConfig
from rlgym.game import Candidate, GameState, Machine, canonical_hash
from rlgym.graph import Graph, apply_notes

# state -> (prior over actions, value). Supplied by the caller so the uninformed control can
# pass a uniform one and spend its budget through the identical code path.
Evaluator = Callable[[GameState], tuple[np.ndarray, float]]
# candidate -> reward. One simulator call.
Oracle = Callable[[Candidate], float]


@dataclass
class Node:
    state: GameState
    prior: np.ndarray  # [A]
    legal: np.ndarray  # [A] bool
    visits: np.ndarray = field(default=None)  # [A] N
    totals: np.ndarray = field(default=None)  # [A] W
    children: dict[int, "Node"] = field(default_factory=dict)
    n: int = 0  # visits to this node

    def __post_init__(self) -> None:
        size = self.prior.shape[0]
        if self.visits is None:
            self.visits = np.zeros(size, dtype=np.int64)
        if self.totals is None:
            self.totals = np.zeros(size, dtype=np.float64)

    def q(self) -> np.ndarray:
        return np.divide(
            self.totals, self.visits, out=np.zeros_like(self.totals), where=self.visits > 0
        )

    def puct(self, c_puct: float) -> np.ndarray:
        """`Q + c_puct . P . sqrt(sum N) / (1 + N)`, with illegal actions at -inf.

        `sqrt(sum N)` uses the visits to this node's edges, which is what makes the bonus shrink
        for everyone as evidence accumulates anywhere, not just on the action being considered.
        """
        total = math.sqrt(max(1e-8, float(self.visits.sum())))
        scores = self.q() + c_puct * self.prior * total / (1 + self.visits)
        return np.where(self.legal, scores, -np.inf)

    def best(self, c_puct: float) -> int:
        return int(np.argmax(self.puct(c_puct)))

    def pi(self, temperature: float) -> np.ndarray:
        """Visit counts as a distribution - the policy-improvement target.

        `pi` is better than `p` because it is backed by real outcomes. Training `p` toward it is
        the entire learning signal; no human ever says which move was good.
        """
        counts = self.visits.astype(np.float64)
        if counts.sum() <= 0:
            return np.zeros_like(counts)
        if temperature <= 1e-6:
            out = np.zeros_like(counts)
            out[int(np.argmax(counts))] = 1.0
            return out
        weighted = counts ** (1.0 / temperature)
        return weighted / weighted.sum()


@dataclass
class SearchResult:
    root: Node
    pi: np.ndarray
    chosen: int
    simulator_calls: int
    attempts: list[tuple[int, Candidate, float]]  # (action taken from root, candidate, reward)
    value_predictions: list[tuple[float, float]]  # (v predicted, z actually found)


class Search:
    """One PUCT search over one base machine.

    The transposition cache is a plain `canonical_hash -> reward` dictionary rather than DAG
    merging: a duplicate terminal costs a lookup instead of a simulator call, which is the whole
    practical benefit, without multi-parent backup. A cache hit is deliberately **not** counted
    as a simulator call, because the budget's job is to measure what was actually spent.
    """

    def __init__(
        self,
        machine: Machine,
        graph: Graph,
        evaluate: Evaluator,
        oracle: Oracle,
        config: SearchConfig | None = None,
        rng: random.Random | None = None,
        cache: dict[str, float] | None = None,
    ):
        self.machine = machine
        self.graph = graph
        self.evaluate = evaluate
        self.oracle = oracle
        self.config = config or SearchConfig()
        self.rng = rng or random.Random(0)
        self.cache = {} if cache is None else cache
        self.calls = 0
        self.attempts: list[tuple[int, Candidate, float]] = []
        self.value_predictions: list[tuple[float, float]] = []

    # --- the four steps -------------------------------------------------------------

    def expand(self, state: GameState, add_noise: bool = False) -> Node:
        prior, _ = self.evaluate(state)
        legal = np.asarray(state.legal_mask(), dtype=bool)
        prior = np.where(legal, prior, 0.0)
        total = prior.sum()
        # A prior that is zero everywhere legal would make PUCT pick action 0 forever. Fall back
        # to uniform rather than letting an untrained network silently collapse the search.
        prior = prior / total if total > 0 else legal / max(1, legal.sum())
        if add_noise:
            prior = self._noisy(prior, legal)
        if self.config.search_top_m > 0:
            prior = self._restrict(prior, self.config.search_top_m)
            legal = prior > 0
        return Node(state=state, prior=prior, legal=legal)

    def _noisy(self, prior: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """`P = 0.75.P + 0.25.Dirichlet(alpha)` at the root.

        Without it an action rated near zero is never selected, so no evidence about it ever
        arrives and the belief becomes permanent regardless of truth - point 20's failure
        exactly. `alpha ~ 10/actions` follows AlphaZero's own scaling (0.3 at 35 moves, 0.03 at
        250), which is about 0.008 here.
        """
        indices = np.flatnonzero(legal)
        if not len(indices):
            return prior
        alpha = self.config.dirichlet_alpha or (10.0 / len(indices))
        noise = self.rng_dirichlet(alpha, len(indices))
        weight = self.config.dirichlet_weight
        out = prior.copy()
        out[indices] = (1 - weight) * prior[indices] + weight * noise
        return out

    def rng_dirichlet(self, alpha: float, size: int) -> np.ndarray:
        generator = np.random.default_rng(self.rng.getrandbits(63))
        return generator.dirichlet(np.full(size, alpha))

    def _restrict(self, prior: np.ndarray, top_m: int) -> np.ndarray:
        """Keep only the top M priors. Default off: M=50 buys 4 visits per action instead of
        0.16, at the price of making everything outside the top 50 unreachable, so a bad early
        prior locks in. Decided by measuring recall@B both ways, not by intuition."""
        if top_m >= int((prior > 0).sum()):
            return prior
        keep = np.argpartition(-prior, top_m)[:top_m]
        out = np.zeros_like(prior)
        out[keep] = prior[keep]
        total = out.sum()
        return out / total if total > 0 else out

    def rollout_to_terminal(self, state: GameState, node: Node) -> GameState:
        """Descend to terminal by the current prior, so the leaf that gets simulated is a
        complete machine. This is the step that is exact rather than estimated."""
        while not state.is_terminal:
            legal = np.asarray(state.legal_mask(), dtype=bool)
            if not legal.any():
                break
            prior, _ = self.evaluate(state)
            prior = np.where(legal, prior, 0.0)
            total = prior.sum()
            weights = prior / total if total > 0 else legal / legal.sum()
            # The exponential race: one INDEPENDENT uniform per action. A single shared uniform
            # makes u**(1/w) monotone in w, so it would collapse to argmax and the rollout would
            # be deterministic - every iteration reaching the same leaf, and the search learning
            # nothing beyond its own prior.
            generator = np.random.default_rng(self.rng.getrandbits(63))
            keys = generator.random(weights.shape[0]) ** (1.0 / np.maximum(weights, 1e-12))
            action = int(np.argmax(np.where(legal, keys, -np.inf)))
            state = state.advance(action)
        return state

    def reward_of(self, state: GameState, first_action: int) -> float:
        candidate = state.to_candidate(cid=0)
        digest = canonical_hash(candidate)
        cached = self.cache.get(digest)
        if cached is not None:
            return cached
        reward = float(self.oracle(candidate))
        self.calls += 1
        self.cache[digest] = reward
        self.attempts.append((first_action, candidate, reward))
        return reward

    # --- one iteration --------------------------------------------------------------

    def iterate(self, root: Node) -> float:
        path: list[tuple[Node, int]] = []
        node = root
        state = root.state
        first_action = -1

        while True:
            if state.is_terminal or not node.legal.any():
                break
            action = node.best(self.config.c_puct)
            if first_action < 0:
                first_action = action
            path.append((node, action))
            state = state.advance(action)
            child = node.children.get(action)
            if child is None:
                child = self.expand(state)
                node.children[action] = child
                break
            node = child

        # EVALUATE. `v` is logged against the exact result the search then finds, but is given
        # no authority over the backup - point 31 applied inside the tree.
        predicted = float(self.evaluate(state)[1]) if not state.is_terminal else float("nan")
        finished = self.rollout_to_terminal(state, node)
        z = self.reward_of(finished, first_action)
        if not math.isnan(predicted):
            self.value_predictions.append((predicted, z))

        # BACKUP. No sign flip: single player.
        for parent, action in path:
            parent.n += 1
            parent.visits[action] += 1
            parent.totals[action] += z
        root.n += 0
        return z

    # --- driving it -----------------------------------------------------------------

    def run(self, root: Node | None = None, budget: int | None = None) -> SearchResult:
        budget = self.config.simulations if budget is None else budget
        if root is None:
            root = self.expand(
                GameState(
                    self.machine, k=self.config.k, allow_stop=self.config.allow_stop
                ),
                add_noise=True,
            )
        start = self.calls
        # A search on a small machine can exhaust its reachable space and then hit the
        # transposition cache forever, spending no budget and never terminating. Give up after
        # a run of cache-only iterations rather than spinning.
        stalled = 0
        stall_limit = max(32, budget)
        while self.calls - start < budget:
            before = self.calls
            self.iterate(root)
            stalled = 0 if self.calls > before else stalled + 1
            if stalled >= stall_limit or not root.legal.any():
                break
        pi = root.pi(self.config.temperature)
        chosen = (
            int(self.rng.choices(range(len(pi)), weights=pi.tolist(), k=1)[0])
            if pi.sum() > 0
            else int(np.argmax(root.prior))
        )
        return SearchResult(
            root=root,
            pi=pi,
            chosen=chosen,
            simulator_calls=self.calls - start,
            attempts=list(self.attempts),
            value_predictions=list(self.value_predictions),
        )

    def reuse(self, root: Node, action: int) -> Node | None:
        """After playing a move, the chosen child becomes the new root with its statistics
        intact - roughly halving the effective cost of an episode."""
        if not self.config.reuse_subtree:
            return None
        return root.children.get(action)


# --- evaluators ------------------------------------------------------------------------


def uniform_evaluator(machine: Machine) -> Evaluator:
    """The uninformed control (point 32). It ignores the model entirely, which is what makes it
    a baseline, an unbiased calibration sample, **and** the only unbiased training data the
    system produces - three jobs for one 5% expenditure."""

    def evaluate(state: GameState) -> tuple[np.ndarray, float]:
        legal = np.asarray(state.legal_mask(), dtype=float)
        total = legal.sum()
        return (legal / total if total > 0 else legal), 0.5

    return evaluate


def net_evaluator(net, base_graph: Graph, cache_size: int = 512) -> Evaluator:
    """The model. The graph is patched with the state's notes rather than rebuilt, so a search
    costs one 157 ms build and then microseconds per node."""
    memo: dict[tuple, tuple[np.ndarray, float]] = {}

    def evaluate(state: GameState) -> tuple[np.ndarray, float]:
        key = tuple(sorted((p.cell, p.slot) for p in state.placements))
        hit = memo.get(key)
        if hit is not None:
            return hit
        graph = apply_notes(base_graph, state.placements)
        legal = state.legal_mask()
        if not any(legal):
            # A terminal has no legal action; the prior is never read, only `v`.
            legal = [True] + [False] * (len(legal) - 1)
        result = net.priors(graph, legal)
        if len(memo) < cache_size:
            memo[key] = result
        return result

    return evaluate
