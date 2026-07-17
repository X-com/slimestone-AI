"""The seam between the fixed training base (train_loop.py, policy.py) and whatever training
goal is currently plugged in (rl_ml/tasks/*.py). Swapping the training objective means writing a
new module that implements this Protocol and pointing main_rl.py at it - train_loop.py and
policy.py never need to change.
"""
from __future__ import annotations

import random
from typing import Any, Protocol

Candidate = dict[str, Any]


class Task(Protocol):
    def sample_contexts(self, base_machines: list[Candidate], rng: random.Random, count: int) -> list[Any]:
        """Draw `count` (machine, ...) contexts to consider this training iteration."""
        ...

    def features(self, context: Any) -> list[float]:
        """Turn a context into the shared feature vector the policy scores."""
        ...

    def build_candidate(self, context: Any, action: bool, candidate_id: int) -> Candidate | None:
        """Build the candidate to simulate for (context, action), or None if this action needs
        no simulation (e.g. "don't act" - reward_of(action, None) is used directly instead)."""
        ...

    def reward_of(self, action: bool, result: dict[str, Any] | None) -> float:
        """Reward for one episode. `result` is the simulator's result dict, or None when
        build_candidate returned None for this action."""
        ...
