"""Pure-Python REINFORCE over a small shared linear-in-features Bernoulli policy. No numpy/torch -
a handful of weights don't need a tensor library, and the sibling genetic algorithm project is
zero-dependency by the same reasoning.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def _sigmoid(x: float) -> float:
    # Numerically stable form - avoids OverflowError from math.exp on a very negative/positive x.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class SharedLinearPolicy:
    def __init__(self, feature_count: int, learning_rate: float = 0.1, task_name: str = "unknown") -> None:
        self.weights: list[float] = [0.0] * feature_count
        self.learning_rate = learning_rate
        self.task_name = task_name
        self.iteration = 0
        self._baseline = 0.0
        self._baseline_count = 0

    def _logit(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, features))

    def probability(self, features: list[float]) -> float:
        return _sigmoid(self._logit(features))

    def sample(self, features: list[float], rng: random.Random) -> bool:
        return rng.random() < self.probability(features)

    def update(self, episodes: list[tuple[list[float], bool, float]]) -> None:
        """episodes: list of (features, action, reward). Applies one REINFORCE step per episode,
        using a running mean of all rewards ever seen as the baseline."""
        for features, action, reward in episodes:
            p = self.probability(features)
            grad_scale = self.learning_rate * (reward - self._baseline) * ((1.0 if action else 0.0) - p)
            for i, feature in enumerate(features):
                self.weights[i] += grad_scale * feature

            self._baseline_count += 1
            self._baseline += (reward - self._baseline) / self._baseline_count

        self.iteration += 1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "task": self.task_name,
            "feature_count": len(self.weights),
            "weights": self.weights,
            "iteration": self.iteration,
            "learning_rate": self.learning_rate,
            "baseline": self._baseline,
            "baseline_count": self._baseline_count,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SharedLinearPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            feature_count=payload["feature_count"],
            learning_rate=payload["learning_rate"],
            task_name=payload["task"],
        )
        policy.weights = list(payload["weights"])
        policy.iteration = payload["iteration"]
        policy._baseline = payload.get("baseline", 0.0)
        policy._baseline_count = payload.get("baseline_count", 0)
        return policy
