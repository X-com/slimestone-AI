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


def _entropy_gradient(p: float) -> float:
    """d/dz [-p*ln(p) - (1-p)*ln(1-p)] where z is the logit and p = sigmoid(z). Pushes the logit
    toward 0 (p toward 0.5), i.e. more exploration - added to the policy gradient so probability
    mass doesn't collapse to a hard 0/1 as weights sharpen."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return p * (1.0 - p) * math.log((1.0 - p) / p)


class SharedLinearPolicy:
    def __init__(
        self,
        feature_count: int,
        learning_rate: float = 0.1,
        task_name: str = "unknown",
        entropy_coef: float = 0.01,
        min_probability: float = 0.02,
    ) -> None:
        self.weights: list[float] = [0.0] * feature_count
        self.learning_rate = learning_rate
        self.task_name = task_name
        self.entropy_coef = entropy_coef
        # Floor/ceiling on the sampled probability, applied after the sigmoid - a reward function
        # that's negative in expectation for one action (e.g. most attachment attempts break the
        # machine) drives that action's logit further negative every step, and once sigmoid(logit)
        # underflows toward 0 BOTH the policy gradient and the entropy bonus vanish with it (each
        # scales with p or p*(1-p)) - a numerical trap with no way back. Clamping guarantees a
        # non-zero sampling/gradient signal no matter how extreme the weights get, same idea as
        # epsilon-greedy.
        self.min_probability = min_probability
        self.iteration = 0
        self._baseline = 0.0
        self._baseline_count = 0

    def _logit(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, features))

    def probability(self, features: list[float]) -> float:
        p = _sigmoid(self._logit(features))
        return min(max(p, self.min_probability), 1.0 - self.min_probability)

    def sample(self, features: list[float], rng: random.Random) -> bool:
        return rng.random() < self.probability(features)

    def greedy(self, features: list[float]) -> bool:
        """Deterministic action - the policy's best guess, not a stochastic draw. Used for
        evaluating/comparing a trained policy rather than for exploration during training."""
        return self.probability(features) > 0.5

    def update(self, episodes: list[tuple[list[float], bool, float]]) -> None:
        """episodes: list of (features, action, reward). Applies one REINFORCE step per episode:

        - Advantage = reward - running-mean baseline (as before), then normalized to zero-mean/
          unit-std across THIS batch (PPO-style advantage normalization) - keeps the gradient
          magnitude consistent regardless of how noisy or skewed a given batch's rewards are,
          instead of the raw baseline-centered value swinging the step size around.
        - An entropy bonus (_entropy_gradient) is added to the policy gradient so the action
          probability doesn't collapse to a hard 0/1 as weights sharpen, keeping some exploration
          alive late in training.
        """
        if not episodes:
            self.iteration += 1
            return

        baseline = self._baseline
        raw_advantages = [reward - baseline for _, _, reward in episodes]
        mean_advantage = sum(raw_advantages) / len(raw_advantages)
        variance = sum((a - mean_advantage) ** 2 for a in raw_advantages) / len(raw_advantages)
        std = math.sqrt(variance)
        norm = std if std > 1e-6 else 1.0

        for (features, action, reward), raw_advantage in zip(episodes, raw_advantages):
            advantage = raw_advantage / norm
            p = self.probability(features)
            policy_grad = advantage * ((1.0 if action else 0.0) - p)
            grad_scale = self.learning_rate * (policy_grad + self.entropy_coef * _entropy_gradient(p))
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
            "entropy_coef": self.entropy_coef,
            "min_probability": self.min_probability,
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
            entropy_coef=payload.get("entropy_coef", 0.01),
            min_probability=payload.get("min_probability", 0.02),
        )
        policy.weights = list(payload["weights"])
        policy.iteration = payload["iteration"]
        policy._baseline = payload.get("baseline", 0.0)
        policy._baseline_count = payload.get("baseline_count", 0)
        return policy
