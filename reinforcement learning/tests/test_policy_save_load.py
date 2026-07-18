from __future__ import annotations

import random

from rl_ml.policy import SharedLinearPolicy


def test_sample_is_deterministic_given_a_seeded_rng():
    policy = SharedLinearPolicy(feature_count=3, learning_rate=0.1, task_name="test")
    policy.weights = [0.2, -0.1, 0.05]
    features = [1.0, 0.5, -1.0]

    result_a = policy.sample(features, random.Random(42))
    result_b = policy.sample(features, random.Random(42))

    assert result_a == result_b


def test_greedy_is_deterministic_and_matches_probability_threshold():
    policy = SharedLinearPolicy(feature_count=1, learning_rate=0.1, task_name="test")

    policy.weights = [5.0]  # probability(features=[1.0]) is clearly > 0.5
    assert policy.greedy([1.0]) is True

    policy.weights = [-5.0]  # clearly < 0.5
    assert policy.greedy([1.0]) is False

    # unlike sample(), greedy() takes no rng and must be stable across repeated calls.
    policy.weights = [0.3]
    results = {policy.greedy([1.0]) for _ in range(5)}
    assert len(results) == 1


def test_update_moves_weights_toward_rewarded_actions():
    policy = SharedLinearPolicy(feature_count=1, learning_rate=0.5, task_name="test")

    policy.update([([1.0], True, 1.0)])

    assert policy.weights[0] > 0.0
    assert policy.iteration == 1


def test_update_normalizes_advantages_across_the_batch():
    # Two episodes with the same relative ordering but a batch shifted/scaled in reward - after
    # advantage normalization, the resulting weight update should be identical either way.
    policy_a = SharedLinearPolicy(feature_count=1, learning_rate=0.5, task_name="test", entropy_coef=0.0)
    policy_a.update([([1.0], True, 1.0), ([1.0], False, -1.0)])

    policy_b = SharedLinearPolicy(feature_count=1, learning_rate=0.5, task_name="test", entropy_coef=0.0)
    policy_b.update([([1.0], True, 100.0), ([1.0], False, -100.0)])

    assert policy_a.weights[0] == policy_b.weights[0]


def test_entropy_bonus_pulls_a_confident_weight_toward_zero_on_a_neutral_batch():
    policy = SharedLinearPolicy(feature_count=1, learning_rate=0.1, task_name="test", entropy_coef=0.5)
    policy.weights = [5.0]  # already near-certain True

    # Single-episode batch -> raw advantage normalizes to 0 (std fallback), so only entropy acts.
    policy.update([([1.0], True, 0.0)])

    assert policy.weights[0] < 5.0


def test_save_load_round_trips_weights_and_metadata(tmp_path):
    policy = SharedLinearPolicy(feature_count=2, learning_rate=0.3, task_name="dummy")
    policy.update([([1.0, 0.0], True, 1.0), ([0.0, 1.0], False, 0.0)])
    path = tmp_path / "checkpoint.json"

    policy.save(path)
    loaded = SharedLinearPolicy.load(path)

    assert loaded.weights == policy.weights
    assert loaded.iteration == policy.iteration
    assert loaded.task_name == policy.task_name
    assert loaded.learning_rate == policy.learning_rate
    assert loaded._baseline == policy._baseline
    assert loaded._baseline_count == policy._baseline_count
