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


def test_update_moves_weights_toward_rewarded_actions():
    policy = SharedLinearPolicy(feature_count=1, learning_rate=0.5, task_name="test")

    policy.update([([1.0], True, 1.0)])

    assert policy.weights[0] > 0.0
    assert policy.iteration == 1


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
