"""The benchmark ladder - md/BENCHMARKS.md.

These tests are about the *measuring instrument*, not the model. A ranking metric that is subtly
wrong is worse than none: it produces confident numbers that move for the wrong reasons, and
every conclusion drawn from them is unsound.
"""
from __future__ import annotations

import random

import pytest

from rlgym.baselines import (
    Evaluation,
    _auc,
    _rank,
    cell_entry_scores,
    cell_scores,
    entry_scores,
    evaluate,
    fit_entry_rates,
    uniform_scores,
)
from rlgym.labeller import Label, LabelSet


def _label(action: int, cell, entry: str, reward: float, cargo: bool = False) -> Label:
    return Label(
        action=action,
        cell=cell,
        slot=0,
        entry=entry,
        reward=reward,
        period=10,
        shift=(1, 0, 0),
        working=reward > 0,
        maybe_cargo=cargo,
    )


def _labelset(labels: list[Label], machine: str = "toy") -> LabelSet:
    return LabelSet(
        machine=machine,
        radius=1,
        k=1,
        action_slots=len(labels) + 1,
        legal_actions=len(labels),
        simulated=len(labels),
        duplicates=0,
        working=sum(1 for label in labels if label.reward > 0),
        maybe_cargo=sum(1 for label in labels if label.maybe_cargo),
        labels=labels,
    )


def test_auc_is_one_for_a_perfect_ranking():
    ranked = [_label(0, (0, 0, 0), "a", 1.0), _label(1, (0, 0, 0), "a", 0.0)]
    assert _auc(ranked) == 1.0


def test_auc_is_zero_for_an_inverted_ranking():
    ranked = [_label(0, (0, 0, 0), "a", 0.0), _label(1, (0, 0, 0), "a", 1.0)]
    assert _auc(ranked) == 0.0


def test_auc_is_half_for_a_symmetric_ranking():
    """One positive at each end: two of the four pairs are concordant, two are not."""
    ranked = [
        _label(0, (0, 0, 0), "a", 1.0),
        _label(1, (0, 0, 0), "a", 0.0),
        _label(2, (0, 0, 0), "a", 0.0),
        _label(3, (0, 0, 0), "a", 1.0),
    ]
    assert _auc(ranked) == pytest.approx(0.5)


def test_auc_counts_pairs_not_positions():
    """Alternating pos/neg is NOT 0.5 - it is 0.75, because three of the four
    (positive, negative) pairs are correctly ordered. Pinned because getting this wrong is how a
    ranking metric ends up quietly reporting the wrong thing."""
    ranked = [
        _label(0, (0, 0, 0), "a", 1.0),
        _label(1, (0, 0, 0), "a", 0.0),
        _label(2, (0, 0, 0), "a", 1.0),
        _label(3, (0, 0, 0), "a", 0.0),
    ]
    assert _auc(ranked) == pytest.approx(0.75)


def test_auc_is_defined_when_one_class_is_absent():
    """A machine where nothing works, or everything does, must not divide by zero."""
    assert _auc([_label(0, (0, 0, 0), "a", 0.0)]) == 0.5
    assert _auc([_label(0, (0, 0, 0), "a", 1.0)]) == 0.5


def test_ties_are_broken_randomly_not_by_action_index():
    """The trap this guards: actions are enumerated sorted by cell, and cells are strongly
    predictive (Milestone 1). A ranker that scores everything equally would inherit that order
    and score far above chance while having learned nothing."""
    labels = [_label(i, (i, 0, 0), "a", 1.0 if i < 20 else 0.0) for i in range(100)]
    scores = {label.action: 0.0 for label in labels}
    first = [label.action for label in _rank(labels, scores, random.Random(1))]
    second = [label.action for label in _rank(labels, scores, random.Random(2))]
    assert first != second
    assert sorted(first) == sorted(second)


def test_uniform_baseline_scores_near_the_base_rate():
    """With 30% working and no signal, precision at any budget should sit near 30%."""
    labels = [_label(i, (i, 0, 0), "a", 1.0 if i % 10 < 3 else 0.0) for i in range(1000)]
    labelset = _labelset(labels)
    results = [
        evaluate("u", labelset, uniform_scores(labelset), budget=100, seed=seed).precision
        for seed in range(5)
    ]
    assert 0.2 < sum(results) / len(results) < 0.4


def test_perfect_memorisation_scores_one():
    labels = [_label(i, (i, 0, 0), "a", 1.0 if i < 50 else 0.0) for i in range(200)]
    labelset = _labelset(labels)
    result = evaluate("3", labelset, cell_entry_scores(labelset), budget=50)
    assert result.precision == 1.0
    assert result.auc == 1.0


def test_entry_rates_are_pooled_not_averaged():
    """A 900-action machine and a 100-action machine must contribute in proportion to the
    evidence they carry, not equally."""
    big = _labelset([_label(i, (0, 0, 0), "glass", 1.0) for i in range(900)], "big")
    small = _labelset([_label(i, (0, 0, 0), "glass", 0.0) for i in range(100)], "small")
    rates = fit_entry_rates([big, small])
    assert rates["glass"] == pytest.approx(0.9)


def test_entry_scores_handle_an_unseen_entry():
    """A held-out machine can contain a palette entry no training machine used. That must score
    zero, not raise."""
    labels = [_label(0, (0, 0, 0), "never_seen", 1.0)]
    labelset = _labelset(labels)
    scores = entry_scores(labelset, {"glass": 0.5})
    assert scores[0] == 0.0


def test_cell_baseline_separates_permissive_from_hostile_cells():
    """The structure Milestone 1 found: some cells accept almost anything, others almost
    nothing. A per-cell table should rank them apart perfectly."""
    labels = []
    action = 0
    for cell_index in range(10):
        permissive = cell_index < 4
        for _ in range(20):
            labels.append(
                _label(action, (cell_index, 0, 0), "x", 1.0 if permissive else 0.0)
            )
            action += 1
    labelset = _labelset(labels)
    result = evaluate("2", labelset, cell_scores(labelset), budget=80)
    assert result.precision == 1.0
    assert result.auc == 1.0


def test_duplicates_are_excluded_from_scoring():
    """A duplicate is the same machine reached twice. Counting it twice would inflate whichever
    ranker happened to surface it."""
    labels = [
        _label(0, (0, 0, 0), "a", 1.0),
        _label(1, (0, 0, 0), "a", 1.0),
    ]
    labels[1] = Label(**{**labels[1].__dict__, "duplicate_of": 0})
    labelset = _labelset(labels)
    result = evaluate("u", labelset, uniform_scores(labelset), budget=10)
    assert result.budget == 1
    assert result.working_available == 1


def test_recall_and_precision_are_distinct():
    """20 working actions, budget 10: precision can be 1.0 while recall is only 0.5."""
    labels = [_label(i, (0, 0, 0), "a", 1.0 if i < 20 else 0.0) for i in range(100)]
    labelset = _labelset(labels)
    result = evaluate("3", labelset, cell_entry_scores(labelset), budget=10)
    assert result.precision == 1.0
    assert result.recall == pytest.approx(0.5)


def test_budget_larger_than_the_action_set_is_clamped():
    labels = [_label(i, (0, 0, 0), "a", 1.0) for i in range(5)]
    labelset = _labelset(labels)
    result = evaluate("u", labelset, uniform_scores(labelset), budget=100)
    assert result.budget == 5
    assert result.precision == 1.0
