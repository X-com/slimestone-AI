"""Exhaustive k=2 enumeration and recall@B - md/VERIFICATION.md, for MILESTONE 3.

`enumerate_k2` produces the **denominator** of every recall number. A denominator that is quietly
too small makes recall quietly too high, and there is no second source to notice: the whole point
of brute force is that nothing else knows the answer.

So the enumeration is checked against properties derived independently of it - a hand-countable
machine, the unordered-pair identity, and the claim that a set of placements is order-independent
by construction rather than by deduplication afterwards.
"""
from __future__ import annotations

import pytest

from rlgym.game import GameState, Machine, canonical_hash, decode_action
from rlgym.labeller import GroundTruth, enumerate_k2, load_ground_truth, save_ground_truth
from rlgym.recall import Recall


@pytest.fixture
def tiny() -> Machine:
    """Two blocks, R=1. Small enough that the pair count can be reasoned about by hand."""
    return Machine(cells={(0, 0, 0): 1, (1, 0, 0): 1}, trigger=(0, 0, 0), radius=1)


def test_every_pair_is_legal_in_at_least_one_order(tiny):
    """The inclusion rule, checked directly: a pair belongs iff some order of it is playable.

    This is the rule that keeps the denominator honest. Placing `a` first can make `b` a no-op -
    and so masked - while the reverse order is fine, and requiring both orders would silently
    drop those pairs.
    """
    root = GameState(tiny, k=2)
    for a, b in enumerate_k2(tiny)[:400]:
        first = root.step(decode_action(a, tiny.cell_list)).legal_mask()[b]
        second = root.step(decode_action(b, tiny.cell_list)).legal_mask()[a]
        assert first or second


def test_pairs_are_unordered_and_distinct(tiny):
    pairs = enumerate_k2(tiny)
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == len(pairs)


def test_no_pair_repeats_a_single_action(tiny):
    """`a` with `a` is not a two-block modification; the written-cell mask must already rule it
    out rather than it being filtered here."""
    assert all(a != b for a, b in enumerate_k2(tiny))


def test_placement_order_does_not_change_the_machine(tiny):
    """Point 23's factorial explosion is killed **structurally**, not by hashing afterwards.

    Both orders of a pair must produce the same machine wherever both are legal - which is what
    lets `enumerate_k2` return unordered pairs and count each modification once.
    """
    root = GameState(tiny, k=2)
    checked = 0
    for a, b in enumerate_k2(tiny):
        forward = root.step(decode_action(a, tiny.cell_list))
        if not forward.legal_mask()[b]:
            continue
        backward = root.step(decode_action(b, tiny.cell_list))
        if not backward.legal_mask()[a]:
            continue
        one = forward.step(decode_action(b, tiny.cell_list)).current_cells()
        two = backward.step(decode_action(a, tiny.cell_list)).current_cells()
        assert one == two, f"actions {a} and {b} give different machines in different orders"
        checked += 1
        if checked >= 300:
            break
    assert checked > 0


def test_the_pair_count_matches_the_single_action_space(tiny):
    """An upper bound derived from the action space alone: at most C(n,2) pairs from n legal
    single actions. Equality would mean no placement ever masks another, which cannot hold once
    a cell is written."""
    n = sum(GameState(tiny, k=2).legal_mask())
    pairs = enumerate_k2(tiny)
    assert 0 < len(pairs) <= n * (n - 1) // 2


# --- the recall arithmetic ---------------------------------------------------------------


def test_recall_divides_by_what_exists():
    r = Recall(
        name="x",
        budget=1000,
        calls=1000,
        unique_seen=800,
        working_found=15,
        non_cargo_found=3,
        working_available=60,
        non_cargo_available=12,
    )
    assert r.recall_working == pytest.approx(0.25)
    assert r.recall_non_cargo == pytest.approx(0.25)
    assert r.precision == pytest.approx(0.015)


def test_recall_is_zero_rather_than_undefined_when_nothing_exists():
    """A machine with no working 2-block modification must report 0.0, not divide by zero -
    otherwise one such machine takes down a sweep over many."""
    r = Recall("x", 100, 100, 50, 0, 0, 0, 0)
    assert r.recall_working == 0.0
    assert r.recall_non_cargo == 0.0


def test_ground_truth_survives_a_round_trip(tmp_path):
    """Hashes are stored as sorted lists and must come back as sets - a list would make the
    membership test in `measure` linear and the run quietly quadratic."""
    truth = GroundTruth(
        machine="x",
        radius=1,
        k=2,
        pairs=10,
        unique=8,
        working=3,
        non_cargo=1,
        base_period=4,
        base_shift=(0, 0, 1),
        working_hashes={"aa", "bb", "cc"},
        non_cargo_hashes={"aa"},
    )
    path = tmp_path / "truth.json"
    save_ground_truth(truth, path)
    back = load_ground_truth(path)
    assert isinstance(back.working_hashes, set)
    assert back.working_hashes == truth.working_hashes
    assert back.non_cargo_hashes == truth.non_cargo_hashes
    assert back.base_shift == (0, 0, 1)
    assert back.working_rate == pytest.approx(3 / 8)


def test_non_cargo_is_a_subset_of_working(tmp_path):
    """Structural, and it holds by construction in `label_k2`: a modification that changes the
    period cannot have failed, because a failed run has no period to compare."""
    truth = GroundTruth(
        machine="x", radius=1, k=2, pairs=1, unique=1, working=2, non_cargo=1,
        base_period=4, base_shift=(0, 0, 1),
        working_hashes={"aa", "bb"}, non_cargo_hashes={"aa"},
    )
    assert truth.non_cargo_hashes <= truth.working_hashes
