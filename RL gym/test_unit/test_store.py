"""Library and attempt log - md/VERIFICATION.md, for Part 6's two destinations.

The single most important rule in this file is that **cargo never becomes a parent**. It is not
a performance concern: cargo accumulates monotonically because nothing removes it, so one leak
compounds into a library where most of every machine is dead weight, and every symptom of that
(bigger inputs, slower simulation, noisier features) looks like something else.

The second is that library size is not progress. Nothing here asserts that it grows.
"""
from __future__ import annotations

import pytest

from rlgym.game import canonical_hash
from rlgym.store import Attempt, Store


def candidate(n_blocks: int = 2, offset: int = 0):
    return {
        "id": 0,
        "trigger": {"x": offset, "y": 0, "z": 0},
        "blocks": [
            {"x": x + offset, "y": 0, "z": 0, "state": 1} for x in range(n_blocks)
        ],
    }


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "library")


def test_cargo_is_refused_from_the_library(store):
    """Part 6's table: generation 10 of a cargo-admitting library carries 20 dead blocks for 10
    useful ones. The refusal is the only thing preventing it."""
    store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    parent = canonical_hash(candidate(2))
    assert store.admit(candidate(3), parent, 1, 4, (0, 0, 1), cargo=True) is None
    assert len(store) == 1


def test_a_non_cargo_discovery_is_admitted_as_a_child(store):
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    entry = store.admit(candidate(3), base.digest, 1, 8, (0, 0, 2), cargo=False)
    assert entry is not None
    assert entry.parent == base.digest
    assert entry.generation == 1
    assert store.entries[base.digest].descendants == 1


def test_cargo_may_be_admitted_only_when_explicitly_allowed(store):
    """The escape hatch exists so the decision can be *measured*, not so it can be forgotten."""
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    assert store.admit(candidate(3), base.digest, 1, 4, (0, 0, 1), cargo=True) is None
    assert (
        store.admit(candidate(3), base.digest, 1, 4, (0, 0, 1), cargo=True, allow_cargo=True)
        is not None
    )


def test_the_same_machine_is_never_admitted_twice(store):
    """`canonical_hash` is translation-invariant, so the same structure at a different position
    is the same library entry - which is also why library size is not a discovery count.

    The trigger moves with the blocks. It has to: the hash normalises the trigger against the
    same corner, so a machine whose trigger sits on a different block is a different machine,
    not a translation of this one.
    """
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    first = store.admit(candidate(3), base.digest, 1, 8, (0, 0, 2), cargo=False)
    again = store.admit(candidate(3, offset=17), base.digest, 1, 8, (0, 0, 2), cargo=False)
    assert first is not None and again is None
    assert len(store) == 2


def test_every_attempt_is_logged_including_the_failures(store):
    """Part 6: a search spends its whole budget and ends on one move. The other outcomes are
    true labels and cost nothing extra to keep - training on them is the point."""
    for index in range(5):
        store.record(
            Attempt(
                digest=f"d{index}",
                parent="p",
                round_index=1,
                source="top",
                reward=float(index % 2),
                working=bool(index % 2),
                cargo=False,
                period=4,
                shift=(0, 0, 1),
                blocks=3,
            )
        )
    rows = store.attempts()
    assert len(rows) == 5
    assert sum(1 for row in rows if not row["working"]) == 3


def test_the_log_survives_a_reopen(store, tmp_path):
    """Append-only, so a crashed run leaves valid data rather than a truncated file."""
    store.record(
        Attempt("d0", "p", 1, "top", 1.0, True, False, 4, (0, 0, 1), 3)
    )
    store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    store.save()

    reopened = Store(tmp_path / "library")
    assert len(reopened) == 1
    assert reopened.seen["d0"] == 1.0


def test_variety_reports_concentration_not_just_size(store):
    """`DEFERRED.md`'s binning problem, measured. Descendants piling onto one root is what the
    self-selected training distribution looks like from outside, and size alone hides it."""
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    for blocks in (3, 4, 5):
        store.admit(candidate(blocks), base.digest, 1, 8, (0, 0, 2), cargo=False)
    report = store.variety()
    assert report["size"] == 4
    assert report["roots"] == 1
    assert report["max_descendants_of_one_root"] == 4
    assert report["by_generation"] == {0: 1, 1: 3}
