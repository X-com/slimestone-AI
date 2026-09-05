"""Library and attempt log - md/VERIFICATION.md, for Part 6's two destinations.

**Nothing is refused for what it does to the flight any more.** The cargo test asked whether a
modification changed the period or shift, which cannot tell a useless block from one placed for
looks, as a floor, or for any other purpose in the game. Waste is now prevented by *removal*
rather than *refusal*: `function.trim` strips redundant blocks and the stripped machine is what
gets admitted, so a genuine discovery carrying a decorative block is kept instead of thrown away.

What this file still pins: a machine enters once, a lineage is tracked, every attempt is logged
including the failures, and **library size is not progress** - nothing here asserts that it grows.
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


def test_a_discovery_that_leaves_the_flight_unchanged_is_still_admitted(store):
    """The behaviour the cargo filter used to prevent, now deliberate.

    Same period and same shift as the parent - the old test called that cargo and refused it.
    A block placed for looks, or as a floor, produces exactly this signature, and refusing it
    threw away the whole discovery to avoid one spare block. Redundant blocks are stripped
    upstream by `function.trim`; the store no longer judges.
    """
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    entry = store.admit(candidate(3), base.digest, 1, 4, (0, 0, 1))
    assert entry is not None
    assert len(store) == 2


def test_a_discovery_is_admitted_as_a_child(store):
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    entry = store.admit(candidate(3), base.digest, 1, 8, (0, 0, 2))
    assert entry is not None
    assert entry.parent == base.digest
    assert entry.generation == 1
    assert store.entries[base.digest].descendants == 1


def test_an_entry_remembers_what_was_stripped_from_it(store):
    """The replacement for the cargo flag: how much waste was removed before admission, and
    whether the block the model actually chose survived."""
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    entry = store.admit(
        candidate(3), base.digest, 1, 8, (0, 0, 2),
        redundant_removed=2, added_is_load_bearing=False,
    )
    assert entry.redundant_removed == 2
    assert entry.added_is_load_bearing is False


def test_the_same_machine_is_never_admitted_twice(store):
    """`canonical_hash` is translation-invariant, so the same structure at a different position
    is the same library entry - which is also why library size is not a discovery count.

    The trigger moves with the blocks. It has to: the hash normalises the trigger against the
    same corner, so a machine whose trigger sits on a different block is a different machine,
    not a translation of this one.
    """
    base = store.seed("base", candidate(2), period=4, shift=(0, 0, 1))
    first = store.admit(candidate(3), base.digest, 1, 8, (0, 0, 2))
    again = store.admit(candidate(3, offset=17), base.digest, 1, 8, (0, 0, 2))
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
        Attempt("d0", "p", 1, "top", 1.0, True, 4, (0, 0, 1), 3)
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
        store.admit(candidate(blocks), base.digest, 1, 8, (0, 0, 2))
    report = store.variety()
    assert report["size"] == 4
    assert report["roots"] == 1
    assert report["max_descendants_of_one_root"] == 4
    assert report["by_generation"] == {0: 1, 1: 3}
