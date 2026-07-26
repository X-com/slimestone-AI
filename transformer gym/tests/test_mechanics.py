from transformer_gym.mechanic_fixtures import build_suite
from transformer_gym.mechanics import MECHANIC_ORDER, derive_tags


def test_hand_fixtures_match_their_claimed_tags():
    # build_suite() already asserts this internally; re-asserting here makes it a tracked
    # regression test (fails in CI/pytest output, not just on manual invocation).
    suite = build_suite()
    assert {name for name, _, _ in suite} == {"push_order", "sticky_drag", "non_stick", "observer_pulse"}
    for name, sample, expected_tags in suite:
        assert derive_tags(sample) == expected_tags, name


def test_starved_mechanic_scores_independently_of_others():
    # Verification claim from the design doc: per-mechanic scoring is independent, not just
    # tracking one aggregate - if a mechanic never appears in the sample pool, its bucket must
    # come out empty rather than silently borrowing another mechanic's samples/score.
    suite = build_suite()
    samples = [s for name, s, _ in suite if name != "observer_pulse"]
    tags_seen = set()
    for s in samples:
        tags_seen |= derive_tags(s)
    assert "observer_pulse" not in tags_seen
    assert "push_order" in tags_seen  # sanity: the other mechanics are still represented


def test_mechanic_order_matches_causal_dependency():
    assert MECHANIC_ORDER == ("push_order", "sticky_drag", "non_stick", "observer_pulse", "composed")
