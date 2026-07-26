"""The fixed held-out fixture list: never used as a mutation/crop/splice seed by any generator,
and never trained on directly - the one true "did this generalize to real machines" instrument
(transformer_gym/eval.py's held-out curve reads this same list). Spans the small hand fixtures
(one per named mechanic) plus a spread of test_flyer_* sizes/outcomes, deliberately excluding the
large schematic-scale machines (those are only ever used as gen_perturb crop *sources*, and a
crop source being held-out-eligible would be a different, weaker guarantee than "never seen").
"""
from __future__ import annotations

HOLDOUT_FIXTURES: tuple[str, ...] = (
    "simple_caterpillar",
    "simple_machine1",
    "simple_machine2",
    "simple_machine3",
    "simple_no_sticky_loop",
    "simple_observer_engine",
    "simple_upwards_engine",
    "test_flyer_1",
    "test_flyer_5",
    "test_flyer_9",
    "test_flyer_12_doesnt_loop",
    "test_flyer_16",
    "test_flyer_19_doesnt_loop",
    "test_flyer_23",
    "test_flyer_27_doesnt_loop",
    "test_flyer_31",
    "test_flyer_34",
)
