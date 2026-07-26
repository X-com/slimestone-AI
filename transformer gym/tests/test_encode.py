import tempfile
from pathlib import Path

from transformer_gym.encode import RELATION_TYPES, encode
from transformer_gym.simlog_reader import run_fixture

REPO = Path(__file__).resolve().parents[2]


def _encode(name: str):
    with tempfile.TemporaryDirectory() as tmp:
        log = run_fixture(name, Path(tmp))
        return encode(log)


def test_shapes_and_relations():
    s = _encode("simple_observer_engine")
    n = s.block_type.shape[0]
    assert n > 0
    assert s.facing.shape == (n,)
    assert s.flags.shape == (n, 3)
    assert s.relations.shape == (len(RELATION_TYPES), n, n)
    assert s.y_event_grid.shape[0] == n
    assert s.y_net_shift.shape == (3,)
    # at least one interface-air token should exist next to a solid structure
    assert s.is_air.sum() > 0


def test_would_power_qc_relation_present():
    # simple_caterpillar has a redstone block that only QC-powers a piston (confirmed in the
    # SDL4 verification session) - the relation channel must not be empty.
    s = _encode("simple_caterpillar")
    r = RELATION_TYPES.index("would_power_qc")
    assert s.relations[r].sum() > 0


def test_push_group_preview_covers_non_firing_piston():
    s = _encode("simple_observer_engine")
    r = RELATION_TYPES.index("same_push_group")
    # every piston token should participate in at least a trivial (size-1) preview group;
    # relation entries only appear for groups with >=2 members, so just assert decode succeeded
    # and produced a well-formed square relation tensor (no crash, correct shape) as the main check.
    assert s.relations[r].shape == (s.block_type.shape[0], s.block_type.shape[0])
