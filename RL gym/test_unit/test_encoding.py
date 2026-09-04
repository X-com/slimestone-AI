"""Candidate serialisation - the wire protocol the C++ side reads in readCandidateCompact.

The expected byte layout is spelled out here with struct literals rather than by calling
encode_candidate, so an error in the encoder cannot validate itself.
"""
from __future__ import annotations

import io
import struct

import pytest

from rlgym.game import Machine, decode_candidate, encode_candidate, map_to_blocks


def test_header_layout_is_id_trigger_count():
    """int32 id | int32 trigger x,y,z | uint32 block_count, little-endian."""
    candidate = {"id": 7, "trigger": {"x": 1, "y": 2, "z": 3}, "blocks": []}
    assert encode_candidate(candidate) == struct.pack("<iiiiI", 7, 1, 2, 3, 0)


def test_block_layout_is_xyz_state():
    candidate = {
        "id": 0,
        "trigger": {"x": 0, "y": 0, "z": 0},
        "blocks": [{"x": 4, "y": 5, "z": 6, "state": 2589}],
    }
    expected = struct.pack("<iiiiI", 0, 0, 0, 0, 1) + struct.pack("<iiiI", 4, 5, 6, 2589)
    assert encode_candidate(candidate) == expected


def test_record_size_is_header_plus_blocks():
    for count in (0, 1, 6, 100):
        candidate = {
            "id": 0,
            "trigger": {"x": 0, "y": 0, "z": 0},
            "blocks": [{"x": i, "y": 0, "z": 0, "state": 1} for i in range(count)],
        }
        assert len(encode_candidate(candidate)) == 20 + 16 * count


def test_round_trip(engine_candidate):
    candidate = Machine.from_candidate(engine_candidate).to_candidate(cid=42)
    stream = io.BytesIO(encode_candidate(candidate))
    assert decode_candidate(stream) == candidate


def test_round_trip_preserves_trigger(engine_candidate):
    candidate = Machine.from_candidate(engine_candidate).to_candidate(cid=0)
    decoded = decode_candidate(io.BytesIO(encode_candidate(candidate)))
    assert decoded["trigger"] == engine_candidate["trigger"]


def test_records_concatenate_self_delimiting(engine_candidate):
    """Multiple records simply concatenate - block_count says how many bytes follow, so no
    outer framing is needed. This is what lets a whole batch be one stdin write."""
    machine = Machine.from_candidate(engine_candidate)
    candidates = [machine.to_candidate(cid=i) for i in range(3)]
    stream = io.BytesIO(b"".join(encode_candidate(c) for c in candidates))
    assert [decode_candidate(stream) for _ in range(3)] == candidates
    assert decode_candidate(stream) is None  # clean EOF


def test_truncated_header_raises():
    with pytest.raises(EOFError):
        decode_candidate(io.BytesIO(b"\x01\x02\x03"))


def test_truncated_block_raises():
    payload = struct.pack("<iiiiI", 0, 0, 0, 0, 2) + struct.pack("<iiiI", 1, 1, 1, 1)
    with pytest.raises(EOFError):
        decode_candidate(io.BytesIO(payload))


def test_state_matches_the_fixture_file_convention(engine_candidate):
    """The fixtures store state as id | meta<<8, so the whole six-block machine can be written
    out by hand: sticky piston facing west (29 | 4<<8 = 1053), north-facing observer
    (218 | 2<<8 = 730), south-facing observer (218 | 3<<8 = 986), piston facing east
    (33 | 5<<8 = 1313) and two slime blocks (165)."""
    states = sorted({b["state"] for b in engine_candidate["blocks"]})
    assert states == [165, 730, 986, 1053, 1313]


def test_map_to_blocks_is_sorted_and_total(engine):
    blocks = map_to_blocks(engine.cells)
    keys = [(b["x"], b["y"], b["z"]) for b in blocks]
    assert keys == sorted(keys)
    assert len(blocks) == len(engine.cells)
