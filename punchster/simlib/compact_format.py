"""Compact binary encoding for flying-machine candidates - a fast, small alternative to the
JSON candidate format, used for the GA<->cpp wire protocol and for compact-working/flyers.data.

Record layout (little-endian, self-delimiting - no outer framing needed):

    int32   id
    int32   trigger_x, trigger_y, trigger_z
    uint32  block_count
    block_count x { int32 x, int32 y, int32 z, uint32 state }

Multiple records can simply be concatenated (a file, or a pipe) and read back by looping
decode_candidate() until EOF - each record's block_count says exactly how many bytes follow it.

data/working/*.json (JSON, one file per candidate) is untouched by this module and stays the
seed-input format; this is a separate, additional format.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

Candidate = dict[str, Any]

_HEADER = struct.Struct("<iiiiI")  # id, trigger_x, trigger_y, trigger_z, block_count
_BLOCK = struct.Struct("<iiiI")  # x, y, z, state


def encode_candidate(candidate: Candidate) -> bytes:
    trigger = candidate["trigger"]
    blocks = candidate["blocks"]
    out = bytearray(_HEADER.size + _BLOCK.size * len(blocks))
    _HEADER.pack_into(out, 0, candidate["id"], trigger["x"], trigger["y"], trigger["z"], len(blocks))
    offset = _HEADER.size
    for block in blocks:
        _BLOCK.pack_into(out, offset, block["x"], block["y"], block["z"], block["state"])
        offset += _BLOCK.size
    return bytes(out)


def decode_candidate(stream: BinaryIO) -> Candidate | None:
    """Reads one record from stream. Returns None on a clean EOF (nothing read at all);
    raises EOFError if the stream is cut off partway through a record."""
    header = stream.read(_HEADER.size)
    if not header:
        return None
    if len(header) < _HEADER.size:
        raise EOFError("truncated compact-format header")

    cid, tx, ty, tz, block_count = _HEADER.unpack(header)
    blocks: list[dict[str, int]] = []
    for _ in range(block_count):
        raw = stream.read(_BLOCK.size)
        if len(raw) < _BLOCK.size:
            raise EOFError("truncated compact-format block record")
        x, y, z, state = _BLOCK.unpack(raw)
        blocks.append({"x": x, "y": y, "z": z, "state": state})

    return {"id": cid, "trigger": {"x": tx, "y": ty, "z": tz}, "blocks": blocks}


def read_compact_file(path: str | Path) -> list[Candidate]:
    """Reads every record in path, in order. Returns an empty list if the file doesn't exist
    yet (the natural state before anything has been discovered)."""
    path = Path(path)
    if not path.exists():
        return []
    candidates: list[Candidate] = []
    with path.open("rb") as handle:
        while True:
            candidate = decode_candidate(handle)
            if candidate is None:
                break
            candidates.append(candidate)
    return candidates


def append_candidate(path: str | Path, candidate: Candidate) -> None:
    """Appends one record onto path, creating it (and its parent directory) if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(encode_candidate(candidate))
        handle.flush()


def json_file_to_compact(json_path: str | Path, compact_path: str | Path) -> int:
    """Converts a JSON/JSON-lines candidate file into a compact-format file (overwritten, not
    appended). Returns the number of candidates written. Lets existing JSON fixtures (e.g. for
    cpp's file mode) be migrated instead of hand-rewritten."""
    from .candidate_io import load_candidates_from_file

    candidates = load_candidates_from_file(json_path)
    compact_path = Path(compact_path)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    with compact_path.open("wb") as handle:
        for candidate in candidates:
            handle.write(encode_candidate(candidate))
    return len(candidates)


def compact_file_to_json(compact_path: str | Path, json_path: str | Path) -> int:
    """Converts a compact-format file into a JSON-lines file, for inspecting/debugging
    contents (e.g. compact-working/flyers.data) in a human-readable form. Returns the number
    of candidates written."""
    import json

    candidates = read_compact_file(compact_path)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, separators=(",", ":")) + "\n")
    return len(candidates)
