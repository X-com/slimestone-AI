"""Convert genetic-ml JSON candidates into legacy MCEdit/Schematica ``.schematic`` files.

genetic-ml candidates store blocks as ``{"x", "y", "z", "state"}`` where
``state = block_id | (meta << 8)`` - the same packing used throughout this project's
C++/Java simulators (see genetic_ml/candidate_io.py). This is a different, simpler
scheme than cli/formats/legacy_schematic.py's MachineIR palette, so rather than bridge
through MachineIR this writes the .schematic NBT directly.

genetic-ml has zero third-party dependencies (see pyproject.toml); this module keeps
it that way with a small hand-rolled NBT writer instead of pulling in nbtlib/numpy.
"""
from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path
from typing import Any

from genetic_ml.candidate_io import load_candidates_from_file, validate_candidate

Candidate = dict[str, Any]
Pos = tuple[int, int, int]

# Legacy block ids (matches the convention used by the cpp/extract/mcp1122 simulators).
MARKER_BLOCK_ID = 95
MARKER_META = 14
TRIGGERABLE_IDS = {33, 29, 218}  # piston, sticky_piston, observer

# down, up, north, south, west, east - Minecraft's canonical neighbor order.
_NEIGHBOR_OFFSETS: tuple[Pos, ...] = (
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
)


def _unpack_state(state: int) -> tuple[int, int]:
    return state & 0xFF, (state >> 8) & 0xFF


def _find_marker_position(trigger_pos: Pos, occupied: dict[Pos, int]) -> Pos:
    tx, ty, tz = trigger_pos
    free_candidates = [
        (tx + dx, ty + dy, tz + dz)
        for dx, dy, dz in _NEIGHBOR_OFFSETS
        if (tx + dx, ty + dy, tz + dz) not in occupied
    ]
    if not free_candidates:
        raise ValueError(
            f"cannot place marker: trigger {trigger_pos} has no free neighboring cell"
        )

    def triggerable_neighbor_count(pos: Pos) -> int:
        px, py, pz = pos
        count = 0
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            neighbor = (px + dx, py + dy, pz + dz)
            state = occupied.get(neighbor)
            if state is not None and _unpack_state(state)[0] in TRIGGERABLE_IDS:
                count += 1
        return count

    # Prefer a marker cell whose only triggerable neighbor is the trigger itself,
    # so the reader can unambiguously resolve which block the marker activates.
    free_candidates.sort(key=triggerable_neighbor_count)
    return free_candidates[0]


def _tag_name(name: str) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack(">H", len(encoded)) + encoded


def _tag_compound_start(name: str) -> bytes:
    return bytes([10]) + _tag_name(name)


def _tag_int(name: str, value: int) -> bytes:
    return bytes([3]) + _tag_name(name) + struct.pack(">i", value)


def _tag_string(name: str, value: str) -> bytes:
    encoded = value.encode("utf-8")
    return bytes([8]) + _tag_name(name) + struct.pack(">H", len(encoded)) + encoded


def _tag_byte_array(name: str, payload: bytes) -> bytes:
    return bytes([7]) + _tag_name(name) + struct.pack(">i", len(payload)) + payload


def _write_schematic_nbt(width: int, height: int, length: int, blocks: bytes, data: bytes) -> bytes:
    buf = bytearray()
    buf += _tag_compound_start("Schematic")
    buf += _tag_int("Width", width)
    buf += _tag_int("Height", height)
    buf += _tag_int("Length", length)
    buf += _tag_string("Materials", "Alpha")
    buf += _tag_byte_array("Blocks", blocks)
    buf += _tag_byte_array("Data", data)
    buf += bytes([0])  # TAG_End closes the root compound
    return gzip.compress(bytes(buf))


def candidate_to_schematic_bytes(candidate: Candidate) -> bytes:
    """Build a gzip-compressed .schematic NBT payload for one genetic-ml candidate."""
    validate_candidate(candidate)
    trigger = candidate["trigger"]
    trigger_pos: Pos = (trigger["x"], trigger["y"], trigger["z"])

    occupied: dict[Pos, int] = {
        (block["x"], block["y"], block["z"]): block["state"] for block in candidate["blocks"]
    }
    if trigger_pos not in occupied:
        raise ValueError(f"candidate {candidate.get('id')}: trigger {trigger_pos} has no block")

    marker_pos = _find_marker_position(trigger_pos, occupied)

    all_positions = list(occupied.keys()) + [marker_pos]
    min_x = min(p[0] for p in all_positions)
    min_y = min(p[1] for p in all_positions)
    min_z = min(p[2] for p in all_positions)
    max_x = max(p[0] for p in all_positions)
    max_y = max(p[1] for p in all_positions)
    max_z = max(p[2] for p in all_positions)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1
    total = width * height * length

    block_bytes = bytearray(total)
    data_bytes = bytearray(total)

    def place(pos: Pos, block_id: int, meta: int) -> None:
        lx = pos[0] - min_x
        ly = pos[1] - min_y
        lz = pos[2] - min_z
        idx = (ly * length + lz) * width + lx
        block_bytes[idx] = block_id
        data_bytes[idx] = meta

    for pos, state in occupied.items():
        block_id, meta = _unpack_state(state)
        place(pos, block_id, meta)
    place(marker_pos, MARKER_BLOCK_ID, MARKER_META)

    return _write_schematic_nbt(width, height, length, bytes(block_bytes), bytes(data_bytes))


def export_candidate_to_schematic(candidate: Candidate, output_dir: Path, filename: str | None = None) -> Path:
    """Write one candidate's .schematic file into output_dir, returning the written path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename or f"candidate-{candidate['id']}.schematic"
    out_path = output_dir / out_name
    out_path.write_bytes(candidate_to_schematic_bytes(candidate))
    return out_path


def convert_json_to_schematics(json_path: Path, output_dir: Path) -> list[Path]:
    """Convert every candidate found in json_path (JSON-lines, one or more candidates)
    into a .schematic file under output_dir. Returns the list of written paths."""
    candidates = load_candidates_from_file(json_path)
    return [export_candidate_to_schematic(candidate, output_dir) for candidate in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a genetic-ml JSON candidate file into .schematic file(s)."
    )
    parser.add_argument("json_path", type=Path, help="Path to a candidate JSON (or JSON-lines) file")
    parser.add_argument("output_dir", type=Path, help="Directory to write the .schematic file(s) into")
    args = parser.parse_args()

    written = convert_json_to_schematics(args.json_path, args.output_dir)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
