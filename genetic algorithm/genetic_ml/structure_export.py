"""Converts a genetic-ml candidate into a real Minecraft structure block NBT file (the format
loaded via the in-game structure block UI / /place command), gzip-compressed per Minecraft's
requirement.

This is a different, more complex format than schematic_export.py's legacy .schematic writer:
blocks reference a deduplicated palette of modern blockstate name+properties strings by index,
rather than a flat legacy id/meta byte array.

    root compound
    |-- DataVersion: int          (1343 - Minecraft 1.12.2, matching mcp1122's target version)
    |-- size: list<int> [w, h, l]
    |-- entities: list<compound>  (always empty - candidates never contain entities)
    |-- palette: list<compound>   [{"Name": "minecraft:piston", "Properties": {...}}, ...]
    `-- blocks: list<compound>    [{"pos": [x,y,z], "state": <palette index>, "nbt": {...}?}, ...]

The trigger is represented with a command block instead of a marker block mixed into the real
geometry (unlike the legacy schematic writer's red-stained-glass marker): pasting a structure,
like our own candidate loader, never fires neighbor-update notifications for blocks already
present, so the trigger block needs a manual "poke" after paste to actually start the machine.
The command block sits at a fixed, deterministic outer corner of the bounding box (one cell
past the minimum-x face, at the minimum y/z), never colliding with a real machine block, and
contains a `setblock ~dx ~dy ~dz <block>[<props>] replace` command using coordinates relative
to itself - which stay correct no matter where the structure is later pasted in the world,
since paste-translation preserves relative offsets exactly. It's left in impulse/needs-redstone
mode (Minecraft's default) so it only fires when manually powered, not the moment the chunk
loads.

genetic-ml has zero third-party dependencies; this hand-rolls the NBT writer rather than
depending on a library, same approach as schematic_export.py.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any, Callable

from genetic_ml.blocks import (
    BLOCK_AIR,
    BLOCK_GLASS,
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_PISTON_HEAD,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
    BLOCK_STONE,
    block_id,
    block_meta,
)

Candidate = dict[str, Any]
Pos = tuple[int, int, int]

_DATA_VERSION = 1343  # Minecraft 1.12.2

_FACING_NAMES = ("down", "up", "north", "south", "west", "east")


# ---------------------------------------------------------------------------
# Minimal hand-rolled NBT writer (big-endian per spec). Two families of helper:
# "tag_*" build a NAMED tag (id + name + payload) for use as a TAG_Compound child;
# "payload_*" build just the payload (no id/name) for use as a TAG_List element,
# since NBT lists store their element type once in the list header, not per element.
# ---------------------------------------------------------------------------

def _name_bytes(name: str) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack(">H", len(encoded)) + encoded


def _named(tag_id: int, name: str, payload: bytes) -> bytes:
    return bytes([tag_id]) + _name_bytes(name) + payload


def tag_int(name: str, value: int) -> bytes:
    return _named(3, name, struct.pack(">i", value))


def tag_byte(name: str, value: int) -> bytes:
    return _named(1, name, struct.pack(">b", value))


def tag_string(name: str, value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _named(8, name, struct.pack(">H", len(encoded)) + encoded)


def tag_compound(name: str, children: bytes) -> bytes:
    return _named(10, name, children + bytes([0]))  # + TAG_End


def tag_list(name: str, element_type: int, element_payloads: list[bytes]) -> bytes:
    header = bytes([element_type]) + struct.pack(">i", len(element_payloads))
    return _named(9, name, header + b"".join(element_payloads))


def payload_int(value: int) -> bytes:
    return struct.pack(">i", value)


def payload_compound(children: bytes) -> bytes:
    return children + bytes([0])  # + TAG_End


# ---------------------------------------------------------------------------
# Legacy id/meta -> modern blockstate name + properties. Covers exactly the fixed set of
# blocks mutation.py's palette (plus the piston-head settle logic) can ever produce - not a
# general-purpose block registry.
# ---------------------------------------------------------------------------

def _simple(name: str) -> Callable[[int], tuple[str, dict[str, str]]]:
    def convert(_meta: int) -> tuple[str, dict[str, str]]:
        return name, {}
    return convert


def _piston_like(name: str) -> Callable[[int], tuple[str, dict[str, str]]]:
    def convert(meta: int) -> tuple[str, dict[str, str]]:
        facing = _FACING_NAMES[meta & 0b111]
        extended = "true" if meta & 0x8 else "false"
        return name, {"facing": facing, "extended": extended}
    return convert


def _observer(meta: int) -> tuple[str, dict[str, str]]:
    facing = _FACING_NAMES[meta & 0b111]
    powered = "true" if meta & 0x8 else "false"
    return "minecraft:observer", {"facing": facing, "powered": powered}


def _piston_head(meta: int) -> tuple[str, dict[str, str]]:
    facing = _FACING_NAMES[meta & 0b111]
    piston_type = "sticky" if meta & 0x8 else "normal"
    return "minecraft:piston_head", {"facing": facing, "type": piston_type, "short": "false"}


_BLOCK_CONVERTERS: dict[int, Callable[[int], tuple[str, dict[str, str]]]] = {
    BLOCK_AIR: _simple("minecraft:air"),
    BLOCK_STONE: _simple("minecraft:stone"),
    BLOCK_GLASS: _simple("minecraft:glass"),
    BLOCK_STICKY_PISTON: _piston_like("minecraft:sticky_piston"),
    BLOCK_PISTON: _piston_like("minecraft:piston"),
    BLOCK_PISTON_HEAD: _piston_head,
    BLOCK_REDSTONE_BLOCK: _simple("minecraft:redstone_block"),
    BLOCK_SLIME: _simple("minecraft:slime"),
    BLOCK_OBSERVER: _observer,
}


def legacy_to_modern(state: int) -> tuple[str, dict[str, str]]:
    bid = block_id(state)
    converter = _BLOCK_CONVERTERS.get(bid)
    if converter is None:
        raise ValueError(f"no modern blockstate mapping for legacy block id {bid}")
    return converter(block_meta(state))


class _PaletteBuilder:
    def __init__(self) -> None:
        self._index: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self.entries: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def id_for(self, name: str, props: dict[str, str]) -> int:
        key = (name, tuple(sorted(props.items())))
        existing = self._index.get(key)
        if existing is not None:
            return existing
        index = len(self.entries)
        self._index[key] = index
        self.entries.append(key)
        return index

    def to_nbt_list(self) -> bytes:
        payloads = []
        for name, props in self.entries:
            children = tag_string("Name", name)
            if props:
                prop_children = b"".join(tag_string(key, value) for key, value in props)
                children += tag_compound("Properties", prop_children)
            payloads.append(payload_compound(children))
        return tag_list("palette", 10, payloads)


def candidate_to_structure_bytes(candidate: Candidate) -> bytes:
    """Builds a gzip-compressed structure-block NBT payload for one genetic-ml candidate."""
    blocks_in = candidate["blocks"]
    if not blocks_in:
        raise ValueError(f"candidate {candidate.get('id')}: cannot export an empty machine")

    min_x = min(block["x"] for block in blocks_in)
    min_y = min(block["y"] for block in blocks_in)
    min_z = min(block["z"] for block in blocks_in)
    max_x = max(block["x"] for block in blocks_in)
    max_y = max(block["y"] for block in blocks_in)
    max_z = max(block["z"] for block in blocks_in)

    trigger = candidate["trigger"]
    trigger_pos: Pos = (trigger["x"], trigger["y"], trigger["z"])

    occupied: dict[Pos, int] = {(block["x"], block["y"], block["z"]): block["state"] for block in blocks_in}
    trigger_state = occupied.get(trigger_pos)
    if trigger_state is None:
        raise ValueError(f"candidate {candidate.get('id')}: trigger {trigger_pos} has no block")

    # Fixed, deterministic outer corner - one cell past the minimum-x face, at the minimum
    # y/z - so it can never collide with a real machine block regardless of the machine's
    # shape, without needing an occupancy search.
    command_pos: Pos = (min_x - 1, min_y, min_z)

    bbox_min = (min(min_x, command_pos[0]), min(min_y, command_pos[1]), min(min_z, command_pos[2]))
    bbox_max = (max(max_x, command_pos[0]), max(max_y, command_pos[1]), max(max_z, command_pos[2]))
    width = bbox_max[0] - bbox_min[0] + 1
    height = bbox_max[1] - bbox_min[1] + 1
    length = bbox_max[2] - bbox_min[2] + 1

    dx = trigger_pos[0] - command_pos[0]
    dy = trigger_pos[1] - command_pos[1]
    dz = trigger_pos[2] - command_pos[2]
    trigger_name, _ = legacy_to_modern(trigger_state)
    # The structure itself leaves the trigger position as air (see the block loop below);
    # placing the real trigger block there is what fires the neighbor update that starts the
    # machine - pasting a structure never fires updates on its own.
    # 1.12.2's /setblock predates the 1.13 flattening: it takes a numeric meta value, not
    # blockstate [property=value] syntax, which 1.12.2 doesn't parse.
    command_text = f"setblock ~{dx} ~{dy} ~{dz} {trigger_name} {block_meta(trigger_state)} replace"

    palette = _PaletteBuilder()
    air_id = palette.id_for("minecraft:air", {})
    command_block_id = palette.id_for("minecraft:command_block", {"facing": "up", "conditional": "false"})

    command_nbt = tag_string("id", "minecraft:command_block") + tag_string("Command", command_text) + tag_byte("TrackOutput", 0)

    block_payloads: list[bytes] = []
    for x in range(bbox_min[0], bbox_max[0] + 1):
        for y in range(bbox_min[1], bbox_max[1] + 1):
            for z in range(bbox_min[2], bbox_max[2] + 1):
                pos = (x, y, z)
                local = (x - bbox_min[0], y - bbox_min[1], z - bbox_min[2])
                local_payload = tag_list("pos", 3, [payload_int(c) for c in local])

                if pos == command_pos:
                    children = local_payload + tag_int("state", command_block_id) + tag_compound("nbt", command_nbt)
                    block_payloads.append(payload_compound(children))
                    continue

                state = None if pos == trigger_pos else occupied.get(pos)
                if state is None:
                    palette_idx = air_id
                else:
                    name, props = legacy_to_modern(state)
                    palette_idx = palette.id_for(name, props)
                children = local_payload + tag_int("state", palette_idx)
                block_payloads.append(payload_compound(children))

    root_children = (
        tag_int("DataVersion", _DATA_VERSION)
        + tag_list("size", 3, [payload_int(width), payload_int(height), payload_int(length)])
        + tag_list("entities", 10, [])
        + tag_list("blocks", 10, block_payloads)
        + palette.to_nbt_list()
    )
    root = tag_compound("", root_children)
    return gzip.compress(root)


def export_candidate_to_structure(candidate: Candidate, output_dir: Path, filename: str | None = None) -> Path:
    """Writes one candidate's structure .nbt file into output_dir, returning the written path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename or f"candidate-{candidate['id']}.nbt"
    out_path = output_dir / out_name
    out_path.write_bytes(candidate_to_structure_bytes(candidate))
    return out_path
