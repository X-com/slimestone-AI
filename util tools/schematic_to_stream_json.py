"""Python port of Mcp1122SchematicToStreamJsonMain.java.

Converts legacy .schematic fixtures into one JSON candidate file per schematic,
in the format accepted by Mcp1122FlyingMachineStreamMain when streamed as one
input line. Trigger block next to the red-stained-glass marker must be a
(sticky) piston -- observers are no longer accepted as triggers.
"""
import gzip
import io
import json
import os
import struct
import sys

DEFAULT_FOLDER = "flying machines/schematics2"
DEFAULT_OUTPUT_FOLDER = "flying machines/json-output"

RED_STAINED_GLASS = 95
RED_STAINED_GLASS_META = 14
STICKY_PISTON = 29
PISTON = 33

# Real schematics can use any wood species/material for blocks that behave identically in
# the simulator (which only recognizes one id per behavior) - collapse every variant onto
# a single canonical id here, at the source, rather than teaching the simulator every
# real-world id. variant block id -> canonical block id.
BLOCK_ID_ALIASES = {
    # Iron trapdoor -> (wood) trapdoor: identical open/close behavior in the simulator,
    # material only affects hardness (mining, not simulated).
    167: 96,  # iron_trapdoor -> trapdoor

    # Fence gates: all six wood species are the same BlockFenceGate behavior in vanilla
    # (Material/texture is the only difference). 107 (oak) is the canonical id.
    183: 107,  # spruce_fence_gate
    184: 107,  # birch_fence_gate
    185: 107,  # jungle_fence_gate
    186: 107,  # dark_oak_fence_gate
    187: 107,  # acacia_fence_gate

    # Fences (not fence gates) and stairs: never specially recognized by the simulator,
    # always just inert non-solid geometry regardless of material - collapsed onto glass.
    85: 20,   # fence (oak)
    113: 20,  # nether_brick_fence
    188: 20,  # spruce_fence
    189: 20,  # birch_fence
    190: 20,  # jungle_fence
    191: 20,  # dark_oak_fence
    192: 20,  # acacia_fence
    53: 20,   # oak_stairs
    67: 20,   # stone_stairs
    108: 20,  # brick_stairs
    109: 20,  # stone_brick_stairs
    114: 20,  # nether_brick_stairs
    128: 20,  # sandstone_stairs
    134: 20,  # spruce_stairs
    135: 20,  # birch_stairs
    136: 20,  # jungle_stairs
    156: 20,  # quartz_stairs
    163: 20,  # acacia_stairs
    164: 20,  # dark_oak_stairs
    180: 20,  # red_sandstone_stairs
    203: 20,  # purpur_stairs

    # Glazed terracotta: 16 colors, all identical opaque-cube behavior. 235 (white) is
    # the canonical id.
    236: 235,  # orange_glazed_terracotta
    237: 235,  # magenta_glazed_terracotta
    238: 235,  # light_blue_glazed_terracotta
    239: 235,  # yellow_glazed_terracotta
    240: 235,  # lime_glazed_terracotta
    241: 235,  # pink_glazed_terracotta
    242: 235,  # gray_glazed_terracotta
    243: 235,  # silver_glazed_terracotta
    244: 235,  # cyan_glazed_terracotta
    245: 235,  # purple_glazed_terracotta
    246: 235,  # blue_glazed_terracotta
    247: 235,  # brown_glazed_terracotta
    248: 235,  # green_glazed_terracotta
    249: 235,  # red_glazed_terracotta
    250: 235,  # black_glazed_terracotta

    # Sandstone and concrete: plain opaque-cube blocks, no simulator-relevant difference
    # from stone.
    24: 1,    # sandstone
    179: 1,   # red_sandstone
    251: 1,   # concrete
    252: 1,   # concrete_powder

    # Stained glass -> glass. Red stained glass (meta 14) is the trigger-marker convention
    # handled separately above (consumed before reaching this table), so any id 95 here is
    # a real placed block, safe to flatten to plain glass.
    95: 20,   # stained_glass

    # Half slabs (single-height, non-full-cube) and glowstone: inert non-solid/decorative
    # blocks, same bucket as fences/stairs above. Note nether brick's fence *gate* doesn't
    # exist as a real block id in vanilla - only nether_brick_fence (113, already above).
    44: 20,   # stone_slab
    126: 20,  # wooden_slab
    182: 20,  # stone_slab2 (red sandstone)
    205: 20,  # purpur_slab
    89: 20,   # glowstone

    # Gold/diamond/iron blocks and log/log2 (all wood species): plain solid decorative
    # blocks, no simulator-relevant behavior difference from stone.
    41: 1,    # gold_block
    42: 1,    # iron_block
    57: 1,    # diamond_block
    17: 1,    # log
    162: 1,   # log2 (acacia/dark_oak)
}

# EnumFacing.values() order: DOWN, UP, NORTH, SOUTH, WEST, EAST
FACING_OFFSETS = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0)]

TAG_END, TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG, TAG_FLOAT, TAG_DOUBLE = range(7)
TAG_BYTE_ARRAY, TAG_STRING, TAG_LIST, TAG_COMPOUND, TAG_INT_ARRAY, TAG_LONG_ARRAY = range(7, 13)


class _NbtReader:
    def __init__(self, data):
        self.buf = io.BytesIO(data)

    def _read(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.buf.read(size))[0]

    def read_string(self):
        length = self._read(">H")
        return self.buf.read(length).decode("utf-8")

    def read_payload(self, tag_id):
        if tag_id == TAG_BYTE:
            return self._read(">b")
        if tag_id == TAG_SHORT:
            return self._read(">h")
        if tag_id == TAG_INT:
            return self._read(">i")
        if tag_id == TAG_LONG:
            return self._read(">q")
        if tag_id == TAG_FLOAT:
            return self._read(">f")
        if tag_id == TAG_DOUBLE:
            return self._read(">d")
        if tag_id == TAG_BYTE_ARRAY:
            length = self._read(">i")
            return self.buf.read(length)
        if tag_id == TAG_STRING:
            return self.read_string()
        if tag_id == TAG_LIST:
            element_type = self._read(">b")
            length = self._read(">i")
            return [self.read_payload(element_type) for _ in range(length)]
        if tag_id == TAG_COMPOUND:
            out = {}
            while True:
                child_id = self._read(">b")
                if child_id == TAG_END:
                    return out
                name = self.read_string()
                out[name] = self.read_payload(child_id)
        if tag_id == TAG_INT_ARRAY:
            length = self._read(">i")
            return [self._read(">i") for _ in range(length)]
        if tag_id == TAG_LONG_ARRAY:
            length = self._read(">i")
            return [self._read(">q") for _ in range(length)]
        raise IOError("unsupported NBT tag id %d" % tag_id)

    def read_root(self):
        tag_id = self._read(">b")
        if tag_id != TAG_COMPOUND:
            raise IOError("root tag must be a named compound tag")
        self.read_string()
        return self.read_payload(TAG_COMPOUND)


def read_schematic_root(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        raw = gzip.decompress(raw)
    return _NbtReader(raw).read_root()


def load_schematic(path):
    root = read_schematic_root(path)
    width, height, length = root["Width"], root["Height"], root["Length"]
    block_ids = root["Blocks"]
    block_data = root["Data"]
    add_blocks = root.get("AddBlocks")
    expected = width * height * length

    if len(block_ids) != expected or len(block_data) != expected:
        raise IOError("schematic Blocks/Data length does not match dimensions")

    blocks = {}
    markers = []

    for y in range(height):
        for z in range(length):
            for x in range(width):
                i = (y * length + z) * width + x
                block_id = block_ids[i]
                if add_blocks is not None:
                    extra = add_blocks[i // 2]
                    block_id |= ((extra >> 4) if i % 2 == 0 else (extra & 15)) << 8
                meta = block_data[i] & 15
                pos = (x, y, z)

                if block_id == RED_STAINED_GLASS and meta == RED_STAINED_GLASS_META:
                    markers.append(pos)
                    continue

                block_id = BLOCK_ID_ALIASES.get(block_id, block_id)

                if 1 <= block_id <= 255:
                    blocks[pos] = block_id | (meta << 8)

    if len(markers) != 1:
        raise IOError("expected exactly one red stained glass marker, found %d" % len(markers))

    marker = markers[0]
    trigger = find_trigger_target(blocks, marker)
    return blocks, marker, trigger


def find_trigger_target(blocks, marker):
    neighbors = []
    triggerable = []

    for dx, dy, dz in FACING_OFFSETS:
        pos = (marker[0] + dx, marker[1] + dy, marker[2] + dz)
        if pos not in blocks:
            continue
        neighbors.append(pos)
        block_id = blocks[pos] & 0xFF
        if block_id in (PISTON, STICKY_PISTON):
            triggerable.append(pos)

    if not neighbors:
        raise IOError("red stained glass marker must touch at least one supported block")
    if len(triggerable) > 1:
        raise IOError("red stained glass marker touches multiple triggerable blocks: %s" % triggerable)
    if triggerable:
        return triggerable[0]

    # BlockPos.compareTo order: y, then z, then x
    neighbors.sort(key=lambda p: (p[1], p[2], p[0]))
    return neighbors[0]


def list_schematics(folder):
    schematics = []
    if not os.path.isdir(folder):
        return schematics
    for entry in os.listdir(folder):
        full = os.path.join(folder, entry)
        if os.path.isdir(full):
            schematics.extend(list_schematics(full))
        elif entry.endswith(".schematic"):
            schematics.append(full)
    schematics.sort()
    return schematics


def base_name(file_name):
    name = file_name
    dot = name.rfind(".")
    if dot > 0:
        name = name[:dot]
    if len(name) == 0 or name in (".", ".."):
        return "schematic"
    return name


def to_json(id_, path, blocks, trigger):
    return json.dumps(
        {
            "id": id_,
            "name": os.path.basename(path),
            "path": path,
            "trigger": {"x": trigger[0], "y": trigger[1], "z": trigger[2]},
            "blocks": [
                {"x": pos[0], "y": pos[1], "z": pos[2], "state": state}
                for pos, state in blocks.items()
            ],
        },
        separators=(",", ":"),
    )


def main(argv):
    if len(argv) > 2:
        print("usage: schematic_to_stream_json.py [schematic-folder] [output-folder]", file=sys.stderr)
        return 2

    folder = argv[0] if len(argv) >= 1 else DEFAULT_FOLDER
    output_folder = argv[1] if len(argv) == 2 else DEFAULT_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    schematics = list_schematics(folder)
    written = 0

    for i, schematic in enumerate(schematics, start=1):
        blocks, marker, trigger = load_schematic(schematic)
        out_name = base_name(os.path.basename(schematic)) + ".json"
        out_path = os.path.join(output_folder, out_name)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(to_json(i, schematic, blocks, trigger))
            f.write("\n")
        print(out_path)
        written += 1

    print("wrote %d json files to %s" % (written, output_folder), file=sys.stderr)
    return 0


def _demo():
    """Minimal self-check for find_trigger_target's piston-only trigger rule."""
    marker = (0, 0, 0)
    # observer neighbor must NOT be selected as trigger
    blocks = {(1, 0, 0): 218, (0, 1, 0): 33}
    assert find_trigger_target(blocks, marker) == (0, 1, 0)

    # sticky piston neighbor selected
    blocks = {(0, -1, 0): 29}
    assert find_trigger_target(blocks, marker) == (0, -1, 0)

    # no triggerable block -> falls back to sorted neighbor (y, z, x)
    blocks = {(1, 0, 0): 1, (0, 0, 1): 1}
    assert find_trigger_target(blocks, marker) == (1, 0, 0)

    print("ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-check":
        _demo()
    else:
        sys.exit(main(sys.argv[1:]))
