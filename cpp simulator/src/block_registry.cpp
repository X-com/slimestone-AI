#include "block_registry.h"

namespace mcp1122 {

namespace {

std::array<BlockData, 256> buildBlockData() {
    std::array<BlockData, 256> data{};
    for (BlockData& entry : data) {
        entry.pushReaction = PushReaction::Normal;
        entry.hardness = 1.0f;
        entry.hasTileEntity = false;
        entry.canProvidePower = false;
        entry.isNormalCube = false;
    }

    data[BLOCK_AIR].hardness = 0.0f;
    data[BLOCK_OBSIDIAN].hardness = 50.0f;
    data[BLOCK_PISTON].pushReaction = PushReaction::Block;
    data[BLOCK_STICKY_PISTON].pushReaction = PushReaction::Block;
    data[BLOCK_PISTON_HEAD].pushReaction = PushReaction::Block;
    data[BLOCK_PISTON_EXTENSION].pushReaction = PushReaction::Block;
    data[BLOCK_PISTON_EXTENSION].hardness = -1.0f;
    data[BLOCK_PISTON_EXTENSION].hasTileEntity = true;
    data[BLOCK_REDSTONE_BLOCK].canProvidePower = true;
    data[BLOCK_OBSERVER].canProvidePower = true;

    const int normalCubes[] = {
        1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 19, 21, 22, 24, 35, 41,
        42, 43, 45, 47, 48, 56, 57, 58, 61, 62, 73, 74, 80, 82, 87, 88,
        97, 98, 99, 100, 110, 112, 121, 123, 124, 125, 129, 133, 137,
        159, 162, 165, 166, 168, 170, 172, 173, 174, 179, 181, 201, 202,
        204, 206, 210, 211, 213, 214, 215, 216, 251, 252, 255
    };
    for (int id : normalCubes) {
        data[static_cast<std::size_t>(id)].isNormalCube = true;
    }

    const int destroyBlocks[] = {
        6, 8, 9, 10, 11, 18, 26, 30, 31, 32, 37, 38, 39, 40, 50, 51, 55,
        64, 65, 70, 71, 72, 75, 76, 78, 81, 83, 86, 91, 92, 93, 94, 103,
        104, 105, 106, 115, 122, 127, 131, 132, 140, 143, 147, 148, 175,
        193, 194, 195, 196, 197, 199, 200, 207, 217, 219, 220, 221, 222,
        223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234
    };
    for (int id : destroyBlocks) {
        data[static_cast<std::size_t>(id)].pushReaction = PushReaction::Destroy;
    }

    const int blockBlocks[] = {7, 90, 119, 145, 166, 209};
    for (int id : blockBlocks) {
        data[static_cast<std::size_t>(id)].pushReaction = PushReaction::Block;
    }

    for (int id = 235; id <= 250; ++id) {
        data[static_cast<std::size_t>(id)].pushReaction = PushReaction::PushOnly;
        data[static_cast<std::size_t>(id)].isNormalCube = true;
    }

    const int tileBlocks[] = {
        23, 25, 26, 52, 54, 61, 62, 63, 68, 84, 116, 119, 130, 137, 138,
        140, 144, 146, 149, 150, 151, 154, 158, 176, 177, 178, 209, 219,
        220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
        233, 234, 255
    };
    for (int id : tileBlocks) {
        data[static_cast<std::size_t>(id)].hasTileEntity = true;
    }

    const int immovable[] = {7, 90, 119, 120, 137, 166, 209, 210, 211, 255};
    for (int id : immovable) {
        data[static_cast<std::size_t>(id)].hardness = -1.0f;
    }

    return data;
}

const std::array<BlockData, 256>& registry() {
    static const std::array<BlockData, 256> data = buildBlockData();
    return data;
}

} // namespace

std::uint32_t makeState(int id, int meta) {
    return static_cast<std::uint32_t>(id | (meta << 8));
}

int blockId(std::uint32_t state) {
    return static_cast<int>(state & 0xffu);
}

int blockMeta(std::uint32_t state) {
    return static_cast<int>(state >> 8);
}

bool metaBit(std::uint32_t state, int bit) {
    return (blockMeta(state) & (1 << bit)) != 0;
}

std::uint32_t setMetaBit(std::uint32_t state, int bit, bool value) {
    int meta = blockMeta(state);
    if (value) {
        meta |= 1 << bit;
    } else {
        meta &= ~(1 << bit);
    }
    return makeState(blockId(state), meta);
}

int facingMeta(std::uint32_t state) {
    return blockMeta(state) & 7;
}

std::uint32_t setFacingMeta(std::uint32_t state, int facingIndex) {
    int meta = (blockMeta(state) & ~7) | (facingIndex & 7);
    return makeState(blockId(state), meta);
}

const BlockData& blockData(int id) {
    return registry()[static_cast<std::size_t>(id & 0xff)];
}

bool isAirState(std::uint32_t state) {
    return blockId(state) == BLOCK_AIR;
}

bool isPistonBlock(int id) {
    return id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON;
}

bool isStickyPistonBlock(int id) {
    return id == BLOCK_STICKY_PISTON;
}

bool isNormalPowerSource(std::uint32_t state) {
    return blockId(state) == BLOCK_REDSTONE_BLOCK;
}

bool isRailBlock(int id) {
    return id == BLOCK_RAIL || id == BLOCK_GOLDEN_RAIL || id == BLOCK_DETECTOR_RAIL || id == BLOCK_ACTIVATOR_RAIL;
}

bool isPoweredRailBlock(int id) {
    return id == BLOCK_GOLDEN_RAIL || id == BLOCK_ACTIVATOR_RAIL;
}

int railShape(int id, std::uint32_t state) {
    return blockMeta(state) & (id == BLOCK_RAIL ? 0xF : 0x7);
}

bool isAscendingRailShape(int shape) {
    return shape >= RAIL_SHAPE_ASCENDING_EAST && shape <= RAIL_SHAPE_ASCENDING_SOUTH;
}

} // namespace mcp1122
