#pragma once

// Device-compatible port of cpp extract's block_registry.h/.cpp. Pure lookup tables, already
// exception-free and allocation-free on host - only change is building the 256-entry BlockData
// table into __constant__ memory once from the host (see uploadBlockRegistry() in
// block_registry.cu) instead of a function-local `static const` initialized on first call,
// since device code can't run that lazy-init the first time a thread touches it. All 256
// entries are tiny (looked up by every thread every tick), so __constant__ broadcast is exactly
// the right memory space for it.

#include "gpu_common.cuh"

#include <array>
#include <cstdint>

namespace mcp1122gpu {

enum class PushReaction : std::uint8_t { Normal, Block, Destroy, PushOnly };

struct BlockData {
    PushReaction pushReaction = PushReaction::Normal;
    float hardness = 0.0f;
    bool hasTileEntity = false;
    bool canProvidePower = false;
    bool isNormalCube = false;
};

constexpr int BLOCK_AIR = 0;
constexpr int BLOCK_GOLDEN_RAIL = 27;
constexpr int BLOCK_DETECTOR_RAIL = 28;
constexpr int BLOCK_OBSIDIAN = 49;
constexpr int BLOCK_RAIL = 66;
constexpr int BLOCK_ACTIVATOR_RAIL = 157;
constexpr int BLOCK_FENCE_GATE = 107;
constexpr int BLOCK_STICKY_PISTON = 29;
constexpr int BLOCK_PISTON = 33;
constexpr int BLOCK_PISTON_HEAD = 34;
constexpr int BLOCK_PISTON_EXTENSION = 36;
constexpr int BLOCK_REDSTONE_BLOCK = 152;
constexpr int BLOCK_SLIME = 165;
constexpr int BLOCK_OBSERVER = 218;
constexpr int BLOCK_REDSTONE_LAMP = 123;
constexpr int BLOCK_LIT_REDSTONE_LAMP = 124;

constexpr int RAIL_SHAPE_NORTH_SOUTH = 0;
constexpr int RAIL_SHAPE_EAST_WEST = 1;
constexpr int RAIL_SHAPE_ASCENDING_EAST = 2;
constexpr int RAIL_SHAPE_ASCENDING_WEST = 3;
constexpr int RAIL_SHAPE_ASCENDING_NORTH = 4;
constexpr int RAIL_SHAPE_ASCENDING_SOUTH = 5;

// Host-built, device-uploaded table (see buildBlockRegistryTable() in block_registry.cu and
// initBlockRegistryDevice() in gpu_kernel.cu). Defined directly in the header (not extern) since
// this project builds without -rdc=true (relocatable device code) - each .cu that includes this
// header gets its own private __constant__ copy, and the one .cu that actually launches the
// kernel (gpu_kernel.cu) is also the one that uploads to it, so its copy is the only one that
// matters. On host builds (no __CUDACC__) this is just an ordinary global.
#if defined(__CUDACC__)
static __constant__ BlockData g_blockRegistry[256];
#define MCPGPU_BLOCK_REGISTRY g_blockRegistry
#else
static BlockData g_blockRegistryHost[256];
#define MCPGPU_BLOCK_REGISTRY g_blockRegistryHost
#endif

std::array<BlockData, 256> buildBlockRegistryTable();

MCPGPU_HD inline std::uint32_t makeState(int id, int meta) { return static_cast<std::uint32_t>(id | (meta << 8)); }
MCPGPU_HD inline int blockId(std::uint32_t state) { return static_cast<int>(state & 0xffu); }
MCPGPU_HD inline int blockMeta(std::uint32_t state) { return static_cast<int>(state >> 8); }
MCPGPU_HD inline bool metaBit(std::uint32_t state, int bit) { return (blockMeta(state) & (1 << bit)) != 0; }

MCPGPU_HD inline std::uint32_t setMetaBit(std::uint32_t state, int bit, bool value) {
    int meta = blockMeta(state);
    if (value) meta |= 1 << bit; else meta &= ~(1 << bit);
    return makeState(blockId(state), meta);
}

MCPGPU_HD inline int facingMeta(std::uint32_t state) { return blockMeta(state) & 7; }

MCPGPU_HD inline std::uint32_t setFacingMeta(std::uint32_t state, int facingIndex) {
    int meta = (blockMeta(state) & ~7) | (facingIndex & 7);
    return makeState(blockId(state), meta);
}

MCPGPU_HD inline const BlockData& blockData(int id) { return MCPGPU_BLOCK_REGISTRY[id & 0xff]; }

MCPGPU_HD inline bool isAirState(std::uint32_t state) { return blockId(state) == BLOCK_AIR; }
MCPGPU_HD inline bool isPistonBlock(int id) { return id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON; }
MCPGPU_HD inline bool isStickyPistonBlock(int id) { return id == BLOCK_STICKY_PISTON; }
MCPGPU_HD inline bool isNormalPowerSource(std::uint32_t state) { return blockId(state) == BLOCK_REDSTONE_BLOCK; }
MCPGPU_HD inline bool isRailBlock(int id) {
    return id == BLOCK_RAIL || id == BLOCK_GOLDEN_RAIL || id == BLOCK_DETECTOR_RAIL || id == BLOCK_ACTIVATOR_RAIL;
}
MCPGPU_HD inline bool isPoweredRailBlock(int id) { return id == BLOCK_GOLDEN_RAIL || id == BLOCK_ACTIVATOR_RAIL; }
MCPGPU_HD inline int railShape(int id, std::uint32_t state) { return blockMeta(state) & (id == BLOCK_RAIL ? 0xF : 0x7); }
MCPGPU_HD inline bool isAscendingRailShape(int shape) {
    return shape >= RAIL_SHAPE_ASCENDING_EAST && shape <= RAIL_SHAPE_ASCENDING_SOUTH;
}

} // namespace mcp1122gpu
