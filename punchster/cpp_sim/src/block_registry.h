#pragma once

#include <array>
#include <cstdint>

namespace mcp1122 {

enum class PushReaction : std::uint8_t {
    Normal,
    Block,
    Destroy,
    PushOnly
};

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
constexpr int BLOCK_REDSTONE_LAMP = 123;      // unlit
constexpr int BLOCK_LIT_REDSTONE_LAMP = 124;  // lit
// Dedicated "trigger" block (punchster-only). Placed in a candidate to mark where the machine is
// kick-started: the simulator deletes every trigger block at tick 0, and the resulting break/neighbor
// update is what starts the machine (replacing the old redstone-pulse-a-piston mechanism). Id 253 is
// unused by every table in buildBlockData().
constexpr int BLOCK_TRIGGER = 253;

// BlockRailBase.EnumRailDirection ordinals (packed into the low meta bits, matching
// vanilla's on-disk metadata values exactly). Only the first six apply to powered rails
// (golden/activator/detector); plain rail additionally uses 6-9 for the curve shapes.
constexpr int RAIL_SHAPE_NORTH_SOUTH = 0;
constexpr int RAIL_SHAPE_EAST_WEST = 1;
constexpr int RAIL_SHAPE_ASCENDING_EAST = 2;
constexpr int RAIL_SHAPE_ASCENDING_WEST = 3;
constexpr int RAIL_SHAPE_ASCENDING_NORTH = 4;
constexpr int RAIL_SHAPE_ASCENDING_SOUTH = 5;

std::uint32_t makeState(int blockId, int meta);
int blockId(std::uint32_t state);
int blockMeta(std::uint32_t state);
bool metaBit(std::uint32_t state, int bit);
std::uint32_t setMetaBit(std::uint32_t state, int bit, bool value);
int facingMeta(std::uint32_t state);
std::uint32_t setFacingMeta(std::uint32_t state, int facingIndex);

const BlockData& blockData(int id);
bool isAirState(std::uint32_t state);
bool isPistonBlock(int id);
bool isStickyPistonBlock(int id);
bool isNormalPowerSource(std::uint32_t state);
bool isRailBlock(int id);
// Golden/activator rails toggle their own powered bit from redstone (BlockRailPowered);
// detector rail is cart-driven in vanilla (no entity model here, so it stays static) and
// plain rail has no powered bit at all.
bool isPoweredRailBlock(int id);
// Decodes the rail shape bits: mask 0xF (10 shapes) for plain rail, mask 0x7 (6 shapes,
// no diagonals) for golden/detector/activator rails.
int railShape(int id, std::uint32_t state);
bool isAscendingRailShape(int shape);

} // namespace mcp1122
