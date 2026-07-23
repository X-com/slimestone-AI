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
    // Mirrors vanilla IBlockState.isNormalCube(): a full, opaque cube that does NOT itself
    // provide power. Used by redstonePower() to decide strong-power (getStrongPower, sums
    // all 6 neighbors) vs weak-power (the block's own facing) routing. A power source that
    // happens to also be a full cube (observer, active detector rail) is deliberately
    // excluded here so its own weak-power output isn't shadowed by this branch.
    bool isNormalCube = false;
    // Mirrors vanilla IBlockState.isFullBlock() (material.blocksMovement() && isFullCube()):
    // used only for physical support checks (isTopSolid, e.g. can a rail sit on top of this
    // block). Unlike isNormalCube, this does NOT exclude power-providing full cubes like the
    // observer - it sits on top of things and things sit on top of it just like any other
    // solid block.
    bool isFullBlock = false;
};

constexpr int BLOCK_AIR = 0;
constexpr int BLOCK_GOLDEN_RAIL = 27;
constexpr int BLOCK_DETECTOR_RAIL = 28;
constexpr int BLOCK_OBSIDIAN = 49;
constexpr int BLOCK_RAIL = 66;
constexpr int BLOCK_ACTIVATOR_RAIL = 157;
constexpr int BLOCK_FENCE_GATE = 107;
constexpr int BLOCK_TRAPDOOR = 96;
constexpr int BLOCK_IRON_TRAPDOOR = 167;
constexpr int BLOCK_STICKY_PISTON = 29;
constexpr int BLOCK_PISTON = 33;
constexpr int BLOCK_PISTON_HEAD = 34;
constexpr int BLOCK_PISTON_EXTENSION = 36;
constexpr int BLOCK_REDSTONE_BLOCK = 152;
constexpr int BLOCK_SLIME = 165;
constexpr int BLOCK_OBSERVER = 218;
constexpr int BLOCK_REDSTONE_LAMP = 123;      // unlit
constexpr int BLOCK_LIT_REDSTONE_LAMP = 124;  // lit

// BlockRailBase.EnumRailDirection ordinals (packed into the low meta bits, matching
// vanilla's on-disk metadata values exactly). Only the first six apply to powered rails
// (golden/activator/detector); plain rail additionally uses 6-9 for the curve shapes.
constexpr int RAIL_SHAPE_NORTH_SOUTH = 0;
constexpr int RAIL_SHAPE_EAST_WEST = 1;
constexpr int RAIL_SHAPE_ASCENDING_EAST = 2;
constexpr int RAIL_SHAPE_ASCENDING_WEST = 3;
constexpr int RAIL_SHAPE_ASCENDING_NORTH = 4;
constexpr int RAIL_SHAPE_ASCENDING_SOUTH = 5;
// Curve shapes - only reachable on plain rail (BLOCK_RAIL), the only type whose Rail helper
// runs with isPowered=false (BlockRailPowered/BlockRailDetector both pass isPowered=true,
// which disables curve selection entirely - see BlockRailBase.Rail.place()/connectTo()).
constexpr int RAIL_SHAPE_SOUTH_EAST = 6;
constexpr int RAIL_SHAPE_SOUTH_WEST = 7;
constexpr int RAIL_SHAPE_NORTH_WEST = 8;
constexpr int RAIL_SHAPE_NORTH_EAST = 9;

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
