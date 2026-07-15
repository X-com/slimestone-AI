#pragma once

#include "packed_pos.h"

#include <array>
#include <cstdint>
#include <vector>

namespace mcp1122gpu {

using mcp1122::BlockPos;

struct BlockEvent {
    BlockPos pos;
    int blockId = 0;
    int eventId = 0;
    int eventParam = 0;
};

struct ScheduledTick {
    std::int64_t time = 0;
    int order = 0;
    BlockPos pos;
    int blockId = 0;
};

struct MovingBlock {
    BlockPos pos;
    std::uint32_t pistonState = 0;
    int pistonBlockId = 0;
    int facing = 0;
    bool extending = false;
    bool shouldHeadBeRendered = false;
};

// Fixed-capacity replacement for std::unordered_set<uint64_t> keyed by packed position -
// watchSet_/observerSet_/ncPowerSet_ hold a handful to a few hundred positions even on the
// largest fixtures (piston/observer/rail counts), so linear scan is cheap and a real GPU
// kernel would need exactly this shape (fixed array, no allocation) for the same sets anyway.
class PosKeySet {
public:
    static constexpr std::size_t kCapacity = 4096;

    void clear() { size_ = 0; }
    bool empty() const { return size_ == 0; }
    std::size_t size() const { return size_; }
    bool contains(std::uint64_t key) const;
    std::size_t count(std::uint64_t key) const { return contains(key) ? 1 : 0; }
    void insert(std::uint64_t key);
    void erase(std::uint64_t key);

private:
    std::array<std::uint64_t, kCapacity> keys_{};
    std::size_t size_ = 0;
};

// Fixed-capacity, GPU-shaped world storage: a dense flat array over a bounded local cube,
// addressed by (worldPos - origin). No hash map, no per-candidate dynamic sizing decided at
// runtime beyond picking an origin - the cube itself is a fixed compile-time capacity, same
// as a real GPU kernel's per-thread storage would need to be.
//
// A flying machine translates through world space over the course of a simulation (that is
// the whole point of "flying"), so the cube cannot simply be sized to the candidate's initial
// bounding box once and left alone - later milestones that add the tick loop must re-center
// origin_ as the structure's anchor approaches the cube edge. Milestone A only loads the
// initial candidate, so no recentering is implemented yet; outOfRange() below is the explicit
// guard that will make a too-small cube or a missing recenter fail loudly instead of silently
// corrupting state.
class FixedWorld {
public:
    // 128 covers every stream-input fixture except a few known oversized "trencher"/tall
    // machines (spans up to 266 blocks on one axis) - those are expected to fail loudly with
    // load_error rather than being silently accommodated by a much larger default cube.
    static constexpr int kExtent = 128;   // cells per axis
    static constexpr int kMargin = 16;    // load-time clearance around the candidate's bbox

    // Fixed capacities for the per-tick queues, mirroring cpp extract's World but with an
    // explicit ceiling instead of unbounded std::vector growth - a push past capacity throws
    // rather than silently reallocating, same "no silent truncation" rule as setBlock() below.
    static constexpr std::size_t kMaxBlockEvents = 512;
    static constexpr std::size_t kMaxScheduledTicks = 512;
    static constexpr std::size_t kMaxMovingPerBucket = 16384;

    std::int64_t time = 0;
    BlockPos origin{0, 0, 0};

    std::array<std::vector<BlockEvent>, 2> blockEvents;
    int blockEventCacheIndex = 0;
    std::vector<ScheduledTick> scheduledTicks;
    int nextTickOrder = 0;
    std::array<std::vector<MovingBlock>, 3> movingBuckets;
    int movingPtr = 0;

    FixedWorld();

    void reset();
    void setOrigin(BlockPos newOrigin);

    // Throws std::out_of_range if pos falls outside the current cube - never silently
    // truncates or wraps, since a debug reference implementation that hides an overflow is
    // worse than no implementation at all.
    void setBlock(BlockPos pos, std::uint32_t state);
    void removeBlock(BlockPos pos); // alias for setBlock(pos, 0), matches cpp extract's World
    std::uint32_t getBlock(BlockPos pos) const;
    bool inRange(BlockPos pos) const;

    struct Entry {
        BlockPos pos;
        std::uint32_t state;
    };
    // Dense, incrementally maintained list of every non-air cell - O(1) amortized to keep in
    // sync from setBlock(), O(live count) to read, no per-call cube scan. Order is not
    // meaningful (swap-remove on delete, like cpp extract's PosStateMap::entries()), which is
    // fine: every consumer (stateKey()'s block hash, loadCandidate()'s watch-set scan) folds
    // entries with a commutative accumulator, so iteration order never affects the result.
    const std::vector<Entry>& entries() const { return entries_; }

    std::size_t movingCount() const;
    void pushBlockEvent(int queueIndex, const BlockEvent& event);
    void pushScheduledTick(const ScheduledTick& tick);
    void pushMovingBlock(int bucket, const MovingBlock& block);

private:
    std::vector<std::uint32_t> cells_;      // size kExtent^3, index = lx + ly*kExtent + lz*kExtent*kExtent
    std::vector<std::int32_t> entryIndex_;  // same indexing; -1 if empty, else index into entries_
    std::vector<Entry> entries_;            // compact, order not meaningful (see entries() above)

    std::size_t indexOf(BlockPos pos) const;
};

} // namespace mcp1122gpu
