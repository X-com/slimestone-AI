#pragma once

// Device-compatible port of ../reference/simulator.h/.cpp (Milestone H: mechanical translation
// of reference/ into a one-thread-per-candidate kernel, per gpu extract/README.md's porting
// milestones). Differences from reference/, all forced by running as device code rather than
// changes to game logic - every line of actual simulation behavior below is line-for-line the
// same as reference/simulator.cpp:
//
//  - No Trace. reference/'s trace output already proved this algorithm byte-identical against
//    cpp extract tick-by-tick (VERIFICATION.md milestones A-G) - that job is done. The kernel's
//    job is only to reproduce reference/'s *results*, checked directly against reference/ output
//    (see ../../VERIFICATION.md's Milestone H note: validated against reference/, not cpp
//    extract directly).
//  - No std::string/std::unordered_map/std::sort/std::vector. Cycle detection's
//    unordered_map<StateKey,SeenState> becomes a fixed-capacity open-addressing table
//    (kSeenCapacity slots, linear probe); tickScheduledUpdates' std::sort becomes an insertion
//    sort over the (<=512 element) scheduledTicks_ array - fine at this size, no different in
//    ordering outcome.
//  - No exceptions. See world.cuh's file header for the sticky-error-flag replacement
//    (world_.error_). Simulator adds its own `bboxError_` for the one throw reference/'s
//    loadCandidate has that isn't a FixedWorld capacity check (the initial "candidate needs a
//    bigger cube" bounds check) - both are checked once per tick / once at load, not after every
//    single mutating call, exactly matching an exception unwind's *outcome* (abort, report
//    simulation_error) without needing device-side exceptions to get there. See world.cuh's
//    header comment for why this is safe: none of the 35 correctness-verified fixtures ever
//    cross a capacity ceiling, so this only changes the reporting path for candidates already
//    documented as out of scope.

#include "block_registry.cuh"
#include "facing.cuh"
#include "gpu_common.cuh"
#include "piston.cuh"
#include "world.cuh"

#include <cstdint>

#if !defined(__CUDACC__)
#include <cstdlib>
#include <iostream>
#endif

namespace mcp1122gpu {

// One candidate's input, as a view into a host-marshalled, pooled block array (CSR-style: all
// candidates' blocks concatenated, each candidate holds an offset + count into that pool) -
// mirrors mcp1122::Candidate but without std::vector.
struct GpuBlockEntry {
    int x = 0;
    int y = 0;
    int z = 0;
    std::uint32_t state = 0;
};

struct GpuCandidateView {
    std::int64_t id = 0;
    BlockPos trigger;
    const GpuBlockEntry* blocks = nullptr;
    int blockCount = 0;
};

enum class GpuErrorCode : int { None = 0, BboxTooLarge = 1, SimulationError = 2 };

struct GpuResult {
    std::int64_t id = 0;
    bool ok = false;
    bool working = false;
    int ticks = 0;
    int start = 0;
    int end = 0;
    int period = 0;
    BlockPos shift;
    GpuErrorCode errorCode = GpuErrorCode::None;
};

struct StateKey {
    std::uint64_t words[12] = {};
    MCPGPU_HD bool equals(const StateKey& o) const {
        for (int i = 0; i < 12; ++i) if (words[i] != o.words[i]) return false;
        return true;
    }
};

namespace detail {

MCPGPU_HD inline bool isWatchedBlock(int id) {
    return id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON || id == BLOCK_PISTON_HEAD ||
           id == BLOCK_FENCE_GATE || id == BLOCK_LIT_REDSTONE_LAMP || id == BLOCK_REDSTONE_LAMP ||
           isRailBlock(id);
}

MCPGPU_HD inline bool isNcPowerSource(int id, std::uint32_t state) {
    if (id == BLOCK_OBSERVER && metaBit(state, 3)) return true;
    if (id == BLOCK_DETECTOR_RAIL && metaBit(state, 3)) return true;
    return false;
}

MCPGPU_HD inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

MCPGPU_HD inline std::uint64_t encodeRel(int dx, int dy, int dz) {
    return (static_cast<std::uint64_t>(dx + 1048576) & 0x1fffffULL) |
           ((static_cast<std::uint64_t>(dy + 1048576) & 0x1fffffULL) << 21) |
           ((static_cast<std::uint64_t>(dz + 1048576) & 0x1fffffULL) << 42);
}

MCPGPU_HD inline bool anchorBeats(BlockPos pos, BlockPos cur, bool hasCur) {
    if (!hasCur) return true;
    if (pos.y != cur.y) return pos.y < cur.y;
    if (pos.z != cur.z) return pos.z < cur.z;
    return pos.x < cur.x;
}

MCPGPU_HD inline bool samePos(BlockPos a, BlockPos b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

} // namespace detail

class Simulator {
public:
    // Cycle-detection hash table capacity. Default maxTicks=6000 means at most 6001 distinct
    // states get inserted before either a cycle is found or the tick budget runs out; 16384
    // keeps load factor under 40% for linear probing, matching the generous-headroom style of
    // every other kMax* cap in this port.
    static constexpr int kSeenCapacity = 16384;

    MCPGPU_HD void bindCube(std::uint32_t* cells, std::int32_t* entryIndex) { world_.bindCube(cells, entryIndex); }

    // Call after simulate() returns and before the next simulate() call on the same worker
    // (Milestone I: one worker now runs many candidates back-to-back) - see
    // FixedWorld::clearLiveCells() for why this is O(live count), not a full-cube rescan.
    MCPGPU_HD void clearWorldCube() { world_.clearLiveCells(); }

    MCPGPU_HD GpuResult simulate(const GpuCandidateView& candidate, int maxTicks) {
        GpuResult result;
        result.id = candidate.id;

        if (!loadCandidate(candidate)) {
            result.ok = false;
            result.errorCode = GpuErrorCode::BboxTooLarge;
            return result;
        }
        trigger(candidate.trigger);
        if (world_.error_) {
            result.ok = false;
            result.errorCode = GpuErrorCode::SimulationError;
            return result;
        }

        ShiftCycle cycle;
        bool found = detectShiftCycle(maxTicks, cycle);
        if (world_.error_) {
            result.ok = false;
            result.errorCode = GpuErrorCode::SimulationError;
            return result;
        }
        result.ok = true;
        if (!found) {
            result.working = false;
            result.ticks = maxTicks;
        } else {
            result.working = !detail::samePos(cycle.shift, BlockPos{0, 0, 0});
            result.ticks = cycle.end;
            result.start = cycle.start;
            result.end = cycle.end;
            result.period = cycle.period;
            result.shift = cycle.shift;
        }
        return result;
    }

private:
    struct ShiftCycle {
        int start = 0;
        int end = 0;
        int period = 0;
        BlockPos shift;
    };
    struct SeenSlot {
        bool used = false;
        StateKey key;
        int tick = 0;
        BlockPos anchor;
    };

    FixedWorld world_;
    PosKeySet watchSet_;
    PosKeySet observerSet_;
    PosKeySet ncPowerSet_;
    SeenSlot seen_[kSeenCapacity];

    // Scratch buffers for the few functions that need to snapshot-then-clear a queue before
    // reprocessing it (see each call site below). These must NOT be function-local arrays: a
    // GPU thread's call stack is a few KB by default, and e.g. a 16384-element MovingBlock
    // snapshot is ~450KB - declaring that on the stack inside updateEntities() overflows it
    // instantly. Living here as Simulator fields keeps them in the same per-candidate
    // global-memory slot as everything else in this class.
    ScheduledTick pendingDueScratch_[FixedWorld::kMaxScheduledTicks];
    BlockEvent eventsScratch_[FixedWorld::kMaxBlockEvents];
    MovingBlock doneScratch_[FixedWorld::kMaxMovingPerBucket];

    // --- load / trigger -----------------------------------------------------------------

    MCPGPU_HD bool loadCandidate(const GpuCandidateView& candidate) {
        world_.reset();
        watchSet_.clear();
        observerSet_.clear();
        ncPowerSet_.clear();

        int minX = 2147483647, minY = 2147483647, minZ = 2147483647;
        int maxX = -2147483648, maxY = -2147483648, maxZ = -2147483648;
        auto include = [&](int x, int y, int z) {
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        };
        for (int i = 0; i < candidate.blockCount; ++i) {
            include(candidate.blocks[i].x, candidate.blocks[i].y, candidate.blocks[i].z);
        }
        include(candidate.trigger.x, candidate.trigger.y, candidate.trigger.z);

        int spanX = maxX - minX + 1, spanY = maxY - minY + 1, spanZ = maxZ - minZ + 1;
        int span = spanX > spanY ? spanX : spanY;
        if (spanZ > span) span = spanZ;
        int needed = span + 2 * FixedWorld::kMargin;
        if (needed > FixedWorld::kExtent) {
            return false;
        }

        world_.setOrigin(BlockPos{minX - FixedWorld::kMargin, minY - FixedWorld::kMargin, minZ - FixedWorld::kMargin});

        for (int i = 0; i < candidate.blockCount; ++i) {
            const GpuBlockEntry& block = candidate.blocks[i];
            if (block.state != 0) {
                world_.setBlock(BlockPos{block.x, block.y, block.z}, block.state);
            }
        }

        for (int i = 0; i < world_.entryCount_; ++i) {
            const FixedWorld::Entry& e = world_.entries_[i];
            int id = blockId(e.state);
            std::uint64_t key = packPos(e.pos);
            if (id == BLOCK_OBSERVER) observerSet_.insert(key, &world_.error_);
            if (detail::isWatchedBlock(id)) watchSet_.insert(key, &world_.error_);
            if (detail::isNcPowerSource(id, e.state)) ncPowerSet_.insert(key, &world_.error_);
        }
        return true;
    }

    MCPGPU_HD void trigger(BlockPos pos) {
        std::uint32_t state = world_.getBlock(pos);
        if (blockId(state) == BLOCK_OBSERVER) {
            scheduleUpdate(pos, BLOCK_OBSERVER, 2);
        } else {
            neighborChanged(pos, 95, pos);
        }
    }

    // --- tick loop -----------------------------------------------------------------------

    MCPGPU_HD void tickWorld() {
        ++world_.time;
        tickScheduledUpdates();
        sendQueuedBlockEvents();
        updateEntities();
    }

    MCPGPU_HD void tickScheduledUpdates() {
        // Insertion sort by (time, order) ascending - scheduledTicks_ is capped at 512 entries,
        // so O(n^2) here is not a real cost; same total order std::sort would produce since
        // (time, order) is already a strict weak ordering with no ties (order is unique).
        int n = world_.scheduledTickCount_;
        ScheduledTick* arr = world_.scheduledTicks_;
        for (int i = 1; i < n; ++i) {
            ScheduledTick key = arr[i];
            int j = i - 1;
            while (j >= 0 && (arr[j].time > key.time || (arr[j].time == key.time && arr[j].order > key.order))) {
                arr[j + 1] = arr[j];
                --j;
            }
            arr[j + 1] = key;
        }

        int dueCount = 0;
        while (dueCount < n && arr[dueCount].time <= world_.time) ++dueCount;

        for (int i = 0; i < dueCount; ++i) pendingDueScratch_[i] = arr[i];
        int remaining = n - dueCount;
        for (int i = 0; i < remaining; ++i) arr[i] = arr[dueCount + i];
        world_.scheduledTickCount_ = remaining;

        for (int i = 0; i < dueCount; ++i) {
            const ScheduledTick& tick = pendingDueScratch_[i];
            std::uint32_t state = world_.getBlock(tick.pos);
            if (blockId(state) != tick.blockId) continue;
            if (tick.blockId == BLOCK_OBSERVER) {
                observerUpdateTick(tick.pos, state);
            } else if (tick.blockId == BLOCK_LIT_REDSTONE_LAMP) {
                lampUpdateTick(tick.pos, state);
            }
        }
    }

    MCPGPU_HD void lampUpdateTick(BlockPos pos, std::uint32_t state) {
        if (blockId(state) == BLOCK_LIT_REDSTONE_LAMP && !isBlockPowered(pos)) {
            setBlockState(pos, makeState(BLOCK_REDSTONE_LAMP, 0), 2);
        }
    }

    MCPGPU_HD void notifyObserverFront(BlockPos pos, std::uint32_t state) {
        const Facing& facing = facingByIndex(facingMeta(state));
        BlockPos front = offset(pos, opposite(facing));
        neighborChanged(front, BLOCK_OBSERVER, pos);
        notifyNeighborsExcept(front, BLOCK_OBSERVER, facing.index);
    }

    MCPGPU_HD void observerUpdateTick(BlockPos pos, std::uint32_t state) {
        if (metaBit(state, 3)) {
            setBlockState(pos, setMetaBit(state, 3, false), 2);
        } else {
            setBlockState(pos, setMetaBit(state, 3, true), 2);
            scheduleUpdate(pos, BLOCK_OBSERVER, 2);
        }
        notifyObserverFront(pos, state);
    }

    MCPGPU_HD void sendQueuedBlockEvents() {
        while (world_.blockEventCount_[world_.blockEventCacheIndex] != 0) {
            int index = world_.blockEventCacheIndex;
            world_.blockEventCacheIndex ^= 1;
            int n = world_.blockEventCount_[index];
            for (int i = 0; i < n; ++i) eventsScratch_[i] = world_.blockEvents_[index][i];
            world_.blockEventCount_[index] = 0;
            for (int i = 0; i < n; ++i) fireBlockEvent(eventsScratch_[i]);
        }
    }

    MCPGPU_HD bool fireBlockEvent(const BlockEvent& event) {
        std::uint32_t state = world_.getBlock(event.pos);
        if (blockId(state) != event.blockId) return false;
        if (event.blockId == BLOCK_PISTON || event.blockId == BLOCK_STICKY_PISTON) {
            return pistonEventReceived(event.pos, state, event.eventId, event.eventParam);
        }
        return false;
    }

    MCPGPU_HD void updateEntities() {
        int oldest = ((world_.movingPtr + 1) % 3 + 3) % 3;

        int doneCount = world_.movingBucketCount_[oldest];
        for (int i = 0; i < doneCount; ++i) doneScratch_[i] = world_.movingBuckets_[oldest][i];
        world_.movingBucketCount_[oldest] = 0;
        world_.movingPtr = ((world_.movingPtr + 1) % 3 + 3) % 3;

        for (int i = 0; i < doneCount; ++i) settleMovingBlock(doneScratch_[i]);
    }

    MCPGPU_HD void settleMovingBlock(const MovingBlock& moving) {
        if (blockId(world_.getBlock(moving.pos)) == BLOCK_PISTON_EXTENSION) {
            setBlockState(moving.pos, moving.pistonState, 3);
            neighborChanged(moving.pos, moving.pistonBlockId, moving.pos);
        }
    }

    MCPGPU_HD void addMovingBlock(const MovingBlock& block) { world_.pushMovingBlock(world_.movingPtr, block); }

    MCPGPU_HD void scheduleUpdate(BlockPos pos, int blockIdValue, int delay) {
        if (blockIdValue == BLOCK_AIR || pos.y < 0 || pos.y >= 256) return;
        if (isUpdateScheduled(pos, blockIdValue)) return;
        ScheduledTick tick;
        tick.time = world_.time + delay;
        tick.order = world_.nextTickOrder++;
        tick.pos = pos;
        tick.blockId = blockIdValue;
        world_.pushScheduledTick(tick);
    }

    MCPGPU_HD bool isUpdateScheduled(BlockPos pos, int blockIdValue) const {
        for (int i = 0; i < world_.scheduledTickCount_; ++i) {
            const ScheduledTick& tick = world_.scheduledTicks_[i];
            if (tick.blockId == blockIdValue && detail::samePos(tick.pos, pos)) return true;
        }
        return false;
    }

    MCPGPU_HD void addBlockEvent(BlockPos pos, int blockIdValue, int eventId, int eventParam) {
        int index = world_.blockEventCacheIndex;
        int n = world_.blockEventCount_[index];
        for (int i = 0; i < n; ++i) {
            const BlockEvent& event = world_.blockEvents_[index][i];
            if (event.blockId == blockIdValue && event.eventId == eventId && event.eventParam == eventParam &&
                detail::samePos(event.pos, pos)) {
                return;
            }
        }
        world_.pushBlockEvent(index, BlockEvent{pos, blockIdValue, eventId, eventParam});
    }

    // --- neighbor propagation --------------------------------------------------------------

    MCPGPU_HD void neighborChangedImpl(BlockPos pos, int sourceBlockId, BlockPos fromPos) {
        std::uint32_t state = world_.getBlock(pos);
        int id = blockId(state);
        if (id == BLOCK_PISTON_HEAD) {
            const Facing& facing = facingByIndex(facingMeta(state));
            BlockPos base = offset(pos, opposite(facing));
            int baseId = blockId(world_.getBlock(base));
            if (baseId != BLOCK_PISTON && baseId != BLOCK_STICKY_PISTON) {
                setBlockToAir(pos);
            } else {
                neighborChanged(base, sourceBlockId, fromPos);
            }
        } else if (id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON) {
            checkForMove(pos, state);
        } else if (id == BLOCK_FENCE_GATE) {
            bool powered = isBlockPowered(pos);
            if (metaBit(state, 3) != powered) {
                std::uint32_t newState = setMetaBit(setMetaBit(state, 3, powered), 2, powered);
                setBlockState(pos, newState, 2);
            }
        } else if (id == BLOCK_LIT_REDSTONE_LAMP) {
            if (!isBlockPowered(pos)) scheduleUpdate(pos, BLOCK_LIT_REDSTONE_LAMP, 4);
        } else if (id == BLOCK_REDSTONE_LAMP) {
            if (isBlockPowered(pos)) setBlockState(pos, makeState(BLOCK_LIT_REDSTONE_LAMP, 0), 2);
        } else if (isRailBlock(id)) {
            railNeighborChanged(pos, state);
        }
    }

    MCPGPU_HD void neighborChanged(BlockPos pos, int sourceBlockId, BlockPos fromPos) {
        if (watchSet_.empty()) return;
        std::uint64_t key = packPos(pos);
        if (!watchSet_.count(key)) return;
        neighborChangedImpl(pos, sourceBlockId, fromPos);
    }

    MCPGPU_HD void observedNeighborChanged(BlockPos pos, int changedBlockId, BlockPos changedPos) {
        if (observerSet_.empty()) return;
        std::uint64_t key = packPos(pos);
        if (!observerSet_.contains(key)) return;
        std::uint32_t state = world_.getBlock(pos);
        if (blockId(state) != BLOCK_OBSERVER) return;
        const Facing& facing = facingByIndex(facingMeta(state));
        BlockPos watched = offset(pos, facing);
        if (detail::samePos(watched, changedPos) && !metaBit(state, 3) && !isUpdateScheduled(pos, BLOCK_OBSERVER)) {
            scheduleUpdate(pos, BLOCK_OBSERVER, 2);
        }
        (void)changedBlockId;
    }

    MCPGPU_HD void notifyNeighbors(BlockPos pos, int sourceBlockId, bool updateObservers) {
        if (!watchSet_.empty()) {
            BlockPos neighbors[6] = {
                {pos.x - 1, pos.y, pos.z}, {pos.x + 1, pos.y, pos.z},
                {pos.x, pos.y - 1, pos.z}, {pos.x, pos.y + 1, pos.z},
                {pos.x, pos.y, pos.z - 1}, {pos.x, pos.y, pos.z + 1},
            };
            for (int i = 0; i < 6; ++i) {
                std::uint64_t k = packPos(neighbors[i]);
                if (watchSet_.count(k)) neighborChangedImpl(neighbors[i], sourceBlockId, pos);
            }
        }
        if (updateObservers) updateObservingBlocksAt(pos, sourceBlockId);
    }

    MCPGPU_HD void notifyNeighborsExcept(BlockPos pos, int sourceBlockId, int skipFacing) {
        if (watchSet_.empty()) return;
        const int javaOrder[6] = {4, 5, 0, 1, 2, 3};
        for (int idx = 0; idx < 6; ++idx) {
            int index = javaOrder[idx];
            if (index != skipFacing) {
                BlockPos n = offset(pos, facingByIndex(index));
                std::uint64_t k = packPos(n);
                if (watchSet_.count(k)) neighborChangedImpl(n, sourceBlockId, pos);
            }
        }
    }

    MCPGPU_HD void updateObservingBlocksAt(BlockPos pos, int sourceBlockId) {
        if (observerSet_.empty()) return;
        observedNeighborChanged({pos.x - 1, pos.y, pos.z}, sourceBlockId, pos);
        observedNeighborChanged({pos.x + 1, pos.y, pos.z}, sourceBlockId, pos);
        observedNeighborChanged({pos.x, pos.y - 1, pos.z}, sourceBlockId, pos);
        observedNeighborChanged({pos.x, pos.y + 1, pos.z}, sourceBlockId, pos);
        observedNeighborChanged({pos.x, pos.y, pos.z - 1}, sourceBlockId, pos);
        observedNeighborChanged({pos.x, pos.y, pos.z + 1}, sourceBlockId, pos);
    }

    // --- piston check / power -------------------------------------------------------------

    MCPGPU_HD void checkForMove(BlockPos pos, std::uint32_t state) {
        int id = blockId(state);
        const Facing& facing = facingByIndex(facingMeta(state));
        bool shouldExtend = shouldPistonBeExtended(pos, state);
        if (shouldExtend && !metaBit(state, 3)) {
            PistonStructureHelper helper(world_, pos, facing, true);
            if (helper.canMove()) addBlockEvent(pos, id, 0, facing.index);
        } else if (!shouldExtend && metaBit(state, 3)) {
            addBlockEvent(pos, id, 1, facing.index);
        }
    }

    MCPGPU_HD bool shouldPistonBeExtended(BlockPos pos, std::uint32_t state) const {
        const Facing& pistonFacing = facingByIndex(facingMeta(state));
        for (int i = 0; i < 6; ++i) {
            const Facing& facing = facingByIndex(i);
            if (facing.index != pistonFacing.index) {
                if (isSidePowered(offset(pos, facing), facing)) return true;
            }
        }
        if (isSidePowered(pos, facingByIndex(0))) return true;
        BlockPos up{pos.x, pos.y + 1, pos.z};
        for (int i = 0; i < 6; ++i) {
            const Facing& facing = facingByIndex(i);
            if (facing.index != 0) {
                if (isSidePowered(offset(up, facing), facing)) return true;
            }
        }
        return false;
    }

    MCPGPU_HD bool isTopSolid(BlockPos pos) const {
        std::uint32_t state = world_.getBlock(pos);
        int id = blockId(state);
        if (id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON) {
            return !metaBit(state, 3) || facingMeta(state) == 0;
        }
        return blockData(id).isNormalCube;
    }

    MCPGPU_HD bool isBlockPowered(BlockPos pos) const {
        return redstonePower({pos.x, pos.y - 1, pos.z}, facingByIndex(0)) > 0 ||
               redstonePower({pos.x, pos.y + 1, pos.z}, facingByIndex(1)) > 0 ||
               redstonePower({pos.x, pos.y, pos.z - 1}, facingByIndex(2)) > 0 ||
               redstonePower({pos.x, pos.y, pos.z + 1}, facingByIndex(3)) > 0 ||
               redstonePower({pos.x - 1, pos.y, pos.z}, facingByIndex(4)) > 0 ||
               redstonePower({pos.x + 1, pos.y, pos.z}, facingByIndex(5)) > 0;
    }

    MCPGPU_HD bool isSidePowered(BlockPos pos, const Facing& side) const { return redstonePower(pos, side) > 0; }

    MCPGPU_HD int redstonePower(BlockPos pos, const Facing& side) const {
        std::uint32_t state = world_.getBlock(pos);
        if (blockData(blockId(state)).isNormalCube) {
            if (ncPowerSet_.empty()) return 0;
            return strongPowerAll(pos);
        }
        return weakPower(state, side);
    }

    MCPGPU_HD int weakPower(std::uint32_t state, const Facing& side) const {
        int id = blockId(state);
        if (id == BLOCK_REDSTONE_BLOCK) return 15;
        if (id == BLOCK_OBSERVER && metaBit(state, 3) && facingMeta(state) == side.index) return 15;
        return 0;
    }

    MCPGPU_HD int strongPower(BlockPos pos, const Facing& side) const {
        std::uint32_t state = world_.getBlock(pos);
        int id = blockId(state);
        if (id == BLOCK_OBSERVER) return weakPower(state, side);
        if (id == BLOCK_DETECTOR_RAIL && metaBit(state, 3) && side.index == 1) return 15;
        return 0;
    }

    MCPGPU_HD int strongPowerAll(BlockPos pos) const {
        int out = 0;
        for (int i = 0; i < 6; ++i) {
            const Facing& facing = facingByIndex(i);
            int power = strongPower(offset(pos, facing), facing);
            if (power > out) out = power;
        }
        return out;
    }

    // --- rail power ------------------------------------------------------------------------

    MCPGPU_HD void railNeighborChanged(BlockPos pos, std::uint32_t state) {
        int id = blockId(state);
        int shape = railShape(id, state);
        bool unsupported = !isTopSolid(offset(pos, facingByIndex(0)));

        if (!unsupported) {
            switch (shape) {
                case RAIL_SHAPE_ASCENDING_EAST: unsupported = !isTopSolid(offset(pos, facingByIndex(5))); break;
                case RAIL_SHAPE_ASCENDING_WEST: unsupported = !isTopSolid(offset(pos, facingByIndex(4))); break;
                case RAIL_SHAPE_ASCENDING_NORTH: unsupported = !isTopSolid(offset(pos, facingByIndex(2))); break;
                case RAIL_SHAPE_ASCENDING_SOUTH: unsupported = !isTopSolid(offset(pos, facingByIndex(3))); break;
                default: break;
            }
        }

        if (unsupported) {
            if (!isAirState(world_.getBlock(pos))) setBlockToAir(pos);
            return;
        }

        if (isPoweredRailBlock(id)) updateRailPowerState(pos, state, id, shape);
    }

    MCPGPU_HD void updateRailPowerState(BlockPos pos, std::uint32_t state, int railId, int shape) {
        bool wasPowered = metaBit(state, 3);
        bool nowPowered = isBlockPowered(pos) || findPoweredRailSignal(railId, pos, state, true, 0) ||
                           findPoweredRailSignal(railId, pos, state, false, 0);

        if (nowPowered != wasPowered) {
            setBlockState(pos, setMetaBit(state, 3, nowPowered), 3);
            notifyNeighbors(offset(pos, facingByIndex(0)), railId, false);
            if (isAscendingRailShape(shape)) notifyNeighbors(offset(pos, facingByIndex(1)), railId, false);
        }
    }

    MCPGPU_HD bool findPoweredRailSignal(int railId, BlockPos pos, std::uint32_t state, bool forward, int distance) const {
        if (distance >= 8) return false;

        int x = pos.x, y = pos.y, z = pos.z;
        bool tryOneLevelDown = true;
        int shape = railShape(railId, state);
        int searchShape = shape;

        switch (shape) {
            case RAIL_SHAPE_NORTH_SOUTH: if (forward) ++z; else --z; break;
            case RAIL_SHAPE_EAST_WEST: if (forward) --x; else ++x; break;
            case RAIL_SHAPE_ASCENDING_EAST:
                if (forward) { --x; } else { ++x; ++y; tryOneLevelDown = false; }
                searchShape = RAIL_SHAPE_EAST_WEST;
                break;
            case RAIL_SHAPE_ASCENDING_WEST:
                if (forward) { --x; ++y; tryOneLevelDown = false; } else { ++x; }
                searchShape = RAIL_SHAPE_EAST_WEST;
                break;
            case RAIL_SHAPE_ASCENDING_NORTH:
                if (forward) { ++z; } else { --z; ++y; tryOneLevelDown = false; }
                searchShape = RAIL_SHAPE_NORTH_SOUTH;
                break;
            case RAIL_SHAPE_ASCENDING_SOUTH:
                if (forward) { ++z; ++y; tryOneLevelDown = false; } else { --z; }
                searchShape = RAIL_SHAPE_NORTH_SOUTH;
                break;
            default: break;
        }

        if (isSameRailWithPower(railId, BlockPos{x, y, z}, forward, distance, searchShape)) return true;
        return tryOneLevelDown && isSameRailWithPower(railId, BlockPos{x, y - 1, z}, forward, distance, searchShape);
    }

    MCPGPU_HD bool isSameRailWithPower(int railId, BlockPos pos, bool forward, int distance, int expectedShape) const {
        std::uint32_t state = world_.getBlock(pos);
        if (blockId(state) != railId) return false;

        int shape = railShape(railId, state);
        if (expectedShape == RAIL_SHAPE_EAST_WEST &&
            (shape == RAIL_SHAPE_NORTH_SOUTH || shape == RAIL_SHAPE_ASCENDING_NORTH || shape == RAIL_SHAPE_ASCENDING_SOUTH)) {
            return false;
        }
        if (expectedShape == RAIL_SHAPE_NORTH_SOUTH &&
            (shape == RAIL_SHAPE_EAST_WEST || shape == RAIL_SHAPE_ASCENDING_EAST || shape == RAIL_SHAPE_ASCENDING_WEST)) {
            return false;
        }

        if (!metaBit(state, 3)) return false;
        if (isBlockPowered(pos)) return true;
        return findPoweredRailSignal(railId, pos, state, forward, distance + 1);
    }

    // --- piston movement ---------------------------------------------------------------------

    MCPGPU_HD bool pistonEventReceived(BlockPos pos, std::uint32_t state, int id, int param) {
        const Facing& facing = facingByIndex(facingMeta(state));
        bool shouldExtend = shouldPistonBeExtended(pos, state);
        if (shouldExtend && id == 1) {
            setBlockState(pos, setMetaBit(state, 3, true), 2);
            return false;
        }
        if (!shouldExtend && id == 0) return false;

        bool sticky = isStickyPistonBlock(blockId(state));
        if (id == 0) {
            if (!doPistonMove(pos, facing, true, sticky)) return false;
            setBlockState(pos, setMetaBit(state, 3, true), 3);
        } else if (id == 1) {
            BlockPos front = offset(pos, facing);
            clearMovingAt(front);
            setBlockState(pos, setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), facing.index), 3);
            addMovingBlock(MovingBlock{pos, makeState(blockId(state), param), blockId(state), facing.index, false, true});

            if (sticky) {
                BlockPos pull = offset(pos, facing, 2);
                std::uint32_t pullState = world_.getBlock(pull);
                int pullId = blockId(pullState);
                bool clearedExtendingMoving = false;
                if (pullId == BLOCK_PISTON_EXTENSION && isExtendingMovingAt(pull, facing.index)) {
                    clearMovingAt(pull);
                    clearedExtendingMoving = true;
                }
                if (!clearedExtendingMoving && pullId != BLOCK_AIR &&
                    canPush(pullState, world_, pull, opposite(facing), false, facing) &&
                    (blockData(pullId).pushReaction == PushReaction::Normal || pullId == BLOCK_PISTON || pullId == BLOCK_STICKY_PISTON)) {
                    doPistonMove(pos, facing, false, sticky);
                }
            } else {
                setBlockToAir(front);
            }
        }
        return true;
    }

    MCPGPU_HD bool doPistonMove(BlockPos pos, const Facing& direction, bool extending, bool sticky) {
        if (!extending) setBlockToAir(offset(pos, direction));

        PistonStructureHelper helper(world_, pos, direction, extending);
        if (!helper.canMove()) return false;

        const int moveCount = helper.moveCount();
        const int destroyCount = helper.destroyCount();

        std::uint32_t movedStates[12]{};
        for (int i = 0; i < moveCount; ++i) movedStates[i] = world_.getBlock(helper.moveAt(i));
        std::uint32_t destroyedStates[12]{};
        for (int i = 0; i < destroyCount; ++i) destroyedStates[i] = world_.getBlock(helper.destroyAt(i));

        // Same scratch-array quirk noted in reference/simulator.cpp's doPistonMove: fill/read
        // indices deliberately don't reset between the destroy and move phases - exact vanilla
        // behavior, replicated bit-for-bit.
        std::uint32_t aiblockstate[24]{};
        int k = moveCount + destroyCount;

        const Facing& moveFacing = extending ? direction : opposite(direction);
        for (int i = destroyCount - 1; i >= 0; --i) {
            setBlockState(helper.destroyAt(i), 0, 4);
            aiblockstate[--k] = destroyedStates[i];
        }
        for (int i = moveCount - 1; i >= 0; --i) {
            BlockPos source = helper.moveAt(i);
            BlockPos target = offset(source, moveFacing);
            std::uint32_t movedState = movedStates[i];
            setBlockState(source, 0, 2);
            setBlockState(target, setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, 0), direction.index), 4);
            addMovingBlock(MovingBlock{target, movedState, blockId(movedState), direction.index, extending, false});
            aiblockstate[--k] = movedState;
        }
        if (extending) {
            int headMeta = direction.index | (sticky ? 8 : 0);
            std::uint32_t movingHead = setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), direction.index);
            BlockPos front = offset(pos, direction);
            setBlockState(front, movingHead, 4);
            addMovingBlock(MovingBlock{front, makeState(BLOCK_PISTON_HEAD, headMeta), BLOCK_PISTON_HEAD, direction.index, true, true});
        }

        k = 0;
        for (int i = destroyCount - 1; i >= 0; --i) {
            notifyNeighbors(helper.destroyAt(i), blockId(aiblockstate[k++]), false);
        }
        for (int i = moveCount - 1; i >= 0; --i) {
            notifyNeighbors(helper.moveAt(i), blockId(aiblockstate[k++]), false);
        }
        if (extending) notifyNeighbors(offset(pos, direction), BLOCK_PISTON_HEAD, false);
        return true;
    }

    // --- state mutation ----------------------------------------------------------------------

    MCPGPU_HD void setBlockState(BlockPos pos, std::uint32_t state, int flags) {
        if (pos.y < 0 || pos.y >= 256) return;
        std::uint32_t oldState = world_.getBlock(pos);
        if (oldState == state) return;
        world_.setBlock(pos, state);

        int oldId = blockId(oldState);
        int newId = blockId(state);

        if (oldId != newId) {
            std::uint64_t setKey = packPos(pos);
            if (oldId == BLOCK_OBSERVER) observerSet_.erase(setKey);
            if (newId == BLOCK_OBSERVER) observerSet_.insert(setKey, &world_.error_);
            bool oldW = detail::isWatchedBlock(oldId);
            bool newW = detail::isWatchedBlock(newId);
            if (oldW) watchSet_.erase(setKey);
            if (newW) watchSet_.insert(setKey, &world_.error_);
        }

        bool wasPS = detail::isNcPowerSource(oldId, oldState);
        bool isPS = detail::isNcPowerSource(newId, state);
        if (wasPS != isPS) {
            std::uint64_t setKey = packPos(pos);
            if (wasPS) ncPowerSet_.erase(setKey);
            if (isPS) ncPowerSet_.insert(setKey, &world_.error_);
        }

        if (oldId != newId && oldId == BLOCK_PISTON_EXTENSION) removeMovingAt(pos);
        if (oldId != newId && oldId == BLOCK_PISTON_HEAD) {
            const Facing& headFacing = facingByIndex(facingMeta(oldState));
            BlockPos base = offset(pos, opposite(headFacing));
            std::uint32_t baseState = world_.getBlock(base);
            int baseId = blockId(baseState);
            if ((baseId == BLOCK_PISTON || baseId == BLOCK_STICKY_PISTON) && metaBit(baseState, 3)) {
                setBlockToAir(base);
            }
        }
        if (oldId != newId && newId == BLOCK_OBSERVER) {
            if (metaBit(state, 3)) {
                observerUpdateTick(pos, state);
            } else if (!isUpdateScheduled(pos, BLOCK_OBSERVER)) {
                scheduleUpdate(pos, BLOCK_OBSERVER, 2);
            }
        }
        if (oldId != newId && oldId == BLOCK_OBSERVER) {
            if (metaBit(oldState, 3) && isUpdateScheduled(pos, BLOCK_OBSERVER)) {
                notifyObserverFront(pos, oldState);
            }
        }
        if (oldId != newId && (newId == BLOCK_PISTON || newId == BLOCK_STICKY_PISTON) && !hasMovingAt(pos)) {
            checkForMove(pos, state);
        }
        if (oldId != newId && isRailBlock(oldId)) {
            int oldShape = railShape(oldId, oldState);
            if (isAscendingRailShape(oldShape)) notifyNeighbors(offset(pos, facingByIndex(1)), oldId, false);
            if (oldId != BLOCK_RAIL) {
                notifyNeighbors(pos, oldId, false);
                notifyNeighbors(offset(pos, facingByIndex(0)), oldId, false);
            }
        }
        if ((flags & 1) != 0) {
            notifyNeighbors(pos, oldId, true);
        } else if ((flags & 16) == 0) {
            updateObservingBlocksAt(pos, newId);
        }
    }

    MCPGPU_HD void setBlockToAir(BlockPos pos) { setBlockState(pos, 0, 3); }

    MCPGPU_HD void removeMovingAt(BlockPos pos) {
        for (int b = 0; b < 3; ++b) {
            int n = world_.movingBucketCount_[b];
            int out = 0;
            for (int i = 0; i < n; ++i) {
                if (!detail::samePos(world_.movingBuckets_[b][i].pos, pos)) {
                    world_.movingBuckets_[b][out++] = world_.movingBuckets_[b][i];
                }
            }
            world_.movingBucketCount_[b] = out;
        }
    }

    MCPGPU_HD bool clearMovingAt(BlockPos pos) {
        for (int b = 0; b < 3; ++b) {
            int n = world_.movingBucketCount_[b];
            for (int i = 0; i < n; ++i) {
                if (!detail::samePos(world_.movingBuckets_[b][i].pos, pos)) continue;
                MovingBlock moving = world_.movingBuckets_[b][i];
                // Order-preserving removal (shift the tail down), matching reference/'s
                // std::vector::erase(it) - NOT swap-with-last. The bucket's iteration order
                // later drives settleMovingBlock()'s side-effecting cascade in updateEntities(),
                // so reordering it here would silently change which neighbor-update happens
                // first and diverge the simulation, not just the storage representation.
                for (int j = i; j < n - 1; ++j) world_.movingBuckets_[b][j] = world_.movingBuckets_[b][j + 1];
                world_.movingBucketCount_[b] = n - 1;
                if (blockId(world_.getBlock(pos)) == BLOCK_PISTON_EXTENSION) {
                    setBlockState(pos, moving.pistonState, 3);
                    neighborChanged(pos, moving.pistonBlockId, pos);
                }
                return true;
            }
        }
        return false;
    }

    MCPGPU_HD bool hasMovingAt(BlockPos pos) const {
        for (int b = 0; b < 3; ++b) {
            int n = world_.movingBucketCount_[b];
            for (int i = 0; i < n; ++i) {
                if (detail::samePos(world_.movingBuckets_[b][i].pos, pos)) return true;
            }
        }
        return false;
    }

    MCPGPU_HD bool isExtendingMovingAt(BlockPos pos, int facing) const {
        for (int b = 0; b < 3; ++b) {
            int n = world_.movingBucketCount_[b];
            for (int i = 0; i < n; ++i) {
                const MovingBlock& moving = world_.movingBuckets_[b][i];
                if (detail::samePos(moving.pos, pos) && moving.facing == facing && moving.extending) return true;
            }
        }
        return false;
    }

    // --- cycle detection ---------------------------------------------------------------------

    MCPGPU_HD static std::uint32_t seenHash(const StateKey& k) {
        std::uint64_t h = k.words[4] ^ k.words[5] ^ k.words[6] ^ k.words[7] ^ k.words[8] ^ k.words[9] ^ k.words[10] ^ k.words[11];
        h = detail::mix64(h);
        return static_cast<std::uint32_t>(h) & (kSeenCapacity - 1);
    }

    // Returns true and fills *foundTick/*foundAnchor if key was already present; otherwise
    // inserts it (recording currentTick/currentAnchor for a future match to return) and returns
    // false. Sets world_.error_ (table exhausted) only if maxTicks was pushed far past the
    // default 6000 - see class doc comment.
    MCPGPU_HD bool seenFind(int currentTick, BlockPos currentAnchor, const StateKey& key, int* foundTick,
                             BlockPos* foundAnchor) {
        std::uint32_t slot = seenHash(key);
        for (int probes = 0; probes < kSeenCapacity; ++probes) {
            SeenSlot& s = seen_[slot];
            if (!s.used) {
                s.used = true;
                s.key = key;
                s.tick = currentTick;
                s.anchor = currentAnchor;
                return false;
            }
            if (s.key.equals(key)) {
                *foundTick = s.tick;
                *foundAnchor = s.anchor;
                return true;
            }
            slot = (slot + 1) & (kSeenCapacity - 1);
        }
        world_.error_ = true;
        return false;
    }

#if !defined(__CUDACC__)
    static void debugDumpKey(int tick, const StateKey& key, BlockPos anchor) {
        if (std::getenv("MCP1122_DEBUG_STATEKEY") == nullptr) return;
        std::cerr << "GPU " << tick << " anchor=" << anchor.x << "," << anchor.y << "," << anchor.z;
        for (int i = 0; i < 12; ++i) std::cerr << ' ' << key.words[i];
        std::cerr << '\n';
    }
#endif

    MCPGPU_HD bool detectShiftCycle(int maxTicks, ShiftCycle& out) {
        for (int i = 0; i < kSeenCapacity; ++i) seen_[i].used = false;

        BlockPos anchor;
        StateKey key = stateKey(&anchor);
#if !defined(__CUDACC__)
        debugDumpKey(0, key, anchor);
#endif
        int firstSlot = static_cast<int>(seenHash(key));
        seen_[firstSlot].used = true;
        seen_[firstSlot].key = key;
        seen_[firstSlot].tick = 0;
        seen_[firstSlot].anchor = anchor;
        // Above assumes the very first insert never collides into an already-used slot, which
        // holds because the table was just cleared - equivalent to seenFind()'s insert branch.

        for (int tick = 1; tick <= maxTicks; ++tick) {
            tickWorld();
            if (world_.error_) return false;
            key = stateKey(&anchor);
#if !defined(__CUDACC__)
            debugDumpKey(tick, key, anchor);
#endif
            int seenTick = 0;
            BlockPos seenAnchor;
            if (seenFind(tick, anchor, key, &seenTick, &seenAnchor)) {
                BlockPos shift{anchor.x - seenAnchor.x, anchor.y - seenAnchor.y, anchor.z - seenAnchor.z};
                if (!detail::samePos(shift, BlockPos{0, 0, 0})) {
                    out.start = seenTick;
                    out.end = tick;
                    out.period = tick - seenTick;
                    out.shift = shift;
                    return true;
                }
            }
            if (world_.error_) return false;
        }
        return false;
    }

    MCPGPU_HD StateKey stateKey(BlockPos* anchorOut) const {
        BlockPos blockAnchor{0, 0, 0};
        bool haveBlockAnchor = false;
        for (int i = 0; i < world_.entryCount_; ++i) {
            const FixedWorld::Entry& e = world_.entries_[i];
            if (detail::anchorBeats(e.pos, blockAnchor, haveBlockAnchor)) {
                blockAnchor = e.pos;
                haveBlockAnchor = true;
            }
        }

        BlockPos anchor = blockAnchor;
        bool haveAnchor = haveBlockAnchor;
        auto consider = [&](BlockPos pos) {
            if (detail::anchorBeats(pos, anchor, haveAnchor)) { anchor = pos; haveAnchor = true; }
        };
        for (int q = 0; q < 2; ++q) {
            for (int i = 0; i < world_.blockEventCount_[q]; ++i) consider(world_.blockEvents_[q][i].pos);
        }
        for (int i = 0; i < world_.scheduledTickCount_; ++i) consider(world_.scheduledTicks_[i].pos);
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < world_.movingBucketCount_[b]; ++i) consider(world_.movingBuckets_[b][i].pos);
        }
        if (!haveAnchor) anchor = BlockPos{0, 0, 0};
        *anchorOut = anchor;

        std::uint64_t blockA = 0x9e3779b97f4a7c15ULL, blockB = 0xc2b2ae3d27d4eb4fULL;
        if (haveAnchor) {
            for (int i = 0; i < world_.entryCount_; ++i) {
                const FixedWorld::Entry& e = world_.entries_[i];
                std::uint64_t enc = detail::encodeRel(e.pos.x - anchor.x, e.pos.y - anchor.y, e.pos.z - anchor.z) ^
                                     (static_cast<std::uint64_t>(e.state) << 17);
                std::uint64_t m = detail::mix64(enc);
                blockA += m;
                blockB ^= (m << 27) | (m >> 37);
            }
        }

        std::uint64_t eventA = 0x165667b19e3779f9ULL, eventB = 0x85ebca77c2b2ae63ULL, eventCount = 0;
        for (int q = 0; q < 2; ++q) {
            for (int i = 0; i < world_.blockEventCount_[q]; ++i) {
                const BlockEvent& event = world_.blockEvents_[q][i];
                std::uint64_t encoded = detail::encodeRel(event.pos.x - anchor.x, event.pos.y - anchor.y, event.pos.z - anchor.z) ^
                    (static_cast<std::uint64_t>(event.blockId & 0xfff) << 13) ^
                    (static_cast<std::uint64_t>(event.eventId & 0xff) << 25) ^
                    (static_cast<std::uint64_t>(event.eventParam & 0xff) << 33);
                std::uint64_t mixed = detail::mix64(encoded);
                eventA += mixed;
                eventB ^= (mixed << 31) | (mixed >> 33);
                ++eventCount;
            }
        }

        std::uint64_t tickA = 0xd6e8feb86659fd93ULL, tickB = 0xa5a3564e27f88621ULL;
        for (int i = 0; i < world_.scheduledTickCount_; ++i) {
            const ScheduledTick& tick = world_.scheduledTicks_[i];
            std::uint64_t encoded = detail::encodeRel(tick.pos.x - anchor.x, tick.pos.y - anchor.y, tick.pos.z - anchor.z) ^
                (static_cast<std::uint64_t>(tick.blockId & 0xfff) << 11) ^
                (static_cast<std::uint64_t>(tick.time - world_.time) << 32);
            std::uint64_t mixed = detail::mix64(encoded);
            tickA += mixed;
            tickB ^= (mixed << 43) | (mixed >> 21);
        }

        std::uint64_t movingA = 0x94d049bb133111ebULL, movingB = 0xbf58476d1ce4e5b9ULL, movingCount = 0;
        for (int phase = 0; phase < 3; ++phase) {
            int bucketIndex = ((world_.movingPtr - phase) % 3 + 3) % 3;
            for (int i = 0; i < world_.movingBucketCount_[bucketIndex]; ++i) {
                const MovingBlock& moving = world_.movingBuckets_[bucketIndex][i];
                std::uint64_t encoded = detail::encodeRel(moving.pos.x - anchor.x, moving.pos.y - anchor.y, moving.pos.z - anchor.z) ^
                    (static_cast<std::uint64_t>(moving.pistonState) << 9) ^
                    (static_cast<std::uint64_t>(phase) << 29) ^
                    (static_cast<std::uint64_t>(moving.facing) << 33) ^
                    (moving.extending ? 0x100000001b3ULL : 0ULL) ^
                    (moving.shouldHeadBeRendered ? 0x9e3779b97f4a7c15ULL : 0ULL);
                std::uint64_t mixed = detail::mix64(encoded);
                movingA += mixed;
                movingB ^= (mixed << 19) | (mixed >> 45);
                ++movingCount;
            }
        }

        StateKey key;
        key.words[0] = static_cast<std::uint64_t>(world_.entryCount_);
        key.words[1] = eventCount;
        key.words[2] = static_cast<std::uint64_t>(world_.scheduledTickCount_);
        key.words[3] = movingCount;
        key.words[4] = blockA;
        key.words[5] = blockB;
        key.words[6] = eventA;
        key.words[7] = eventB;
        key.words[8] = tickA;
        key.words[9] = tickB;
        key.words[10] = movingA;
        key.words[11] = movingB;
        return key;
    }
};

} // namespace mcp1122gpu
