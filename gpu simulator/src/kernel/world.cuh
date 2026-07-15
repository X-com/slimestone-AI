#pragma once

// Device-compatible port of ../reference/world.h. Two changes from the reference version, both
// forced by "this has to run as a CUDA kernel, not just be shaped like one":
//
// 1. No std::vector/std::array-of-vector: every queue becomes a fixed-capacity C array plus a
//    count, embedded directly in FixedWorld (which itself lives in one big per-candidate device
//    allocation - see gpu_state.cuh). cells_/entryIndex_ stay pointers (not embedded arrays)
//    because kExtent^3 is 2M elements each - the host slices one big pool per candidate and
//    hands FixedWorld the pointers, same shape reference/'s std::vector was already giving it
//    (heap-backed, not stack/register-backed), just device-global instead of host-heap.
//
// 2. No exceptions: every push/set operation that would have thrown std::out_of_range instead
//    sets a sticky `error_` flag and drops the write (no mutation, no crash). This does NOT
//    change behavior for any of the 35 in-range fixtures reference/ was verified against -
//    those never hit a capacity ceiling. It only changes *how* an overflow is reported: instead
//    of an exception unwinding mid-tick, the write is silently skipped and Simulator checks
//    world_.error_ once per tick boundary (see simulator.cuh's detectShiftCycle), reporting the
//    same "simulation_error" result a caught exception would have. A write mid-tick that
//    triggers this is already the "candidate is out of scope" case reference/'s own
//    VERIFICATION.md documents (oversized trencher fixtures) - not a path any correctness-
//    verified fixture takes.

#include "gpu_common.cuh"
#include "packed_pos.cuh"

#include <cstdint>

namespace mcp1122gpu {

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

class PosKeySet {
public:
    static constexpr int kCapacity = 4096;

    MCPGPU_HD void clear() { size_ = 0; }
    MCPGPU_HD bool empty() const { return size_ == 0; }
    MCPGPU_HD int size() const { return size_; }

    MCPGPU_HD bool contains(std::uint64_t key) const {
        for (int i = 0; i < size_; ++i) {
            if (keys_[i] == key) return true;
        }
        return false;
    }
    MCPGPU_HD int count(std::uint64_t key) const { return contains(key) ? 1 : 0; }

    MCPGPU_HD void insert(std::uint64_t key, bool* error) {
        if (contains(key)) return;
        if (size_ >= kCapacity) { *error = true; return; }
        keys_[size_++] = key;
    }

    MCPGPU_HD void erase(std::uint64_t key) {
        for (int i = 0; i < size_; ++i) {
            if (keys_[i] == key) {
                keys_[i] = keys_[size_ - 1];
                --size_;
                return;
            }
        }
    }

private:
    std::uint64_t keys_[kCapacity]{};
    int size_ = 0;
};

// Fixed-capacity, GPU-shaped world storage - see file header. Every per-candidate FixedWorld
// gets its own kExtent^3 slice of a big device pool for cells_/entryIndex_ (bindCube()), and
// owns its queues/entries directly (embedded fixed arrays).
class FixedWorld {
public:
    static constexpr int kExtent = 128;
    static constexpr int kMargin = 16;
    static constexpr int kCellCount = kExtent * kExtent * kExtent;

    static constexpr int kMaxBlockEvents = 512;
    static constexpr int kMaxScheduledTicks = 512;
    static constexpr int kMaxMovingPerBucket = 16384;
    // Live occupied-cell cap. Largest known fixture (tank.json) holds 4731 blocks; pistons add
    // at most 2 transient entries (head + extension) each while moving. 16384 leaves generous
    // headroom, matching the margin style of the other kMax* constants above.
    static constexpr int kMaxEntries = 16384;

    struct Entry {
        BlockPos pos;
        std::uint32_t state;
    };

    std::int64_t time = 0;
    BlockPos origin{0, 0, 0};
    bool error_ = false;

    // Not owned - set once by the host via bindCube() to a per-candidate slice of one big
    // device allocation sized numCandidates * kCellCount.
    std::uint32_t* cells_ = nullptr;
    std::int32_t* entryIndex_ = nullptr;

    Entry entries_[kMaxEntries];
    int entryCount_ = 0;

    BlockEvent blockEvents_[2][kMaxBlockEvents];
    int blockEventCount_[2] = {0, 0};
    int blockEventCacheIndex = 0;

    ScheduledTick scheduledTicks_[kMaxScheduledTicks];
    int scheduledTickCount_ = 0;
    int nextTickOrder = 0;

    MovingBlock movingBuckets_[3][kMaxMovingPerBucket];
    int movingBucketCount_[3] = {0, 0, 0};
    int movingPtr = 0;

    MCPGPU_HD void bindCube(std::uint32_t* cells, std::int32_t* entryIndex) {
        cells_ = cells;
        entryIndex_ = entryIndex;
    }

    // Does NOT touch cells_/entryIndex_ - the very first use of a worker's cube slice relies on
    // the host's one-shot bulk cudaMemset before the first kernel launch (see gpu_kernel.cu's
    // runBatch()); every use after that relies on clearLiveCells() (below) having already been
    // called at the end of the previous candidate. A single GPU thread serially zeroing
    // 2 * kCellCount (~4M) global-memory words blows through Windows' ~2s WDDM kernel timeout
    // well before it finishes, so neither path re-scans the full cube here.
    MCPGPU_HD void reset() {
        time = 0;
        origin = BlockPos{0, 0, 0};
        error_ = false;
        entryCount_ = 0;
        blockEventCount_[0] = blockEventCount_[1] = 0;
        blockEventCacheIndex = 0;
        scheduledTickCount_ = 0;
        nextTickOrder = 0;
        movingBucketCount_[0] = movingBucketCount_[1] = movingBucketCount_[2] = 0;
        movingPtr = 0;
    }

    // Milestone I (persistent-kernel/work-queue dispatch): a worker thread now runs many
    // candidates back-to-back against the same cube slice instead of exactly one, so cells_/
    // entryIndex_ from the PREVIOUS candidate have to be cleared before the next one loads -
    // otherwise a position the new candidate never touches would read back the old candidate's
    // stale block instead of air. entries_ is already, by construction, the exact set of every
    // non-air cell at the moment a candidate's simulate() call ends (setBlock() keeps it in
    // sync incrementally), so clearing exactly those O(live count) cells is enough - no
    // full-cube rescan needed. Must be called before setOrigin() runs for the next candidate:
    // it uses the CURRENT origin to recompute each entry's index, which is only valid while
    // origin still matches the candidate that produced these entries.
    MCPGPU_HD void clearLiveCells() {
        for (int i = 0; i < entryCount_; ++i) {
            int idx = indexOf(entries_[i].pos);
            cells_[idx] = 0;
            entryIndex_[idx] = -1;
        }
    }

    MCPGPU_HD void setOrigin(BlockPos newOrigin) { origin = newOrigin; }

    MCPGPU_HD bool inRange(BlockPos pos) const {
        int lx = pos.x - origin.x, ly = pos.y - origin.y, lz = pos.z - origin.z;
        return lx >= 0 && lx < kExtent && ly >= 0 && ly < kExtent && lz >= 0 && lz < kExtent;
    }

    MCPGPU_HD int indexOf(BlockPos pos) const {
        int lx = pos.x - origin.x, ly = pos.y - origin.y, lz = pos.z - origin.z;
        return lx + ly * kExtent + lz * kExtent * kExtent;
    }

    MCPGPU_HD void setBlock(BlockPos pos, std::uint32_t state) {
        if (!inRange(pos)) { error_ = true; return; }
        int idx = indexOf(pos);
        std::uint32_t old = cells_[idx];
        if (old == state) return;
        cells_[idx] = state;

        if (state == 0) {
            std::int32_t ei = entryIndex_[idx];
            int lastEi = entryCount_ - 1;
            if (ei != lastEi) {
                entries_[ei] = entries_[lastEi];
                int movedIdx = indexOf(entries_[ei].pos);
                entryIndex_[movedIdx] = ei;
            }
            --entryCount_;
            entryIndex_[idx] = -1;
        } else if (old == 0) {
            if (entryCount_ >= kMaxEntries) { error_ = true; cells_[idx] = old; return; }
            entryIndex_[idx] = entryCount_;
            entries_[entryCount_++] = Entry{pos, state};
        } else {
            entries_[entryIndex_[idx]].state = state;
        }
    }

    MCPGPU_HD void removeBlock(BlockPos pos) { setBlock(pos, 0); }

    MCPGPU_HD std::uint32_t getBlock(BlockPos pos) const {
        if (!inRange(pos)) return 0;
        return cells_[indexOf(pos)];
    }

    MCPGPU_HD int movingCount() const {
        return movingBucketCount_[0] + movingBucketCount_[1] + movingBucketCount_[2];
    }

    MCPGPU_HD void pushBlockEvent(int queueIndex, const BlockEvent& event) {
        int& n = blockEventCount_[queueIndex];
        if (n >= kMaxBlockEvents) { error_ = true; return; }
        blockEvents_[queueIndex][n++] = event;
    }

    MCPGPU_HD void pushScheduledTick(const ScheduledTick& tick) {
        if (scheduledTickCount_ >= kMaxScheduledTicks) { error_ = true; return; }
        scheduledTicks_[scheduledTickCount_++] = tick;
    }

    MCPGPU_HD void pushMovingBlock(int bucket, const MovingBlock& block) {
        int& n = movingBucketCount_[bucket];
        if (n >= kMaxMovingPerBucket) { error_ = true; return; }
        movingBuckets_[bucket][n++] = block;
    }
};

} // namespace mcp1122gpu
