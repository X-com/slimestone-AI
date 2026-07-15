#pragma once

// Device-compatible port of ../reference/world.h. Three changes from the reference version, all
// forced by "this has to run as a CUDA kernel, not just be shaped like one":
//
// 1. No std::vector/std::array-of-vector: every queue becomes a fixed-capacity C array plus a
//    count, embedded directly in FixedWorld (which itself lives in one big per-candidate device
//    allocation - see gpu_state.cuh). hashKeys_/hashSlotEntry_ stay pointers (not embedded
//    arrays) because kHashCapacity elements per worker is still sized to be sliced out of one
//    big pool per worker - the host slices one big pool per worker and hands FixedWorld the
//    pointers, same shape reference/'s std::vector was already giving it (heap-backed, not
//    stack/register-backed), just device-global instead of host-heap.
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
//
// 3. Sparse hash lookup instead of a dense kExtent^3 cube. reference/'s (and this port's earlier
//    version's) cells_/entryIndex_ were a dense array over the whole 128^3 = 2M-cell cube - 8
//    bytes/cell whether or not anything lives there. Real fixtures hold at most a few thousand
//    live blocks (see kMaxEntries below), so that dense array was ~16MB of mostly-empty storage
//    PER WORKER, and pickWorkerCount() (gpu_kernel.cu) sizes the whole kernel's parallelism off
//    how many of those fit in free device memory - inflating per-worker footprint directly
//    shrinks how many workers run concurrently. Swapping to an open-addressing hash table keyed
//    by the block's absolute packPos() (same packing PosKeySet already uses) drops per-worker
//    storage from kCellCount*8 bytes to kHashCapacity*12 bytes - about 20x smaller - which lets
//    pickWorkerCount() admit proportionally more workers for the same memory budget. inRange()
//    is unchanged and still gates every access (see reference/world.h: a candidate whose bbox
//    doesn't fit in a kExtent cube around its origin is a genuine load_error, not a storage
//    artifact) - only the backing store for positions inside that cube changed shape.

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

namespace detail {

// Same murmur3-style finalizer as simulator.cuh's detail::mix64, duplicated locally so
// world.cuh doesn't need to reorder its include relative to simulator.cuh (which includes
// world.cuh, not the other way around) just to share four lines of bit-mixing.
MCPGPU_HD inline std::uint32_t hashSlotFor(std::uint64_t key, int capacityMask) {
    std::uint64_t x = key;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<std::uint32_t>(x) & static_cast<std::uint32_t>(capacityMask);
}

} // namespace detail

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

// Fixed-capacity, GPU-shaped world storage - see file header. Every worker's FixedWorld gets its
// own kHashCapacity-sized slice of a big device pool for hashKeys_/hashSlotEntry_ (bindCube()),
// and owns its queues/entries directly (embedded fixed arrays).
class FixedWorld {
public:
    static constexpr int kExtent = 128;
    static constexpr int kMargin = 16;

    // Open-addressing hash table capacity backing position lookups (see file header point 3).
    // Must be a power of two (slot index = hash & (kHashCapacity - 1)). kMaxEntries (below) is
    // 16384 live blocks at most, so kHashCapacity=65536 keeps load factor <= 25% - comfortable
    // headroom for linear-probing performance, same "generous headroom" style as every other
    // kMax* cap in this port.
    static constexpr int kHashCapacity = 65536;

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

    // Not owned - set once by the host via bindCube() to a per-worker slice of one big device
    // allocation sized numWorkers * kHashCapacity. hashSlotEntry_[slot] == -1 means empty;
    // otherwise it's an index into entries_. hashKeys_[slot] is only meaningful when the slot is
    // occupied (packPos() of that entry's position).
    std::uint64_t* hashKeys_ = nullptr;
    std::int32_t* hashSlotEntry_ = nullptr;

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

    MCPGPU_HD void bindCube(std::uint64_t* hashKeys, std::int32_t* hashSlotEntry) {
        hashKeys_ = hashKeys;
        hashSlotEntry_ = hashSlotEntry;
    }

    // Does NOT touch hashKeys_/hashSlotEntry_ - the very first use of a worker's hash-table
    // slice relies on the host's one-shot bulk cudaMemset before the first kernel launch (see
    // gpu_kernel.cu's runBatch(), which memsets hashSlotEntry_'s pool to 0xFF so every slot
    // starts "empty"); every use after that relies on clearLiveCells() (below) having already
    // been called at the end of the previous candidate. A single GPU thread serially zeroing the
    // whole table would blow through Windows' ~2s WDDM kernel timeout on a big enough capacity,
    // so neither path re-scans the full table here - only ever O(live count) via
    // clearLiveCells().
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
    // candidates back-to-back against the same hash-table slice instead of exactly one, so
    // hashKeys_/hashSlotEntry_ entries from the PREVIOUS candidate have to be cleared before the
    // next one loads - otherwise a position the new candidate never touches would read back the
    // old candidate's stale block instead of air. entries_ is already, by construction, the
    // exact set of every non-air cell at the moment a candidate's simulate() call ends
    // (setBlock() keeps it in sync incrementally), so erasing exactly those O(live count) hash
    // entries is enough - no full-table rescan needed. Unlike the old dense-cube version, this
    // no longer depends on origin (packPos() keys on absolute position), so it may run either
    // before or after setOrigin() for the next candidate.
    MCPGPU_HD void clearLiveCells() {
        for (int i = 0; i < entryCount_; ++i) {
            hashErase(packPos(entries_[i].pos));
        }
    }

    MCPGPU_HD void setOrigin(BlockPos newOrigin) { origin = newOrigin; }

    // Bounding-box gate, not a storage-indexing detail: a candidate whose live blocks (plus
    // margin) don't fit in a kExtent cube around its origin is a genuine load_error in both this
    // port and reference/ (see reference/world.h) - the hash table backing positions *inside*
    // that cube has no bearing on this check.
    MCPGPU_HD bool inRange(BlockPos pos) const {
        int lx = pos.x - origin.x, ly = pos.y - origin.y, lz = pos.z - origin.z;
        return lx >= 0 && lx < kExtent && ly >= 0 && ly < kExtent && lz >= 0 && lz < kExtent;
    }

    // Finds the hash slot currently holding `key`, or -1 if absent. No tombstones: hashErase()
    // below uses backward-shift deletion to keep every probe chain contiguous, so an empty slot
    // (hashSlotEntry_ == -1) reliably means "not found", not "possibly found further along".
    MCPGPU_HD int findSlot(std::uint64_t key) const {
        std::uint32_t slot = detail::hashSlotFor(key, kHashCapacity - 1);
        for (int probes = 0; probes < kHashCapacity; ++probes) {
            std::int32_t entryIdx = hashSlotEntry_[slot];
            if (entryIdx == -1) return -1;
            if (hashKeys_[slot] == key) return static_cast<int>(slot);
            slot = (slot + 1) & (kHashCapacity - 1);
        }
        return -1;
    }

    MCPGPU_HD void hashInsert(std::uint64_t key, std::int32_t entryIdx) {
        std::uint32_t slot = detail::hashSlotFor(key, kHashCapacity - 1);
        for (int probes = 0; probes < kHashCapacity; ++probes) {
            if (hashSlotEntry_[slot] == -1) {
                hashKeys_[slot] = key;
                hashSlotEntry_[slot] = entryIdx;
                return;
            }
            slot = (slot + 1) & (kHashCapacity - 1);
        }
        // Unreachable at kMaxEntries <= 16384 live blocks against a 65536-slot table (<=25%
        // load factor always leaves an empty slot), but fail loudly the same way every other
        // capacity ceiling in this port does rather than silently dropping the insert.
        error_ = true;
    }

    // Standard backward-shift deletion for linear-probed open addressing (see e.g. Knuth vol 3):
    // walks forward from the freed slot and pulls back any entry whose own probe start doesn't
    // "wrap past" the gap, so every remaining entry stays reachable by findSlot()'s scan without
    // ever needing a tombstone marker.
    MCPGPU_HD void hashErase(std::uint64_t key) {
        int slot = findSlot(key);
        if (slot < 0) return;
        hashSlotEntry_[slot] = -1;

        int i = slot;
        int j = (i + 1) & (kHashCapacity - 1);
        while (hashSlotEntry_[j] != -1) {
            int home = static_cast<int>(detail::hashSlotFor(hashKeys_[j], kHashCapacity - 1));
            bool shouldMove = (i <= j) ? (home <= i || home > j) : (home <= i && home > j);
            if (shouldMove) {
                hashKeys_[i] = hashKeys_[j];
                hashSlotEntry_[i] = hashSlotEntry_[j];
                hashSlotEntry_[j] = -1;
                i = j;
            }
            j = (j + 1) & (kHashCapacity - 1);
        }
    }

    MCPGPU_HD void setBlock(BlockPos pos, std::uint32_t state) {
        if (!inRange(pos)) { error_ = true; return; }
        std::uint64_t key = packPos(pos);
        int slot = findSlot(key);
        std::uint32_t old = (slot >= 0) ? entries_[hashSlotEntry_[slot]].state : 0;
        if (old == state) return;

        if (state == 0) {
            int ei = hashSlotEntry_[slot];
            int lastEi = entryCount_ - 1;
            if (ei != lastEi) {
                entries_[ei] = entries_[lastEi];
                // The moved entry's key is already in the table (pointing at lastEi) - update
                // that existing slot in place, don't hashInsert() a second slot for it.
                int movedSlot = findSlot(packPos(entries_[ei].pos));
                hashSlotEntry_[movedSlot] = ei;
            }
            --entryCount_;
            hashErase(key);
        } else if (old == 0) {
            if (entryCount_ >= kMaxEntries) { error_ = true; return; }
            hashInsert(key, entryCount_);
            if (error_) return;
            entries_[entryCount_++] = Entry{pos, state};
        } else {
            entries_[hashSlotEntry_[slot]].state = state;
        }
    }

    MCPGPU_HD void removeBlock(BlockPos pos) { setBlock(pos, 0); }

    MCPGPU_HD std::uint32_t getBlock(BlockPos pos) const {
        if (!inRange(pos)) return 0;
        int slot = findSlot(packPos(pos));
        return slot >= 0 ? entries_[hashSlotEntry_[slot]].state : 0;
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
