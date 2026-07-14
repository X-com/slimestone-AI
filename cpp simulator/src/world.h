#pragma once

#include "packed_pos.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

namespace mcp1122 {

class PosStateMap {
public:
    // Compact record of every occupied slot — iterated by stateKey() in O(n).
    struct Entry {
        std::uint64_t key;
        std::uint32_t state;
        std::uint32_t slot;  // back-pointer into hash table (used by remove())
    };

    void clear();
    void reserve(std::size_t capacity);
    void set(std::uint64_t key, std::uint32_t state);
    void remove(std::uint64_t key);
    std::uint32_t get(std::uint64_t key) const;
    // Read old state and write new state in a single hash probe.
    // If newState == 0 the slot is deleted (equivalent to remove()).
    std::uint32_t exchange(std::uint64_t key, std::uint32_t newState);
    bool contains(std::uint64_t key) const;
    std::size_t size() const { return size_; }
    // Dense, cache-friendly list of all occupied entries.  Use this for
    // iteration instead of the old capacity()/occupiedAt(i)/keyAt(i)/stateAt(i) API.
    const std::vector<Entry>& entries() const { return entries_; }

    // Legacy slot-walk API (kept for compatibility — prefer entries() above).
    std::size_t capacity() const { return keys_.size(); }
    bool occupiedAt(std::size_t index) const { return index < meta_.size() && meta_[index] == FULL; }
    std::uint64_t keyAt(std::size_t index) const { return keys_[index]; }
    std::uint32_t stateAt(std::size_t index) const { return states_[index]; }

private:
    static constexpr std::uint8_t EMPTY = 0;
    static constexpr std::uint8_t FULL = 1;
    static constexpr std::uint8_t DELETED = 2;

    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> states_;
    std::vector<std::uint8_t>  meta_;
    std::vector<int>           slotToEntry_;  // slot index → index in entries_, or -1
    std::vector<Entry>         entries_;      // compact occupied list
    std::size_t size_ = 0;
    std::size_t tombstones_ = 0;  // DELETED slots; block probing same as FULL, must count toward resize

    void ensureCapacity();
    void rehash(std::size_t newCapacity);
    std::size_t findSlot(std::uint64_t key) const;
    std::size_t findInsertSlot(std::uint64_t key) const;
};

struct World {
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

    std::int64_t time = 0;
    PosStateMap blocks;
    std::array<std::vector<BlockEvent>, 2> blockEvents;
    int blockEventCacheIndex = 0;
    std::vector<ScheduledTick> scheduledTicks;
    int nextTickOrder = 0;
    std::array<std::vector<MovingBlock>, 3> movingBuckets;
    int movingPtr = 0;

    void reset();
    void setBlock(BlockPos pos, std::uint32_t state);
    void removeBlock(BlockPos pos);
    std::uint32_t getBlock(BlockPos pos) const;
    std::size_t liveBlocks() const { return blocks.size(); }
    std::size_t movingCount() const;
};

} // namespace mcp1122
