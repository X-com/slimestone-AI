#include "world.h"

#include <algorithm>
#include <stdexcept>

namespace mcp1122gpu {

bool PosKeySet::contains(std::uint64_t key) const {
    for (std::size_t i = 0; i < size_; ++i) {
        if (keys_[i] == key) return true;
    }
    return false;
}

void PosKeySet::insert(std::uint64_t key) {
    if (contains(key)) {
        return; // set semantics: inserting an existing key is a no-op
    }
    if (size_ >= kCapacity) {
        throw std::out_of_range("PosKeySet: exceeded kCapacity=" + std::to_string(kCapacity));
    }
    keys_[size_++] = key;
}

void PosKeySet::erase(std::uint64_t key) {
    for (std::size_t i = 0; i < size_; ++i) {
        if (keys_[i] == key) {
            keys_[i] = keys_[size_ - 1];
            --size_;
            return;
        }
    }
    // erase() on a missing key is a no-op, matching std::unordered_set::erase.
}

FixedWorld::FixedWorld() {
    const std::size_t cellCount = static_cast<std::size_t>(kExtent) * kExtent * kExtent;
    cells_.assign(cellCount, 0);
    entryIndex_.assign(cellCount, -1);
    entries_.reserve(4096);
}

void FixedWorld::reset() {
    time = 0;
    origin = BlockPos{0, 0, 0};
    std::fill(cells_.begin(), cells_.end(), 0u);
    std::fill(entryIndex_.begin(), entryIndex_.end(), -1);
    entries_.clear();
    blockEvents[0].clear();
    blockEvents[1].clear();
    blockEventCacheIndex = 0;
    scheduledTicks.clear();
    nextTickOrder = 0;
    for (auto& bucket : movingBuckets) {
        bucket.clear();
    }
    movingPtr = 0;
}

void FixedWorld::setOrigin(BlockPos newOrigin) {
    origin = newOrigin;
}

bool FixedWorld::inRange(BlockPos pos) const {
    int lx = pos.x - origin.x;
    int ly = pos.y - origin.y;
    int lz = pos.z - origin.z;
    return lx >= 0 && lx < kExtent && ly >= 0 && ly < kExtent && lz >= 0 && lz < kExtent;
}

std::size_t FixedWorld::indexOf(BlockPos pos) const {
    if (!inRange(pos)) {
        throw std::out_of_range(
            "FixedWorld: position outside fixed reference cube (extent=" +
            std::to_string(kExtent) + ") - candidate needs a larger cube or an origin recenter");
    }
    int lx = pos.x - origin.x;
    int ly = pos.y - origin.y;
    int lz = pos.z - origin.z;
    return static_cast<std::size_t>(lx) +
           static_cast<std::size_t>(ly) * kExtent +
           static_cast<std::size_t>(lz) * kExtent * kExtent;
}

void FixedWorld::setBlock(BlockPos pos, std::uint32_t state) {
    std::size_t idx = indexOf(pos);
    std::uint32_t old = cells_[idx];
    if (old == state) {
        return;
    }
    cells_[idx] = state;

    if (state == 0) {
        // Removing: swap-remove the entry with the last one, same as PosStateMap::remove().
        std::int32_t ei = entryIndex_[idx];
        std::size_t lastEi = entries_.size() - 1;
        if (static_cast<std::size_t>(ei) != lastEi) {
            entries_[static_cast<std::size_t>(ei)] = entries_[lastEi];
            std::size_t movedIdx = indexOf(entries_[static_cast<std::size_t>(ei)].pos);
            entryIndex_[movedIdx] = ei;
        }
        entries_.pop_back();
        entryIndex_[idx] = -1;
    } else if (old == 0) {
        // New entry.
        entryIndex_[idx] = static_cast<std::int32_t>(entries_.size());
        entries_.push_back(Entry{pos, state});
    } else {
        // Existing entry, state changed in place.
        entries_[static_cast<std::size_t>(entryIndex_[idx])].state = state;
    }
}

void FixedWorld::removeBlock(BlockPos pos) {
    setBlock(pos, 0);
}

std::uint32_t FixedWorld::getBlock(BlockPos pos) const {
    if (!inRange(pos)) {
        return 0; // outside the cube reads as air, same as an unset hash-map slot would
    }
    return cells_[indexOf(pos)];
}

std::size_t FixedWorld::movingCount() const {
    return movingBuckets[0].size() + movingBuckets[1].size() + movingBuckets[2].size();
}

void FixedWorld::pushBlockEvent(int queueIndex, const BlockEvent& event) {
    auto& queue = blockEvents[static_cast<std::size_t>(queueIndex)];
    if (queue.size() >= kMaxBlockEvents) {
        throw std::out_of_range("FixedWorld: block event queue exceeded kMaxBlockEvents=" +
                                 std::to_string(kMaxBlockEvents));
    }
    queue.push_back(event);
}

void FixedWorld::pushScheduledTick(const ScheduledTick& tick) {
    if (scheduledTicks.size() >= kMaxScheduledTicks) {
        throw std::out_of_range("FixedWorld: scheduledTicks exceeded kMaxScheduledTicks=" +
                                 std::to_string(kMaxScheduledTicks));
    }
    scheduledTicks.push_back(tick);
}

void FixedWorld::pushMovingBlock(int bucket, const MovingBlock& block) {
    auto& b = movingBuckets[static_cast<std::size_t>(bucket)];
    if (b.size() >= kMaxMovingPerBucket) {
        throw std::out_of_range("FixedWorld: moving bucket exceeded kMaxMovingPerBucket=" +
                                 std::to_string(kMaxMovingPerBucket));
    }
    b.push_back(block);
}

} // namespace mcp1122gpu
