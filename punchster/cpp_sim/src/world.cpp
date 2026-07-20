#include "world.h"

#include <algorithm>

namespace mcp1122 {

namespace {

std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

std::size_t nextPowerOfTwo(std::size_t value) {
    std::size_t out = 16;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

} // namespace

void PosStateMap::clear() {
    std::fill(meta_.begin(), meta_.end(), EMPTY);
    std::fill(slotToEntry_.begin(), slotToEntry_.end(), -1);
    entries_.clear();
    size_ = 0;
    tombstones_ = 0;
}

void PosStateMap::reserve(std::size_t capacity) {
    if (keys_.size() < capacity) {
        rehash(nextPowerOfTwo(capacity * 2));
    }
    entries_.reserve(capacity);
}

void PosStateMap::set(std::uint64_t key, std::uint32_t state) {
    if (state == 0) {
        remove(key);
        return;
    }
    ensureCapacity();
    std::size_t slot = findInsertSlot(key);
    if (meta_[slot] == FULL) {
        // Update existing entry in place
        states_[slot] = state;
        entries_[static_cast<std::size_t>(slotToEntry_[slot])].state = state;
    } else {
        // New entry
        if (meta_[slot] == DELETED) --tombstones_;
        int ei = static_cast<int>(entries_.size());
        entries_.push_back(Entry{key, state, static_cast<std::uint32_t>(slot)});
        keys_[slot]        = key;
        states_[slot]      = state;
        meta_[slot]        = FULL;
        slotToEntry_[slot] = ei;
        ++size_;
    }
}

void PosStateMap::remove(std::uint64_t key) {
    if (keys_.empty()) return;
    std::size_t slot = findSlot(key);
    if (slot == keys_.size()) return;

    int ei = slotToEntry_[slot];
    std::size_t lastEi = entries_.size() - 1;

    // Swap the removed entry with the last entry to keep entries_ compact
    if (static_cast<std::size_t>(ei) != lastEi) {
        entries_[static_cast<std::size_t>(ei)] = entries_[lastEi];
        // Fix up the moved entry's slot back-pointer
        slotToEntry_[entries_[static_cast<std::size_t>(ei)].slot] = ei;
    }
    entries_.pop_back();

    slotToEntry_[slot] = -1;
    meta_[slot]        = DELETED;
    states_[slot]      = 0;
    --size_;
    ++tombstones_;
}

std::uint32_t PosStateMap::get(std::uint64_t key) const {
    if (keys_.empty()) return 0;
    std::size_t slot = findSlot(key);
    return slot == keys_.size() ? 0 : states_[slot];
}

std::uint32_t PosStateMap::exchange(std::uint64_t key, std::uint32_t newState) {
    if (newState == 0) {
        // Removing: get old state then delete
        if (keys_.empty()) return 0;
        std::size_t slot = findSlot(key);
        if (slot == keys_.size()) return 0;
        std::uint32_t old = states_[slot];

        int ei = slotToEntry_[slot];
        std::size_t lastEi = entries_.size() - 1;
        if (static_cast<std::size_t>(ei) != lastEi) {
            entries_[static_cast<std::size_t>(ei)] = entries_[lastEi];
            slotToEntry_[entries_[static_cast<std::size_t>(ei)].slot] = ei;
        }
        entries_.pop_back();

        slotToEntry_[slot] = -1;
        meta_[slot]        = DELETED;
        states_[slot]      = 0;
        --size_;
        ++tombstones_;
        return old;
    }

    // Setting: find or insert
    ensureCapacity();
    std::size_t slot = findInsertSlot(key);
    if (meta_[slot] == FULL) {
        std::uint32_t old = states_[slot];
        states_[slot] = newState;
        entries_[static_cast<std::size_t>(slotToEntry_[slot])].state = newState;
        return old;
    }
    // New entry
    if (meta_[slot] == DELETED) --tombstones_;
    int ei = static_cast<int>(entries_.size());
    entries_.push_back(Entry{key, newState, static_cast<std::uint32_t>(slot)});
    keys_[slot]        = key;
    states_[slot]      = newState;
    meta_[slot]        = FULL;
    slotToEntry_[slot] = ei;
    ++size_;
    return 0;
}

bool PosStateMap::contains(std::uint64_t key) const {
    return get(key) != 0;
}

void PosStateMap::ensureCapacity() {
    if (keys_.empty()) {
        rehash(64);
        return;
    }
    // Tombstones occupy slots and block the EMPTY terminator that findSlot/findInsertSlot's
    // linear probe relies on to terminate. A churn-heavy run (many remove+insert cycles, e.g.
    // pistons repeatedly placing/clearing blocks) can fill the table with tombstones while
    // size_ (live entries) stays small, so the resize check must count occupied slots
    // (live + tombstones), not just live entries — otherwise probing can spin forever once
    // every slot is FULL or DELETED. Rehashing to the same capacity is enough to purge
    // tombstones when live entries don't actually warrant growing.
    std::size_t occupied = size_ + tombstones_;
    if ((occupied + 1) * 10 >= keys_.size() * 7) {
        std::size_t newCapacity = (size_ + 1) * 10 >= keys_.size() * 7 ? keys_.size() << 1 : keys_.size();
        rehash(newCapacity);
    }
}

void PosStateMap::rehash(std::size_t newCapacity) {
    auto oldKeys        = std::move(keys_);
    auto oldStates      = std::move(states_);
    auto oldMeta        = std::move(meta_);
    auto oldSlotToEntry = std::move(slotToEntry_);

    keys_.assign(newCapacity, 0);
    states_.assign(newCapacity, 0);
    meta_.assign(newCapacity, EMPTY);
    slotToEntry_.assign(newCapacity, -1);
    size_ = 0;
    tombstones_ = 0;  // fresh table has no tombstones; the loop below only inserts live entries
    // entries_ keeps its elements; we just re-insert them and update their slot fields
    for (auto& e : entries_) {
        // Re-insert via raw logic (avoids growing entries_ again)
        ensureCapacity();
        std::size_t slot = findInsertSlot(e.key);
        keys_[slot]        = e.key;
        states_[slot]      = e.state;
        meta_[slot]        = FULL;
        slotToEntry_[slot] = static_cast<int>(&e - entries_.data());
        e.slot             = static_cast<std::uint32_t>(slot);
        ++size_;
    }
    (void)oldKeys; (void)oldStates; (void)oldMeta; (void)oldSlotToEntry;
}

std::size_t PosStateMap::findSlot(std::uint64_t key) const {
    const std::size_t mask = keys_.size() - 1;
    std::size_t slot = static_cast<std::size_t>(mix64(key)) & mask;
    while (meta_[slot] != EMPTY) {
        if (meta_[slot] == FULL && keys_[slot] == key) {
            return slot;
        }
        slot = (slot + 1) & mask;
    }
    return keys_.size();
}

std::size_t PosStateMap::findInsertSlot(std::uint64_t key) const {
    const std::size_t mask = keys_.size() - 1;
    std::size_t slot = static_cast<std::size_t>(mix64(key)) & mask;
    std::size_t firstDeleted = keys_.size();
    while (meta_[slot] != EMPTY) {
        if (meta_[slot] == FULL && keys_[slot] == key) {
            return slot;
        }
        if (meta_[slot] == DELETED && firstDeleted == keys_.size()) {
            firstDeleted = slot;
        }
        slot = (slot + 1) & mask;
    }
    return firstDeleted == keys_.size() ? slot : firstDeleted;
}

void World::reset() {
    time = 0;
    blocks.clear();
    blockEvents[0].clear();
    blockEvents[1].clear();
    blockEventCacheIndex = 0;
    scheduledTicks.clear();
    nextTickOrder = 0;
    for (std::vector<MovingBlock>& bucket : movingBuckets) {
        bucket.clear();
    }
    movingPtr = 0;
}

void World::setBlock(BlockPos pos, std::uint32_t state) {
    blocks.set(packPos(pos), state);
}

void World::removeBlock(BlockPos pos) {
    blocks.remove(packPos(pos));
}

std::uint32_t World::getBlock(BlockPos pos) const {
    return blocks.get(packPos(pos));
}

std::size_t World::movingCount() const {
    return movingBuckets[0].size() + movingBuckets[1].size() + movingBuckets[2].size();
}

} // namespace mcp1122
