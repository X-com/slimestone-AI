#include "simulator.h"

#include "block_registry.h"
#include "json_stream.h"  // mcp1122::quoteJson

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mcp1122gpu {

using namespace mcp1122;

std::string Result::toJson() const {
    std::ostringstream out;
    double ticksPerSecond = elapsedNs > 0
        ? static_cast<double>(ticks) * 1000000000.0 / static_cast<double>(elapsedNs)
        : 0.0;
    out << '{'
        << "\"id\":" << id
        << ",\"ok\":" << (ok ? "true" : "false")
        << ",\"working\":" << (working ? "true" : "false")
        << ",\"ticks\":" << ticks
        << ",\"start\":" << start
        << ",\"end\":" << end
        << ",\"period\":" << period
        << ",\"shift\":{\"x\":" << shift.x
        << ",\"y\":" << shift.y
        << ",\"z\":" << shift.z << '}'
        << ",\"cycles\":" << (cycles ? "true" : "false")
        << ",\"settled\":" << (settled ? "true" : "false")
        << ",\"validCycle\":" << (validCycle ? "true" : "false")
        << ",\"finalShift\":{\"x\":" << finalShift.x
        << ",\"y\":" << finalShift.y
        << ",\"z\":" << finalShift.z << '}'
        << ",\"elapsedNs\":" << elapsedNs
        << ",\"ticksPerSecond\":" << ticksPerSecond;
    if (!ok) {
        out << ",\"errorCode\":\"" << mcp1122::quoteJson(errorCode) << '"'
            << ",\"error\":\"" << mcp1122::quoteJson(error) << '"';
    }
    out << '}';
    return out.str();
}

Result Simulator::simulate(const mcp1122::Candidate& candidate) {
    auto started = std::chrono::steady_clock::now();
    Result result;
    result.id = candidate.id;

    try {
        loadCandidate(candidate);
        trigger(candidate.trigger);

        int maxTicks = 6000;
        if (const char* env = std::getenv("MCP1122_GPU_REF_MAX_TICKS")) {
            maxTicks = std::atoi(env);
            if (maxTicks <= 0) {
                maxTicks = 6000;
            }
        }

        ShiftCycle cycle;
        ShiftCycle* found = detectShiftCycle(maxTicks, cycle);
        result.ok = true;
        if (found == nullptr) {
            result.working = false;
            result.ticks = maxTicks;
        } else {
            result.working = !samePos(found->shift, BlockPos{0, 0, 0});
            result.ticks = found->end;
            result.start = found->start;
            result.end = found->end;
            result.period = found->period;
            result.shift = found->shift;
            result.cycles = result.working;

            if (result.working) {
                // Cycle confirmed: the trigger has done its job and burns out - it stops
                // reacting to neighbor changes and stops emitting piston events, but the
                // block itself is left in place (still part of the structure's shape).
                triggerDisabled_ = true;
                watchSet_.erase(packPos(triggerPos_));

                for (int tick = found->end; tick < maxTicks; ++tick) {
                    tickWorld();
                    if (isQuiescent()) {
                        result.settled = true;
                        break;
                    }
                }

                result.validCycle = result.settled && compareFinalToInitial(candidate, result.finalShift);
            }
        }
    } catch (const std::exception& error) {
        result.ok = false;
        result.errorCode = "simulation_error";
        result.error = error.what();
    }

    auto ended = std::chrono::steady_clock::now();
    result.elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    return result;
}

// Mirrors cpp extract's isWatchedBlock/isNcPowerSource file-local helpers exactly.
namespace {

bool isWatchedBlock(int id) {
    return id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON || id == BLOCK_PISTON_HEAD
        || id == BLOCK_FENCE_GATE || id == BLOCK_LIT_REDSTONE_LAMP || id == BLOCK_REDSTONE_LAMP
        || isRailBlock(id);
}

bool isNcPowerSource(int id, std::uint32_t state) {
    if (id == BLOCK_OBSERVER && metaBit(state, 3)) return true;
    if (id == BLOCK_DETECTOR_RAIL && metaBit(state, 3)) return true;
    return false;
}

std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

std::uint64_t encodeRel(int dx, int dy, int dz) {
    return (static_cast<std::uint64_t>(dx + 1048576) & 0x1fffffULL)
         | ((static_cast<std::uint64_t>(dy + 1048576) & 0x1fffffULL) << 21)
         | ((static_cast<std::uint64_t>(dz + 1048576) & 0x1fffffULL) << 42);
}

bool anchorBeats(BlockPos pos, BlockPos cur, bool hasCur) {
    if (!hasCur) return true;
    if (pos.y != cur.y) return pos.y < cur.y;
    if (pos.z != cur.z) return pos.z < cur.z;
    return pos.x < cur.x;
}

} // namespace

void Simulator::loadCandidate(const mcp1122::Candidate& candidate) {
    world_.reset();
    watchSet_.clear();
    observerSet_.clear();
    ncPowerSet_.clear();
    triggerPos_ = BlockPos{0, 0, 0};
    triggerCharged_ = false;
    triggerEndTick_ = -1;
    triggerDisabled_ = false;

    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int minZ = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    int maxZ = std::numeric_limits<int>::min();

    auto include = [&](int x, int y, int z) {
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
        minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
    };
    for (const mcp1122::BlockEntry& block : candidate.blocks) {
        include(block.x, block.y, block.z);
    }
    include(candidate.trigger.x, candidate.trigger.y, candidate.trigger.z);

    int spanX = maxX - minX + 1;
    int spanY = maxY - minY + 1;
    int spanZ = maxZ - minZ + 1;
    int needed = std::max({spanX, spanY, spanZ}) + 2 * FixedWorld::kMargin;
    if (needed > FixedWorld::kExtent) {
        throw std::out_of_range(
            "candidate bounding box (+ margin) needs extent " + std::to_string(needed) +
            " but FixedWorld::kExtent is " + std::to_string(FixedWorld::kExtent));
    }

    world_.setOrigin(BlockPos{minX - FixedWorld::kMargin, minY - FixedWorld::kMargin, minZ - FixedWorld::kMargin});

    for (const mcp1122::BlockEntry& block : candidate.blocks) {
        if (block.state != 0) {
            BlockPos pos{block.x, block.y, block.z};
            world_.setBlock(pos, block.state);
            if (trace_ != nullptr) {
                trace_->log(world_.time, "cpp.load", &pos, static_cast<int>(block.state), 0);
            }
        }
    }

    // Build watchSet_ / observerSet_ / ncPowerSet_ from loaded blocks - world_.setBlock()
    // bypasses setBlockState(), so scan once after load (same as cpp extract's loadCandidate).
    for (const auto& e : world_.entries()) {
        int id = blockId(e.state);
        std::uint64_t key = packPos(e.pos);
        if (id == BLOCK_OBSERVER) observerSet_.insert(key);
        if (isWatchedBlock(id)) watchSet_.insert(key);
        if (isNcPowerSource(id, e.state)) ncPowerSet_.insert(key);
    }

    if (trace_ != nullptr) {
        trace_->log(world_.time, "cpp.trigger", &candidate.trigger, 0, 0);
        BlockPos anchor;
        StateKey key = stateKey(anchor);
        trace_->logHash(world_.time, "b.initkey", key.words[0], key.words[4], key.words[5]);
    }
}

void Simulator::trigger(BlockPos pos) {
    std::uint32_t state = world_.getBlock(pos);
    int id = blockId(state);
    if (!isPistonBlock(id)) {
        throw std::runtime_error("trigger must point at a piston or sticky piston");
    }

    triggerPos_ = pos;
    if (shouldPistonBeExtended(pos, state)) {
        // Already powered by the structure itself - no external kick needed.
        neighborChanged(pos, 95, pos);
    } else {
        // Cold start: emulate a brief external redstone pulse (2 game ticks, same duration
        // as an observer's own pulse) to kick the piston into motion once. checkForMove sees
        // shouldPistonBeExtended() == true for the trigger position while triggerCharged_ is
        // set (see shouldPistonBeExtended below), then tickWorld() ends the pulse and lets the
        // piston re-evaluate on real power alone.
        triggerCharged_ = true;
        triggerEndTick_ = world_.time + 2;
        checkForMove(pos, state);
    }
}

void Simulator::tickWorld() {
    if (trace_ != nullptr) {
        trace_->log(world_.time, "h.tick", nullptr, 1, 0);
    }
    ++world_.time;
    if (triggerCharged_ && world_.time >= triggerEndTick_) {
        // Pulse over - re-check the trigger piston on real power alone; it retracts on its
        // own here if nothing else in the structure is holding it extended.
        triggerCharged_ = false;
        std::uint32_t state = world_.getBlock(triggerPos_);
        int id = blockId(state);
        if (isPistonBlock(id)) {
            checkForMove(triggerPos_, state);
        }
    }
    tickScheduledUpdates();
    sendQueuedBlockEvents();
    if (trace_ != nullptr) {
        trace_->log(world_.time, "h.ent", nullptr, 1, 0);
    }
    updateEntities();
    if (trace_ != nullptr) {
        trace_->log(world_.time, "h.done", nullptr, 1, 0);
    }
}

void Simulator::tickScheduledUpdates() {
    if (world_.scheduledTicks.empty()) {
        return;
    }
    std::sort(world_.scheduledTicks.begin(), world_.scheduledTicks.end(),
        [](const ScheduledTick& a, const ScheduledTick& b) {
            if (a.time != b.time) return a.time < b.time;
            return a.order < b.order;
        });

    auto partIt = std::partition_point(world_.scheduledTicks.begin(), world_.scheduledTicks.end(),
        [this](const ScheduledTick& t) { return t.time <= world_.time; });

    pendingDue_.clear();
    pendingDue_.insert(pendingDue_.end(),
        std::make_move_iterator(world_.scheduledTicks.begin()),
        std::make_move_iterator(partIt));
    world_.scheduledTicks.erase(world_.scheduledTicks.begin(), partIt);

    for (const ScheduledTick& tick : pendingDue_) {
        std::uint32_t state = world_.getBlock(tick.pos);
        if (blockId(state) != tick.blockId) {
            continue;
        }
        if (tick.blockId == BLOCK_OBSERVER) {
            observerUpdateTick(tick.pos, state);
        } else if (tick.blockId == BLOCK_LIT_REDSTONE_LAMP) {
            lampUpdateTick(tick.pos, state);
        }
    }
}

void Simulator::lampUpdateTick(BlockPos pos, std::uint32_t state) {
    if (blockId(state) == BLOCK_LIT_REDSTONE_LAMP && !isBlockPowered(pos)) {
        setBlockState(pos, makeState(BLOCK_REDSTONE_LAMP, 0), 2);
    }
}

void Simulator::notifyObserverFront(BlockPos pos, std::uint32_t state) {
    const Facing& facing = facingByIndex(facingMeta(state));
    BlockPos front = offset(pos, opposite(facing));
    neighborChanged(front, BLOCK_OBSERVER, pos);
    notifyNeighborsExcept(front, BLOCK_OBSERVER, facing.index);
}

void Simulator::observerUpdateTick(BlockPos pos, std::uint32_t state) {
    if (trace_ != nullptr) {
        trace_->logState(world_.time, "o.tick", &pos, state);
    }
    if (metaBit(state, 3)) {
        setBlockState(pos, setMetaBit(state, 3, false), 2);
    } else {
        setBlockState(pos, setMetaBit(state, 3, true), 2);
        scheduleUpdate(pos, BLOCK_OBSERVER, 2);
    }

    notifyObserverFront(pos, state);
}

void Simulator::sendQueuedBlockEvents() {
    while (!world_.blockEvents[static_cast<std::size_t>(world_.blockEventCacheIndex)].empty()) {
        int index = world_.blockEventCacheIndex;
        world_.blockEventCacheIndex ^= 1;
        pendingBlockEvents_.clear();
        pendingBlockEvents_.swap(world_.blockEvents[static_cast<std::size_t>(index)]);
        for (const BlockEvent& event : pendingBlockEvents_) {
            fireBlockEvent(event);
        }
    }
}

bool Simulator::fireBlockEvent(const BlockEvent& event) {
    std::uint32_t state = world_.getBlock(event.pos);
    if (blockId(state) != event.blockId) {
        return false;
    }
    if (event.blockId == BLOCK_PISTON || event.blockId == BLOCK_STICKY_PISTON) {
        return pistonEventReceived(event.pos, state, event.eventId, event.eventParam);
    }
    return false;
}

void Simulator::updateEntities() {
    const bool trace = trace_ != nullptr;

    int oldest = ((world_.movingPtr + 1) % 3 + 3) % 3;
    int mid    = ((world_.movingPtr + 2) % 3 + 3) % 3;
    int newest = ((world_.movingPtr) % 3 + 3) % 3;

    if (trace) {
        int n0 = static_cast<int>(world_.movingCount());
        trace_->log(world_.time, "te.begin", nullptr, n0, n0);
        trace_->log(world_.time, "te.pcnt", nullptr, n0, n0);
        trace_->log(world_.time, "te.pending", nullptr, 0, 0);
        trace_->log(world_.time, "te.pendp", nullptr, 0, 0);
    }

    pendingDoneMoving_.clear();
    pendingDoneMoving_.swap(world_.movingBuckets[static_cast<std::size_t>(oldest)]);
    world_.movingPtr = ((world_.movingPtr + 1) % 3 + 3) % 3;

    for (const MovingBlock& moving : pendingDoneMoving_) {
        if (trace) {
            trace_->logState(world_.time, "te.p", &moving.pos, moving.pistonState, 100, 100);
        }
        settleMovingBlock(moving);
    }

    if (trace) {
        for (const MovingBlock& m : world_.movingBuckets[static_cast<std::size_t>(mid)]) {
            trace_->logState(world_.time, "te.p", &m.pos, m.pistonState, 50, 50);
        }
        for (const MovingBlock& m : world_.movingBuckets[static_cast<std::size_t>(newest)]) {
            trace_->logState(world_.time, "te.p", &m.pos, m.pistonState, 0, 0);
        }
        int mm = static_cast<int>(world_.movingCount());
        trace_->log(world_.time, "te.after", nullptr, mm, mm);
        trace_->log(world_.time, "te.end", nullptr, mm, mm);
        trace_->log(world_.time, "te.pend", nullptr, mm, mm);
    }
}

void Simulator::settleMovingBlock(const MovingBlock& moving) {
    if (trace_ != nullptr) {
        trace_->logState(world_.time, "te.done", &moving.pos, moving.pistonState);
        trace_->log(world_.time, "te.rem", &moving.pos, 1, 0);
    }
    if (blockId(world_.getBlock(moving.pos)) == BLOCK_PISTON_EXTENSION) {
        setBlockState(moving.pos, moving.pistonState, 3);
        neighborChanged(moving.pos, moving.pistonBlockId, moving.pos);
    }
    if (trace_ != nullptr) {
        trace_->log(world_.time, "te.rm", &moving.pos, 1, 0);
    }
}

void Simulator::addMovingBlock(const MovingBlock& block) {
    world_.pushMovingBlock(world_.movingPtr, block);
    if (trace_ != nullptr) {
        trace_->log(world_.time, "te.add", &block.pos, 1, 1);
        int m = static_cast<int>(world_.movingCount());
        trace_->log(world_.time, "te.added", &block.pos, m, m);
    }
}

void Simulator::scheduleUpdate(BlockPos pos, int blockIdValue, int delay) {
    if (blockIdValue == BLOCK_AIR || pos.y < 0 || pos.y >= 256) {
        return;
    }
    if (isUpdateScheduled(pos, blockIdValue)) {
        return;
    }
    ScheduledTick tick;
    tick.time = world_.time + delay;
    tick.order = world_.nextTickOrder++;
    tick.pos = pos;
    tick.blockId = blockIdValue;
    world_.pushScheduledTick(tick);
}

bool Simulator::isUpdateScheduled(BlockPos pos, int blockIdValue) const {
    for (const ScheduledTick& tick : world_.scheduledTicks) {
        if (tick.blockId == blockIdValue && samePos(tick.pos, pos)) {
            return true;
        }
    }
    return false;
}

void Simulator::addBlockEvent(BlockPos pos, int blockIdValue, int eventId, int eventParam) {
    auto& queue = world_.blockEvents[static_cast<std::size_t>(world_.blockEventCacheIndex)];
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "w.beq", &pos, blockIdValue, eventId, eventParam);
    }
    for (const BlockEvent& event : queue) {
        if (event.blockId == blockIdValue && event.eventId == eventId
                && event.eventParam == eventParam && samePos(event.pos, pos)) {
            return;
        }
    }
    world_.pushBlockEvent(world_.blockEventCacheIndex, BlockEvent{pos, blockIdValue, eventId, eventParam});
}

void Simulator::neighborChangedImpl(BlockPos pos, std::uint64_t key, int sourceBlockId, BlockPos fromPos) {
    (void)key;
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
        if (triggerDisabled_ && samePos(pos, triggerPos_)) {
            // Burned-out trigger: stops receiving updates entirely.
            return;
        }
        if (trace_ != nullptr) {
            trace_->logState(world_.time, "p.nc", &pos, state);
            trace_->logBlock(world_.time, "p.src", &fromPos, sourceBlockId);
        }
        checkForMove(pos, state);
    } else if (id == BLOCK_FENCE_GATE) {
        bool powered = isBlockPowered(pos);
        if (metaBit(state, 3) != powered) {
            std::uint32_t newState = setMetaBit(setMetaBit(state, 3, powered), 2, powered);
            setBlockState(pos, newState, 2);
        }
    } else if (id == BLOCK_LIT_REDSTONE_LAMP) {
        if (!isBlockPowered(pos)) {
            scheduleUpdate(pos, BLOCK_LIT_REDSTONE_LAMP, 4);
        }
    } else if (id == BLOCK_REDSTONE_LAMP) {
        if (isBlockPowered(pos)) {
            setBlockState(pos, makeState(BLOCK_LIT_REDSTONE_LAMP, 0), 2);
        }
    } else if (isRailBlock(id)) {
        railNeighborChanged(pos, state);
    }
}

void Simulator::neighborChanged(BlockPos pos, int sourceBlockId, BlockPos fromPos) {
    if (watchSet_.empty()) return;
    std::uint64_t key = packPos(pos);
    if (!watchSet_.count(key)) return;
    neighborChangedImpl(pos, key, sourceBlockId, fromPos);
}

void Simulator::observedNeighborChanged(BlockPos pos, int changedBlockId, BlockPos changedPos) {
    if (observerSet_.empty()) return;
    std::uint64_t key = packPos(pos);
    if (!observerSet_.contains(key)) return;
    std::uint32_t state = world_.getBlock(pos);
    if (blockId(state) != BLOCK_OBSERVER) {
        return;
    }
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "o.obs", &pos, BLOCK_OBSERVER);
        trace_->logBlock(world_.time, "o.src", &changedPos, changedBlockId);
    }
    const Facing& facing = facingByIndex(facingMeta(state));
    BlockPos watched = offset(pos, facing);
    if (samePos(watched, changedPos) && !metaBit(state, 3) && !isUpdateScheduled(pos, BLOCK_OBSERVER)) {
        scheduleUpdate(pos, BLOCK_OBSERVER, 2);
    }
}

void Simulator::notifyNeighbors(BlockPos pos, int sourceBlockId, bool updateObservers) {
    if (!watchSet_.empty()) {
        auto tryNC = [&](BlockPos n) {
            std::uint64_t k = packPos(n);
            if (watchSet_.count(k)) neighborChangedImpl(n, k, sourceBlockId, pos);
        };
        tryNC({pos.x - 1, pos.y, pos.z});
        tryNC({pos.x + 1, pos.y, pos.z});
        tryNC({pos.x, pos.y - 1, pos.z});
        tryNC({pos.x, pos.y + 1, pos.z});
        tryNC({pos.x, pos.y, pos.z - 1});
        tryNC({pos.x, pos.y, pos.z + 1});
    }
    if (updateObservers) {
        updateObservingBlocksAt(pos, sourceBlockId);
    }
}

void Simulator::notifyNeighborsExcept(BlockPos pos, int sourceBlockId, int skipFacing) {
    if (watchSet_.empty()) return;
    const int javaOrder[] = {4, 5, 0, 1, 2, 3};
    for (int index : javaOrder) {
        if (index != skipFacing) {
            BlockPos n = offset(pos, facingByIndex(index));
            std::uint64_t k = packPos(n);
            if (watchSet_.count(k)) neighborChangedImpl(n, k, sourceBlockId, pos);
        }
    }
}

void Simulator::updateObservingBlocksAt(BlockPos pos, int sourceBlockId) {
    if (observerSet_.empty()) return;
    observedNeighborChanged(BlockPos{pos.x - 1, pos.y, pos.z}, sourceBlockId, pos);
    observedNeighborChanged(BlockPos{pos.x + 1, pos.y, pos.z}, sourceBlockId, pos);
    observedNeighborChanged(BlockPos{pos.x, pos.y - 1, pos.z}, sourceBlockId, pos);
    observedNeighborChanged(BlockPos{pos.x, pos.y + 1, pos.z}, sourceBlockId, pos);
    observedNeighborChanged(BlockPos{pos.x, pos.y, pos.z - 1}, sourceBlockId, pos);
    observedNeighborChanged(BlockPos{pos.x, pos.y, pos.z + 1}, sourceBlockId, pos);
}

void Simulator::checkForMove(BlockPos pos, std::uint32_t state) {
    int id = blockId(state);
    const Facing& facing = facingByIndex(facingMeta(state));
    bool shouldExtend = shouldPistonBeExtended(pos, state);
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "p.chk", &pos, id, facing.index, shouldExtend ? 1 : 0);
    }
    if (shouldExtend && !metaBit(state, 3)) {
        PistonStructureHelper helper(world_, pos, facing, true);
        if (helper.canMove()) {
            if (trace_ != nullptr) {
                trace_->logBlock(world_.time, "p.q+", &pos, id, 0, facing.index);
            }
            addBlockEvent(pos, id, 0, facing.index);
        }
    } else if (!shouldExtend && metaBit(state, 3)) {
        if (trace_ != nullptr) {
            trace_->logBlock(world_.time, "p.q-", &pos, id, 1, facing.index);
        }
        addBlockEvent(pos, id, 1, facing.index);
    }
}

bool Simulator::shouldPistonBeExtended(BlockPos pos, std::uint32_t state) const {
    if (triggerCharged_ && samePos(pos, triggerPos_)) {
        return true;
    }
    const Facing& pistonFacing = facingByIndex(facingMeta(state));
    for (const Facing& facing : facings()) {
        if (facing.index != pistonFacing.index) {
            BlockPos probe = offset(pos, facing);
            bool powered = isSidePowered(probe, facing);
            if (trace_ != nullptr) {
                std::uint32_t probeState = world_.getBlock(probe);
                trace_->logBlock(world_.time, "p.sbe1", &probe, blockId(probeState), facing.index, powered ? 1 : 0);
            }
            if (powered) {
                return true;
            }
        }
    }
    bool downPowered = isSidePowered(pos, facings()[0]);
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "p.sbe2", &pos, blockId(world_.getBlock(pos)), 0, downPowered ? 1 : 0);
    }
    if (downPowered) {
        return true;
    }
    BlockPos up{pos.x, pos.y + 1, pos.z};
    for (const Facing& facing : facings()) {
        if (facing.index != 0) {
            BlockPos probe = offset(up, facing);
            bool powered = isSidePowered(probe, facing);
            if (trace_ != nullptr) {
                std::uint32_t probeState = world_.getBlock(probe);
                trace_->logBlock(world_.time, "p.sbe3", &probe, blockId(probeState), facing.index, powered ? 1 : 0);
            }
            if (powered) {
                return true;
            }
        }
    }
    return false;
}

bool Simulator::isTopSolid(BlockPos pos) const {
    std::uint32_t state = world_.getBlock(pos);
    int id = blockId(state);
    if (id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON) {
        return !metaBit(state, 3) || facingMeta(state) == 0;
    }
    return blockData(id).isNormalCube;
}

bool Simulator::isBlockPowered(BlockPos pos) const {
    return redstonePower(BlockPos{pos.x, pos.y - 1, pos.z}, facings()[0]) > 0
        || redstonePower(BlockPos{pos.x, pos.y + 1, pos.z}, facings()[1]) > 0
        || redstonePower(BlockPos{pos.x, pos.y, pos.z - 1}, facings()[2]) > 0
        || redstonePower(BlockPos{pos.x, pos.y, pos.z + 1}, facings()[3]) > 0
        || redstonePower(BlockPos{pos.x - 1, pos.y, pos.z}, facings()[4]) > 0
        || redstonePower(BlockPos{pos.x + 1, pos.y, pos.z}, facings()[5]) > 0;
}

bool Simulator::isSidePowered(BlockPos pos, const Facing& side) const {
    return redstonePower(pos, side) > 0;
}

int Simulator::redstonePower(BlockPos pos, const Facing& side) const {
    std::uint32_t state = world_.getBlock(pos);
    if (blockData(blockId(state)).isNormalCube) {
        if (ncPowerSet_.empty()) return 0;
        return strongPowerAll(pos);
    }
    return weakPower(state, side);
}

int Simulator::weakPower(std::uint32_t state, const Facing& side) const {
    int id = blockId(state);
    if (id == BLOCK_REDSTONE_BLOCK) {
        return 15;
    }
    if (id == BLOCK_OBSERVER && metaBit(state, 3) && facingMeta(state) == side.index) {
        return 15;
    }
    return 0;
}

int Simulator::strongPower(BlockPos pos, const Facing& side) const {
    std::uint32_t state = world_.getBlock(pos);
    int id = blockId(state);
    if (id == BLOCK_OBSERVER) {
        return weakPower(state, side);
    }
    if (id == BLOCK_DETECTOR_RAIL && metaBit(state, 3) && side.index == 1) {
        return 15;
    }
    return 0;
}

int Simulator::strongPowerAll(BlockPos pos) const {
    int out = 0;
    for (const Facing& facing : facings()) {
        BlockPos source = offset(pos, facing);
        int power = strongPower(source, facing);
        out = std::max(out, power);
    }
    return out;
}

void Simulator::railNeighborChanged(BlockPos pos, std::uint32_t state) {
    int id = blockId(state);
    int shape = railShape(id, state);
    bool unsupported = !isTopSolid(offset(pos, facings()[0])); // DOWN

    if (!unsupported) {
        switch (shape) {
            case RAIL_SHAPE_ASCENDING_EAST:
                unsupported = !isTopSolid(offset(pos, facings()[5])); // EAST
                break;
            case RAIL_SHAPE_ASCENDING_WEST:
                unsupported = !isTopSolid(offset(pos, facings()[4])); // WEST
                break;
            case RAIL_SHAPE_ASCENDING_NORTH:
                unsupported = !isTopSolid(offset(pos, facings()[2])); // NORTH
                break;
            case RAIL_SHAPE_ASCENDING_SOUTH:
                unsupported = !isTopSolid(offset(pos, facings()[3])); // SOUTH
                break;
            default:
                break;
        }
    }

    if (unsupported) {
        if (!isAirState(world_.getBlock(pos))) {
            setBlockToAir(pos);
        }
        return;
    }

    if (isPoweredRailBlock(id)) {
        updateRailPowerState(pos, state, id, shape);
    }
}

void Simulator::updateRailPowerState(BlockPos pos, std::uint32_t state, int railId, int shape) {
    bool wasPowered = metaBit(state, 3);
    bool nowPowered = isBlockPowered(pos)
        || findPoweredRailSignal(railId, pos, state, true, 0)
        || findPoweredRailSignal(railId, pos, state, false, 0);

    if (nowPowered != wasPowered) {
        setBlockState(pos, setMetaBit(state, 3, nowPowered), 3);
        notifyNeighbors(offset(pos, facings()[0]), railId, false); // DOWN
        if (isAscendingRailShape(shape)) {
            notifyNeighbors(offset(pos, facings()[1]), railId, false); // UP
        }
    }
}

bool Simulator::findPoweredRailSignal(int railId, BlockPos pos, std::uint32_t state, bool forward, int distance) const {
    if (distance >= 8) {
        return false;
    }

    int x = pos.x, y = pos.y, z = pos.z;
    bool tryOneLevelDown = true;
    int shape = railShape(railId, state);
    int searchShape = shape;

    switch (shape) {
        case RAIL_SHAPE_NORTH_SOUTH:
            if (forward) ++z; else --z;
            break;
        case RAIL_SHAPE_EAST_WEST:
            if (forward) --x; else ++x;
            break;
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
        default:
            break;
    }

    if (isSameRailWithPower(railId, BlockPos{x, y, z}, forward, distance, searchShape)) {
        return true;
    }
    return tryOneLevelDown && isSameRailWithPower(railId, BlockPos{x, y - 1, z}, forward, distance, searchShape);
}

bool Simulator::isSameRailWithPower(int railId, BlockPos pos, bool forward, int distance, int expectedShape) const {
    std::uint32_t state = world_.getBlock(pos);
    if (blockId(state) != railId) {
        return false;
    }

    int shape = railShape(railId, state);
    if (expectedShape == RAIL_SHAPE_EAST_WEST
            && (shape == RAIL_SHAPE_NORTH_SOUTH || shape == RAIL_SHAPE_ASCENDING_NORTH || shape == RAIL_SHAPE_ASCENDING_SOUTH)) {
        return false;
    }
    if (expectedShape == RAIL_SHAPE_NORTH_SOUTH
            && (shape == RAIL_SHAPE_EAST_WEST || shape == RAIL_SHAPE_ASCENDING_EAST || shape == RAIL_SHAPE_ASCENDING_WEST)) {
        return false;
    }

    if (!metaBit(state, 3)) {
        return false;
    }
    if (isBlockPowered(pos)) {
        return true;
    }
    return findPoweredRailSignal(railId, pos, state, forward, distance + 1);
}

bool Simulator::pistonEventReceived(BlockPos pos, std::uint32_t state, int id, int param) {
    if (triggerDisabled_ && samePos(pos, triggerPos_)) {
        // Burned-out trigger: produces no further output actions, even for events already
        // queued in the same tick the cycle was detected.
        return false;
    }
    const Facing& facing = facingByIndex(facingMeta(state));
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "p.ev", &pos, blockId(state), id, param);
    }
    bool shouldExtend = shouldPistonBeExtended(pos, state);
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, "p.evchk", &pos, blockId(state), facing.index, shouldExtend ? 1 : 0);
    }
    if (shouldExtend && id == 1) {
        setBlockState(pos, setMetaBit(state, 3, true), 2);
        return false;
    }
    if (!shouldExtend && id == 0) {
        return false;
    }

    bool sticky = isStickyPistonBlock(blockId(state));
    if (id == 0) {
        if (!doPistonMove(pos, facing, true, sticky)) {
            return false;
        }
        setBlockState(pos, setMetaBit(state, 3, true), 3);
    } else if (id == 1) {
        BlockPos front = offset(pos, facing);
        clearMovingAt(front);
        setBlockState(pos, setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), facing.index), 3);
        if (trace_ != nullptr) {
            trace_->log(world_.time, "te.set", &pos);
        }
        addMovingBlock(
            MovingBlock{pos, makeState(blockId(state), param), blockId(state), facing.index, false, true});

        if (sticky) {
            BlockPos pull = offset(pos, facing, 2);
            std::uint32_t pullState = world_.getBlock(pull);
            int pullId = blockId(pullState);
            bool clearedExtendingMoving = false;
            if (pullId == BLOCK_PISTON_EXTENSION && isExtendingMovingAt(pull, facing.index)) {
                clearMovingAt(pull);
                clearedExtendingMoving = true;
            }
            if (!clearedExtendingMoving && pullId != BLOCK_AIR
                    && canPush(pullState, world_, pull, opposite(facing), false, facing)
                    && (blockData(pullId).pushReaction == PushReaction::Normal
                        || pullId == BLOCK_PISTON || pullId == BLOCK_STICKY_PISTON)) {
                doPistonMove(pos, facing, false, sticky);
            }
        } else {
            setBlockToAir(front);
        }
    }
    return true;
}

bool Simulator::doPistonMove(BlockPos pos, const Facing& direction, bool extending, bool sticky) {
    if (trace_ != nullptr) {
        trace_->logBlock(world_.time, extending ? "p.mv+" : "p.mv-", &pos, blockId(world_.getBlock(pos)), direction.index, extending ? 1 : 0);
    }
    if (!extending) {
        setBlockToAir(offset(pos, direction));
    }

    PistonStructureHelper helper(world_, pos, direction, extending);
    if (!helper.canMove()) {
        return false;
    }

    const int moveCount = helper.moveCount();
    const int destroyCount = helper.destroyCount();

    std::array<std::uint32_t, 12> movedStates{};
    for (int i = 0; i < moveCount; ++i) {
        movedStates[static_cast<std::size_t>(i)] = world_.getBlock(helper.moveAt(i));
    }
    std::array<std::uint32_t, 12> destroyedStates{};
    for (int i = 0; i < destroyCount; ++i) {
        destroyedStates[static_cast<std::size_t>(i)] = world_.getBlock(helper.destroyAt(i));
    }

    // See cpp extract's doPistonMove for why this scratch array's fill/read indices don't
    // reset between the destroy and move phases - it's exact vanilla behavior that flying
    // machine redstone timing can depend on, replicated here bit-for-bit.
    std::array<std::uint32_t, 24> aiblockstate{};
    int k = moveCount + destroyCount;

    const Facing& moveFacing = extending ? direction : opposite(direction);
    for (int i = destroyCount - 1; i >= 0; --i) {
        setBlockState(helper.destroyAt(i), 0, 4);
        aiblockstate[static_cast<std::size_t>(--k)] = destroyedStates[static_cast<std::size_t>(i)];
    }
    for (int i = moveCount - 1; i >= 0; --i) {
        BlockPos source = helper.moveAt(i);
        BlockPos target = offset(source, moveFacing);
        std::uint32_t movedState = movedStates[static_cast<std::size_t>(i)];
        if (samePos(source, triggerPos_)) {
            // The trigger piston is being carried along as part of the structure moving
            // (pushed by a different piston) - its base physically relocates, so keep
            // triggerPos_ pointing at the same real block rather than a stale coordinate.
            triggerPos_ = target;
        }
        setBlockState(source, 0, 2);
        setBlockState(target, setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, 0), direction.index), 4);
        if (trace_ != nullptr) {
            trace_->log(world_.time, "te.set", &target);
        }
        addMovingBlock(
            MovingBlock{target, movedState, blockId(movedState), direction.index, extending, false});
        aiblockstate[static_cast<std::size_t>(--k)] = movedState;
    }
    if (extending) {
        int headMeta = direction.index | (sticky ? 8 : 0);
        std::uint32_t movingHead = setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), direction.index);
        BlockPos front = offset(pos, direction);
        setBlockState(front, movingHead, 4);
        if (trace_ != nullptr) {
            trace_->log(world_.time, "te.set", &front);
        }
        addMovingBlock(
            MovingBlock{front, makeState(BLOCK_PISTON_HEAD, headMeta), BLOCK_PISTON_HEAD, direction.index, true, true});
    }

    k = 0;
    for (int i = destroyCount - 1; i >= 0; --i) {
        notifyNeighbors(helper.destroyAt(i), blockId(aiblockstate[static_cast<std::size_t>(k++)]), false);
    }
    for (int i = moveCount - 1; i >= 0; --i) {
        notifyNeighbors(helper.moveAt(i), blockId(aiblockstate[static_cast<std::size_t>(k++)]), false);
    }
    if (extending) {
        notifyNeighbors(offset(pos, direction), BLOCK_PISTON_HEAD, false);
    }
    return true;
}

void Simulator::setBlockState(BlockPos pos, std::uint32_t state, int flags) {
    if (pos.y < 0 || pos.y >= 256) {
        return;
    }
    std::uint32_t oldState = world_.getBlock(pos);
    if (oldState == state) {
        return;
    }
    world_.setBlock(pos, state);

    int oldId = blockId(oldState);
    int newId = blockId(state);

    if (oldId != newId) {
        std::uint64_t setKey = packPos(pos);
        if (oldId == BLOCK_OBSERVER) observerSet_.erase(setKey);
        if (newId == BLOCK_OBSERVER) observerSet_.insert(setKey);
        bool oldW = isWatchedBlock(oldId);
        bool newW = isWatchedBlock(newId);
        if (oldW) watchSet_.erase(setKey);
        if (newW) watchSet_.insert(setKey);
    }

    bool wasPS = isNcPowerSource(oldId, oldState);
    bool isPS  = isNcPowerSource(newId, state);
    if (wasPS != isPS) {
        std::uint64_t setKey = packPos(pos);
        if (wasPS) ncPowerSet_.erase(setKey);
        if (isPS)  ncPowerSet_.insert(setKey);
    }

    if (oldId != newId && oldId == BLOCK_PISTON_EXTENSION) {
        removeMovingAt(pos);
    }
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
        if (isAscendingRailShape(oldShape)) {
            notifyNeighbors(offset(pos, facings()[1]), oldId, false); // UP
        }
        if (oldId != BLOCK_RAIL) {
            notifyNeighbors(pos, oldId, false);
            notifyNeighbors(offset(pos, facings()[0]), oldId, false); // DOWN
        }
    }
    if ((flags & 1) != 0) {
        notifyNeighbors(pos, oldId, true);
    } else if ((flags & 16) == 0) {
        updateObservingBlocksAt(pos, newId);
    }
}

void Simulator::setBlockToAir(BlockPos pos) {
    setBlockState(pos, 0, 3);
}

void Simulator::removeMovingAt(BlockPos pos) {
    for (std::vector<MovingBlock>& bucket : world_.movingBuckets) {
        auto oldSize = bucket.size();
        bucket.erase(std::remove_if(bucket.begin(), bucket.end(),
            [pos](const MovingBlock& moving) {
                return samePos(moving.pos, pos);
            }), bucket.end());
        if (trace_ != nullptr && bucket.size() != oldSize) {
            trace_->log(world_.time, "te.rem", &pos);
        }
    }
}

bool Simulator::clearMovingAt(BlockPos pos) {
    for (std::vector<MovingBlock>& bucket : world_.movingBuckets) {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (!samePos(it->pos, pos)) {
                continue;
            }
            MovingBlock moving = *it;
            bucket.erase(it);
            if (trace_ != nullptr) {
                trace_->log(world_.time, "te.rem", &pos);
            }
            if (blockId(world_.getBlock(pos)) == BLOCK_PISTON_EXTENSION) {
                setBlockState(pos, moving.pistonState, 3);
                neighborChanged(pos, moving.pistonBlockId, pos);
            }
            return true;
        }
    }
    return false;
}

bool Simulator::hasMovingAt(BlockPos pos) const {
    for (const std::vector<MovingBlock>& bucket : world_.movingBuckets) {
        for (const MovingBlock& moving : bucket) {
            if (samePos(moving.pos, pos)) {
                return true;
            }
        }
    }
    return false;
}

bool Simulator::isExtendingMovingAt(BlockPos pos, int facing) const {
    for (const std::vector<MovingBlock>& bucket : world_.movingBuckets) {
        for (const MovingBlock& moving : bucket) {
            if (samePos(moving.pos, pos) && moving.facing == facing && moving.extending) {
                return true;
            }
        }
    }
    return false;
}

Simulator::ShiftCycle* Simulator::detectShiftCycle(int maxTicks, ShiftCycle& out) {
    std::unordered_map<StateKey, SeenState, StateKeyHash> seen;
    seen.reserve(256);
    BlockPos anchor;
    StateKey key = stateKey(anchor);
    seen.emplace(key, SeenState{0, anchor});
    bool zeroShiftSeenOnce = false;
    for (int tick = 1; tick <= maxTicks; ++tick) {
        tickWorld();
        key = stateKey(anchor);
        auto it = seen.find(key);
        if (it != seen.end()) {
            BlockPos shift{anchor.x - it->second.anchor.x, anchor.y - it->second.anchor.y, anchor.z - it->second.anchor.z};
            if (!samePos(shift, BlockPos{0, 0, 0})) {
                out.start = it->second.tick;
                out.end = tick;
                out.period = tick - it->second.tick;
                out.shift = shift;
                return &out;
            }
            // Stationary repeat (shift == 0): confirm it twice before giving up early - a
            // single hit could in principle be coincidental, but seeing it again proves the
            // loop is truly stuck and will never accumulate a net shift.
            if (zeroShiftSeenOnce) {
                out.start = it->second.tick;
                out.end = tick;
                out.period = tick - it->second.tick;
                out.shift = shift;
                return &out;
            }
            zeroShiftSeenOnce = true;
        } else {
            seen.emplace(key, SeenState{tick, anchor});
        }
    }
    return nullptr;
}

bool Simulator::isQuiescent() const {
    if (!world_.scheduledTicks.empty()) return false;
    for (const auto& queue : world_.blockEvents) {
        if (!queue.empty()) return false;
    }
    for (const auto& bucket : world_.movingBuckets) {
        if (!bucket.empty()) return false;
    }
    return true;
}

// Direct final-vs-initial block set comparison: finds the translation that lines up each
// snapshot's anchor block (same ordering rule as stateKey's block-anchor), then requires every
// original block to have an exact state match at pos+shift in the final snapshot with no extras
// or omissions. This is the ground truth for "looks identical to the original, shifted N blocks" -
// independent of the period/shift the cycle detector guessed from the state-key hash.
bool Simulator::compareFinalToInitial(const mcp1122::Candidate& candidate, BlockPos& outShift) const {
    BlockPos origAnchor{0, 0, 0};
    bool haveOrig = false;
    for (const mcp1122::BlockEntry& block : candidate.blocks) {
        if (block.state == 0) continue;
        BlockPos pos{block.x, block.y, block.z};
        if (anchorBeats(pos, origAnchor, haveOrig)) {
            origAnchor = pos;
            haveOrig = true;
        }
    }

    const auto& finalEntries = world_.entries();
    BlockPos finalAnchor{0, 0, 0};
    bool haveFinal = false;
    for (const auto& entry : finalEntries) {
        if (anchorBeats(entry.pos, finalAnchor, haveFinal)) {
            finalAnchor = entry.pos;
            haveFinal = true;
        }
    }

    if (!haveOrig || !haveFinal) {
        return false;
    }

    BlockPos shift{finalAnchor.x - origAnchor.x, finalAnchor.y - origAnchor.y, finalAnchor.z - origAnchor.z};

    std::size_t origCount = 0;
    for (const mcp1122::BlockEntry& block : candidate.blocks) {
        if (block.state == 0) continue;
        ++origCount;
        BlockPos shiftedPos{block.x + shift.x, block.y + shift.y, block.z + shift.z};
        // FixedWorld::getBlock() throws on an out-of-range position (unlike cpp extract's
        // unbounded block map, which just returns 0/not-present) - a shift landing outside the
        // fixed cube is "no match" here, same as cpp extract reading an unset position, not a
        // whole-simulation error.
        if (!world_.inRange(shiftedPos) || world_.getBlock(shiftedPos) != block.state) {
            return false;
        }
    }
    if (origCount != finalEntries.size()) {
        return false;
    }

    outShift = shift;
    return true;
}

StateKey Simulator::stateKey(BlockPos& anchor) const {
    const std::vector<FixedWorld::Entry>& blockEntries = world_.entries();

    BlockPos blockAnchor{0, 0, 0};
    bool haveBlockAnchor = false;
    for (const auto& e : blockEntries) {
        if (anchorBeats(e.pos, blockAnchor, haveBlockAnchor)) {
            blockAnchor = e.pos;
            haveBlockAnchor = true;
        }
    }

    anchor = blockAnchor;
    bool haveAnchor = haveBlockAnchor;
    auto consider = [&](BlockPos pos) {
        if (anchorBeats(pos, anchor, haveAnchor)) {
            anchor = pos;
            haveAnchor = true;
        }
    };
    for (const auto& queue : world_.blockEvents) {
        for (const BlockEvent& event : queue) consider(event.pos);
    }
    for (const ScheduledTick& tick : world_.scheduledTicks) consider(tick.pos);
    for (const auto& bucket : world_.movingBuckets) {
        for (const MovingBlock& moving : bucket) consider(moving.pos);
    }
    if (!haveAnchor) {
        anchor = BlockPos{0, 0, 0};
    }

    std::uint64_t blockA = 0x9e3779b97f4a7c15ULL;
    std::uint64_t blockB = 0xc2b2ae3d27d4eb4fULL;
    if (haveAnchor) {
        for (const auto& e : blockEntries) {
            std::uint64_t enc = encodeRel(e.pos.x - anchor.x, e.pos.y - anchor.y, e.pos.z - anchor.z)
                               ^ (static_cast<std::uint64_t>(e.state) << 17);
            std::uint64_t m = mix64(enc);
            blockA += m;
            blockB ^= (m << 27) | (m >> 37);
        }
    }

    std::uint64_t eventA = 0x165667b19e3779f9ULL;
    std::uint64_t eventB = 0x85ebca77c2b2ae63ULL;
    std::uint64_t eventCount = 0;
    for (const auto& queue : world_.blockEvents) {
        for (const BlockEvent& event : queue) {
            std::uint64_t encoded = encodeRel(event.pos.x - anchor.x, event.pos.y - anchor.y, event.pos.z - anchor.z)
                ^ (static_cast<std::uint64_t>(event.blockId & 0xfff) << 13)
                ^ (static_cast<std::uint64_t>(event.eventId & 0xff) << 25)
                ^ (static_cast<std::uint64_t>(event.eventParam & 0xff) << 33);
            std::uint64_t mixed = mix64(encoded);
            eventA += mixed;
            eventB ^= (mixed << 31) | (mixed >> 33);
            ++eventCount;
        }
    }

    std::uint64_t tickA = 0xd6e8feb86659fd93ULL;
    std::uint64_t tickB = 0xa5a3564e27f88621ULL;
    for (const ScheduledTick& tick : world_.scheduledTicks) {
        std::uint64_t encoded = encodeRel(tick.pos.x - anchor.x, tick.pos.y - anchor.y, tick.pos.z - anchor.z)
            ^ (static_cast<std::uint64_t>(tick.blockId & 0xfff) << 11)
            ^ (static_cast<std::uint64_t>(tick.time - world_.time) << 32);
        std::uint64_t mixed = mix64(encoded);
        tickA += mixed;
        tickB ^= (mixed << 43) | (mixed >> 21);
    }

    std::uint64_t movingA = 0x94d049bb133111ebULL;
    std::uint64_t movingB = 0xbf58476d1ce4e5b9ULL;
    std::uint64_t movingCount = 0;
    for (std::size_t phase = 0; phase < world_.movingBuckets.size(); ++phase) {
        int bucketIndex = ((world_.movingPtr - static_cast<int>(phase)) % 3 + 3) % 3;
        const auto& bucket = world_.movingBuckets[static_cast<std::size_t>(bucketIndex)];
        for (const MovingBlock& moving : bucket) {
            std::uint64_t encoded = encodeRel(moving.pos.x - anchor.x, moving.pos.y - anchor.y, moving.pos.z - anchor.z)
                ^ (static_cast<std::uint64_t>(moving.pistonState) << 9)
                ^ (static_cast<std::uint64_t>(phase) << 29)
                ^ (static_cast<std::uint64_t>(moving.facing) << 33)
                ^ (moving.extending ? 0x100000001b3ULL : 0ULL)
                ^ (moving.shouldHeadBeRendered ? 0x9e3779b97f4a7c15ULL : 0ULL);
            std::uint64_t mixed = mix64(encoded);
            movingA += mixed;
            movingB ^= (mixed << 19) | (mixed >> 45);
            ++movingCount;
        }
    }

    StateKey key;
    key.words[0]  = static_cast<std::uint64_t>(blockEntries.size());
    key.words[1]  = eventCount;
    key.words[2]  = static_cast<std::uint64_t>(world_.scheduledTicks.size());
    key.words[3]  = movingCount;
    key.words[4]  = blockA;
    key.words[5]  = blockB;
    key.words[6]  = eventA;
    key.words[7]  = eventB;
    key.words[8]  = tickA;
    key.words[9]  = tickB;
    key.words[10] = movingA;
    key.words[11] = movingB;
    return key;
}

bool Simulator::samePos(BlockPos a, BlockPos b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

} // namespace mcp1122gpu
