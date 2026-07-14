#include "simulator.h"

#include "block_registry.h"
#include "facing.h"
#include "json_stream.h"
#include "piston.h"
#include "profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <unordered_map>

namespace mcp1122 {

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
        << ",\"elapsedNs\":" << elapsedNs
        << ",\"ticksPerSecond\":" << ticksPerSecond;
    if (!ok) {
        out << ",\"errorCode\":\"" << quoteJson(errorCode) << '"'
            << ",\"error\":\"" << quoteJson(error) << '"';
    }
    out << '}';
    return out.str();
}

Result Simulator::simulate(const Candidate& candidate) {
    auto started = std::chrono::steady_clock::now();
    Result result;
    result.id = candidate.id;

    try {
        loadCandidate(candidate);
        trigger(candidate.trigger);

        int maxTicks = 6000;
        if (const char* env = std::getenv("MCP1122_CPP_MAX_TICKS")) {
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

void Simulator::trigger(BlockPos pos) {
    std::uint32_t state = world_.getBlock(pos);
    if (blockId(state) == BLOCK_OBSERVER) {
        scheduleUpdate(pos, BLOCK_OBSERVER, 2);
    } else {
        neighborChanged(pos, 95, pos);
    }
}

void Simulator::tickWorld() {
    PROF_SCOPE("tickWorld");
    if (trace_ != nullptr) {
        trace_->log(world_, "h.tick", nullptr, 1, 0);
    }
    ++world_.time;
    tickScheduledUpdates();
    sendQueuedBlockEvents();
    if (trace_ != nullptr) {
        trace_->log(world_, "h.ent", nullptr, 1, 0);
    }
    updateEntities();
    if (trace_ != nullptr) {
        trace_->log(world_, "h.done", nullptr, 1, 0);
    }
}

void Simulator::tickScheduledUpdates() {
    PROF_SCOPE("tickScheduledUpdates");
    // order is unique (monotonically increasing) so std::sort equals stable_sort here
    std::sort(world_.scheduledTicks.begin(), world_.scheduledTicks.end(),
        [](const World::ScheduledTick& a, const World::ScheduledTick& b) {
            if (a.time != b.time) return a.time < b.time;
            return a.order < b.order;
        });

    // Due ticks are at the front after sorting; find the boundary in O(log n)
    auto partIt = std::partition_point(world_.scheduledTicks.begin(), world_.scheduledTicks.end(),
        [this](const World::ScheduledTick& t) { return t.time <= world_.time; });

    // Move due ticks into reused member buffer (no allocation after warm-up)
    pendingDue_.clear();
    pendingDue_.insert(pendingDue_.end(),
        std::make_move_iterator(world_.scheduledTicks.begin()),
        std::make_move_iterator(partIt));
    world_.scheduledTicks.erase(world_.scheduledTicks.begin(), partIt);

    for (const World::ScheduledTick& tick : pendingDue_) {
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
    // Fired by the 4-tick scheduled update queued in neighborChangedImpl. A lit lamp still unpowered
    // turns off (124 -> 123). Without this the lamp stayed lit forever and the observers driving the
    // machine never pulsed. Mirrors mcp1122 BlockRedstoneLight.updateTick.
    if (blockId(state) == BLOCK_LIT_REDSTONE_LAMP && !isBlockPowered(pos)) {
        setBlockState(pos, makeState(BLOCK_REDSTONE_LAMP, 0), 2);
    }
}

// Mirrors BlockObserver.updateNeighborsInFront: pokes the block the observer is facing away
// from, both directly and (except back toward the observer itself) its other neighbors.
// Shared by the tick pulse and the breakBlock hook below, which only cares about facing —
// not the powered bit — so both can pass whatever state they have on hand.
void Simulator::notifyObserverFront(BlockPos pos, std::uint32_t state) {
    const Facing& facing = facingByIndex(facingMeta(state));
    BlockPos front = offset(pos, opposite(facing));
    neighborChanged(front, BLOCK_OBSERVER, pos);
    notifyNeighborsExcept(front, BLOCK_OBSERVER, facing.index);
}

void Simulator::observerUpdateTick(BlockPos pos, std::uint32_t state) {
    if (trace_ != nullptr) {
        trace_->logState(world_, "o.tick", &pos, state);
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
    PROF_SCOPE("sendQueuedBlockEvents");
    while (!world_.blockEvents[static_cast<std::size_t>(world_.blockEventCacheIndex)].empty()) {
        int index = world_.blockEventCacheIndex;
        world_.blockEventCacheIndex ^= 1;
        std::vector<World::BlockEvent> events;
        events.swap(world_.blockEvents[static_cast<std::size_t>(index)]);
        for (const World::BlockEvent& event : events) {
            fireBlockEvent(event);
        }
    }
}

bool Simulator::fireBlockEvent(const World::BlockEvent& event) {
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
    PROF_SCOPE("updateEntities");
    // The whole tile-entity bookkeeping trace shims mcp1122's five-list infra. mcp1122 iterates a
    // single insertion-ordered list; that order equals iterating our buckets oldest->newest. All of
    // it is trace-only and skipped entirely when tracing is off.
    const bool trace = trace_ != nullptr;

    // Capture phase buckets before advancing movingPtr:
    //   oldest = movingPtr+1 -> completing this tick (phase 2, progress 100,100)
    //   mid    = movingPtr+2 -> phase 1 (progress 50,50)
    //   newest = movingPtr   -> added this tick (phase 0, progress 0,0)
    int oldest = ((world_.movingPtr + 1) % 3 + 3) % 3;
    int mid    = ((world_.movingPtr + 2) % 3 + 3) % 3;
    int newest = ((world_.movingPtr) % 3 + 3) % 3;

    if (trace) {
        int n0 = static_cast<int>(world_.movingCount()); // all live arms, incl. those completing
        trace_->log(world_, "te.begin", nullptr, n0, n0);
        trace_->log(world_, "te.pcnt", nullptr, n0, n0);   // all moving blocks are pistons
        trace_->log(world_, "te.pending", nullptr, 0, 0);  // no deferred queue
        trace_->log(world_, "te.pendp", nullptr, 0, 0);
    }

    std::vector<World::MovingBlock> done;
    done.swap(world_.movingBuckets[static_cast<std::size_t>(oldest)]);
    world_.movingPtr = ((world_.movingPtr + 1) % 3 + 3) % 3;

    for (const World::MovingBlock& moving : done) {
        if (trace) {
            trace_->logState(world_, "te.p", &moving.pos, moving.pistonState, 100, 100);
        }
        settleMovingBlock(moving);
    }

    if (trace) {
        // Non-completing arms emit te.p only, in insertion order after the completions:
        // phase 1 (mid, 50,50) then phase 0 (newest, 0,0). Pure trace, no simulation effect.
        for (const World::MovingBlock& m : world_.movingBuckets[static_cast<std::size_t>(mid)]) {
            trace_->logState(world_, "te.p", &m.pos, m.pistonState, 50, 50);
        }
        for (const World::MovingBlock& m : world_.movingBuckets[static_cast<std::size_t>(newest)]) {
            trace_->logState(world_, "te.p", &m.pos, m.pistonState, 0, 0);
        }
        int mm = static_cast<int>(world_.movingCount()); // after completions removed
        trace_->log(world_, "te.after", nullptr, mm, mm);
        trace_->log(world_, "te.end", nullptr, mm, mm);
        trace_->log(world_, "te.pend", nullptr, mm, mm);
    }
}

void Simulator::settleMovingBlock(const World::MovingBlock& moving) {
    // Per completing arm mcp1122 emits: te.p(100,100, logged by caller) -> te.done -> te.rem(1,0) ->
    // [neighbor cascade from setBlockState] -> te.rm(1,0). The 1,0 flag pair is (processingLoadedTiles=1, 0):
    // completion runs inside the tile-tick loop. Trace-only.
    if (trace_ != nullptr) {
        trace_->logState(world_, "te.done", &moving.pos, moving.pistonState);
        trace_->log(world_, "te.rem", &moving.pos, 1, 0);
    }
    if (blockId(world_.getBlock(moving.pos)) == BLOCK_PISTON_EXTENSION) {
        setBlockState(moving.pos, moving.pistonState, 3);
        neighborChanged(moving.pos, moving.pistonBlockId, moving.pos);
    }
    if (trace_ != nullptr) {
        trace_->log(world_, "te.rm", &moving.pos, 1, 0);
    }
}

void Simulator::addMovingBlock(const World::MovingBlock& block) {
    world_.movingBuckets[static_cast<std::size_t>(world_.movingPtr)].push_back(block);
    // mcp1122 logs te.add (added-ok, is-tickable => constant 1,1) then te.added (running total)
    // from World.addTileEntity, right after te.set. Trace-only.
    if (trace_ != nullptr) {
        trace_->log(world_, "te.add", &block.pos, 1, 1);
        int m = static_cast<int>(world_.movingCount());
        trace_->log(world_, "te.added", &block.pos, m, m);
    }
}

void Simulator::scheduleUpdate(BlockPos pos, int blockIdValue, int delay) {
    if (blockIdValue == BLOCK_AIR || pos.y < 0 || pos.y >= 256) {
        return;
    }
    if (isUpdateScheduled(pos, blockIdValue)) {
        return;
    }
    World::ScheduledTick tick;
    tick.time = world_.time + delay;
    tick.order = world_.nextTickOrder++;
    tick.pos = pos;
    tick.blockId = blockIdValue;
    world_.scheduledTicks.push_back(tick);
}

bool Simulator::isUpdateScheduled(BlockPos pos, int blockIdValue) const {
    PROF_SCOPE("isUpdateScheduled");
    for (const World::ScheduledTick& tick : world_.scheduledTicks) {
        if (tick.blockId == blockIdValue && samePos(tick.pos, pos)) {
            return true;
        }
    }
    return false;
}

void Simulator::addBlockEvent(BlockPos pos, int blockIdValue, int eventId, int eventParam) {
    std::vector<World::BlockEvent>& queue = world_.blockEvents[static_cast<std::size_t>(world_.blockEventCacheIndex)];
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "w.beq", &pos, blockIdValue, eventId, eventParam);
    }
    for (const World::BlockEvent& event : queue) {
        if (event.blockId == blockIdValue && event.eventId == eventId
                && event.eventParam == eventParam && samePos(event.pos, pos)) {
            return;
        }
    }
    queue.push_back(World::BlockEvent{pos, blockIdValue, eventId, eventParam});
}

// Core logic for neighborChanged, separated so notifyNeighbors can call it directly
// with a pre-computed packed key (avoiding double packPos + watchSet_ lookup).
void Simulator::neighborChangedImpl(BlockPos pos, std::uint64_t key, int sourceBlockId, BlockPos fromPos) {
    std::uint32_t state = world_.blocks.get(key);
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
        if (trace_ != nullptr) {
            trace_->logState(world_, "p.nc", &pos, state);
            trace_->logBlock(world_, "p.src", &fromPos, sourceBlockId);
        }
        checkForMove(pos, state);
    } else if (id == BLOCK_FENCE_GATE) {
        bool powered = isBlockPowered(pos);
        if (metaBit(state, 3) != powered) {
            std::uint32_t newState = setMetaBit(setMetaBit(state, 3, powered), 2, powered);
            setBlockState(pos, newState, 2);
        }
    } else if (id == BLOCK_LIT_REDSTONE_LAMP) {
        // Lit lamp losing power turns off after a 4-tick scheduled delay (mcp1122
        // BlockRedstoneLight.neighborChanged). The actual turn-off happens in lampUpdateTick.
        if (!isBlockPowered(pos)) {
            scheduleUpdate(pos, BLOCK_LIT_REDSTONE_LAMP, 4);
        }
    } else if (id == BLOCK_REDSTONE_LAMP) {
        // Unlit lamp gaining power turns on immediately.
        if (isBlockPowered(pos)) {
            setBlockState(pos, makeState(BLOCK_LIT_REDSTONE_LAMP, 0), 2);
        }
    } else if (isRailBlock(id)) {
        railNeighborChanged(pos, state);
    }
}

void Simulator::neighborChanged(BlockPos pos, int sourceBlockId, BlockPos fromPos) {
    PROF_SCOPE("neighborChanged");
    // Fast exit: skip the main 65k-slot hash table lookup when this position
    // holds nothing that reacts to neighbor changes (the common case is air).
    if (watchSet_.empty()) return;
    std::uint64_t key = packPos(pos);
    if (!watchSet_.count(key)) return;
    neighborChangedImpl(pos, key, sourceBlockId, fromPos);
}

void Simulator::observedNeighborChanged(BlockPos pos, int changedBlockId, BlockPos changedPos) {
    PROF_SCOPE("observedNeighborChanged");
    // Fast exit: skip the hash table lookup when nothing at this position is an observer.
    if (observerSet_.empty()) return;
    std::uint64_t key = packPos(pos);
    if (observerSet_.find(key) == observerSet_.end()) return;
    std::uint32_t state = world_.blocks.get(key);
    if (blockId(state) != BLOCK_OBSERVER) {
        return;
    }
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "o.obs", &pos, BLOCK_OBSERVER);
        trace_->logBlock(world_, "o.src", &changedPos, changedBlockId);
    }
    const Facing& facing = facingByIndex(facingMeta(state));
    BlockPos watched = offset(pos, facing);
    if (samePos(watched, changedPos) && !metaBit(state, 3) && !isUpdateScheduled(pos, BLOCK_OBSERVER)) {
        scheduleUpdate(pos, BLOCK_OBSERVER, 2);
    }
}

void Simulator::notifyNeighbors(BlockPos pos, int sourceBlockId, bool updateObservers) {
    // Inline the watchSet_ check so we skip the function-call overhead entirely for
    // the ~98 % of neighbor positions that hold air, glass, or other inert blocks.
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
    PROF_SCOPE("checkForMove");
    int id = blockId(state);
    const Facing& facing = facingByIndex(facingMeta(state));
    bool shouldExtend = shouldPistonBeExtended(pos, state);
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "p.chk", &pos, id, facing.index, shouldExtend ? 1 : 0);
    }
    if (shouldExtend && !metaBit(state, 3)) {
        PistonStructureHelper helper(world_, pos, facing, true);
        if (helper.canMove()) {
            if (trace_ != nullptr) {
                trace_->logBlock(world_, "p.q+", &pos, id, 0, facing.index);
            }
            addBlockEvent(pos, id, 0, facing.index);
        }
    } else if (!shouldExtend && metaBit(state, 3)) {
        if (trace_ != nullptr) {
            trace_->logBlock(world_, "p.q-", &pos, id, 1, facing.index);
        }
        addBlockEvent(pos, id, 1, facing.index);
    }
}

bool Simulator::shouldPistonBeExtended(BlockPos pos, std::uint32_t state) const {
    PROF_SCOPE("shouldPistonBeExtended");
    const Facing& pistonFacing = facingByIndex(facingMeta(state));
    for (const Facing& facing : facings()) {
        if (facing.index != pistonFacing.index) {
            BlockPos probe = offset(pos, facing);
            bool powered = isSidePowered(probe, facing);
            // Log before the early return so the deciding powered side is recorded (mcp1122
            // logs then returns). Guarded so the getBlock probe costs nothing when tracing is off.
            if (trace_ != nullptr) {
                std::uint32_t probeState = world_.getBlock(probe);
                trace_->logBlock(world_, "p.sbe1", &probe, blockId(probeState), facing.index, powered ? 1 : 0);
            }
            if (powered) {
                return true;
            }
        }
    }
    bool downPowered = isSidePowered(pos, facings()[0]);
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "p.sbe2", &pos, blockId(world_.getBlock(pos)), 0, downPowered ? 1 : 0);
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
                trace_->logBlock(world_, "p.sbe3", &probe, blockId(probeState), facing.index, powered ? 1 : 0);
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
        // BlockPistonBase.isTopSolid: a retracted piston (or one facing down) still counts
        // as a solid top; an extended piston facing any other way does not.
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
        // strongPowerAll only returns non-zero when the cube is adjacent to an
        // observer or detector rail.  Skip it when no power sources exist.
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

// Mirrors BlockRailBase.neighborChanged: breaks the rail if it's no longer supported
// (nothing solid below it, or below the raised end for an ascending slope), otherwise
// hands off to the powered-rail redstone relay for golden/activator rails.
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

// Mirrors BlockRailPowered.updateState: recomputes the rail's own powered bit from direct
// redstone power plus the up-to-8-hop powered-rail relay in both directions, and — only
// when the bit actually flips — explicitly pokes the block below (and above, for a slope)
// so a piston or lamp sitting against the rail sees the change even though the rail itself
// never provides power to neighbors.
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

// Mirrors BlockRailPowered.findPoweredRailSignal: walks along the rail's own axis (bending
// up/down one Y level across an ascending segment) looking for a same-type powered rail
// carrying signal, up to 8 hops.
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

// Mirrors BlockRailPowered.isSameRailWithPower: the candidate must be the exact same rail
// block (golden-to-golden, activator-to-activator, never cross-type), must not be a
// perpendicular crossing relative to the axis the signal is traveling along, and must
// itself be powered (directly, or by recursing one hop further).
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
    const Facing& facing = facingByIndex(facingMeta(state));
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "p.ev", &pos, blockId(state), id, param);
    }
    bool shouldExtend = shouldPistonBeExtended(pos, state);
    if (trace_ != nullptr) {
        trace_->logBlock(world_, "p.evchk", &pos, blockId(state), facing.index, shouldExtend ? 1 : 0);
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
            trace_->log(world_, "te.set", &pos);
        }
        addMovingBlock(
            World::MovingBlock{pos, makeState(blockId(state), param), blockId(state), facing.index, false, true});

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
    PROF_SCOPE("doPistonMove");
    if (trace_ != nullptr) {
        trace_->logBlock(world_, extending ? "p.mv+" : "p.mv-", &pos, blockId(world_.getBlock(pos)), direction.index, extending ? 1 : 0);
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

    // mcp1122's doMove() fills one shared scratch array (aiblockstate) with the destroyed
    // and moved blocks' *original* states via a continuously-decrementing index, then
    // reads it back during notification via a continuously-incrementing index that is
    // never reset between the destroy and move phases. That mismatch cross-wires which
    // original state gets reported as the neighborChanged() source for each notified
    // position (e.g. with no destroyed blocks the first and last moved position swap
    // reported states). It looks like an accident but it's exact vanilla behavior, and
    // flying-machine redstone timing can depend on it, so it must be replicated exactly
    // rather than each position reporting its own state.
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
        setBlockState(source, 0, 2);
        setBlockState(target, setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, 0), direction.index), 4);
        if (trace_ != nullptr) {
            trace_->log(world_, "te.set", &target);
        }
        addMovingBlock(
            World::MovingBlock{target, movedState, blockId(movedState), direction.index, extending, false});
        aiblockstate[static_cast<std::size_t>(--k)] = movedState;
    }
    if (extending) {
        int headMeta = direction.index | (sticky ? 8 : 0);
        std::uint32_t movingHead = setFacingMeta(makeState(BLOCK_PISTON_EXTENSION, sticky ? 8 : 0), direction.index);
        BlockPos front = offset(pos, direction);
        setBlockState(front, movingHead, 4);
        if (trace_ != nullptr) {
            trace_->log(world_, "te.set", &front);
        }
        addMovingBlock(
            World::MovingBlock{front, makeState(BLOCK_PISTON_HEAD, headMeta), BLOCK_PISTON_HEAD, direction.index, true, true});
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

// Returns true for blocks that can provide *strong* power through a normal cube.
// Only BLOCK_OBSERVER (when active) and BLOCK_DETECTOR_RAIL (when activated) qualify;
// BLOCK_REDSTONE_BLOCK provides only weak power and is handled directly in weakPower().
static bool isNcPowerSource(int id, std::uint32_t state) {
    if (id == BLOCK_OBSERVER && metaBit(state, 3)) return true;
    if (id == BLOCK_DETECTOR_RAIL && metaBit(state, 3)) return true;
    return false;
}

// Positions that react to neighborChanged and so must stay in watchSet_. Shared by
// setBlockState() (incremental add/remove on a type change) and loadCandidate() (initial
// bulk scan) so the two can't drift out of sync.
static bool isWatchedBlock(int id) {
    return id == BLOCK_PISTON || id == BLOCK_STICKY_PISTON || id == BLOCK_PISTON_HEAD
        || id == BLOCK_FENCE_GATE || id == BLOCK_LIT_REDSTONE_LAMP || id == BLOCK_REDSTONE_LAMP
        || isRailBlock(id);
}

// mix64 and encodeRel are shared between stateKey() and the inline cache update below.
static std::uint64_t bhcMix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
static std::uint64_t bhcEncodeRel(int dx, int dy, int dz) {
    return (static_cast<std::uint64_t>(dx + 1048576) & 0x1fffffULL)
         | ((static_cast<std::uint64_t>(dy + 1048576) & 0x1fffffULL) << 21)
         | ((static_cast<std::uint64_t>(dz + 1048576) & 0x1fffffULL) << 42);
}

// Returns true if pos has strictly smaller anchor-ordering than cur (y < cur.y, else z, else x).
static bool beatsAnchor(BlockPos pos, BlockPos cur) {
    if (pos.y != cur.y) return pos.y < cur.y;
    if (pos.z != cur.z) return pos.z < cur.z;
    return pos.x < cur.x;
}

void Simulator::setBlockState(BlockPos pos, std::uint32_t state, int flags) {
    PROF_SCOPE("setBlockState");
    if (pos.y < 0 || pos.y >= 256) {
        return;
    }
    // exchange() reads old state and writes new in a single hash probe
    std::uint32_t oldState = world_.blocks.exchange(packPos(pos), state);
    if (oldState == state) {
        return;
    }

    // --- Incremental block-hash cache maintenance ---
    if (!bhc_.dirty) {
        bool removing = (state == 0);
        bool adding   = (oldState == 0);
        bool isAnchor = bhc_.hasAnchor && samePos(pos, bhc_.anchor);
        bool newBeats = !bhc_.hasAnchor || beatsAnchor(pos, bhc_.anchor);

        if (removing && isAnchor) {
            // Anchor block removed; need O(n) scan for new anchor.
            bhc_.dirty = true;
        } else if (!removing && newBeats) {
            // New block becomes the anchor; all relative positions change.
            bhc_.dirty = true;
        } else {
            // Anchor unchanged — update accumulators incrementally.
            if (oldState != 0) {
                std::uint64_t enc = bhcEncodeRel(pos.x - bhc_.anchor.x,
                                                  pos.y - bhc_.anchor.y,
                                                  pos.z - bhc_.anchor.z)
                                  ^ (static_cast<std::uint64_t>(oldState) << 17);
                std::uint64_t m = bhcMix64(enc);
                bhc_.blockA -= m;
                bhc_.blockB ^= (m << 27) | (m >> 37);
            }
            if (state != 0) {
                std::uint64_t enc = bhcEncodeRel(pos.x - bhc_.anchor.x,
                                                  pos.y - bhc_.anchor.y,
                                                  pos.z - bhc_.anchor.z)
                                  ^ (static_cast<std::uint64_t>(state) << 17);
                std::uint64_t m = bhcMix64(enc);
                bhc_.blockA += m;
                bhc_.blockB ^= (m << 27) | (m >> 37);
            }
            // If we removed the anchor (but anchor stayed because block is still
            // present via another entry), nothing extra to do — the above
            // subtracted the old contribution correctly.
            (void)adding; (void)isAnchor;
        }
    }
    // ------------------------------------------------

    int oldId = blockId(oldState);
    int newId = blockId(state);

    // Maintain watchSet_ / observerSet_ when the block type changes.
    if (oldId != newId) {
        std::uint64_t setKey = packPos(pos);
        if (oldId == BLOCK_OBSERVER) observerSet_.erase(setKey);
        if (newId == BLOCK_OBSERVER) observerSet_.insert(setKey);
        bool oldW = isWatchedBlock(oldId);
        bool newW = isWatchedBlock(newId);
        if (oldW) watchSet_.erase(setKey);
        if (newW) watchSet_.insert(setKey);
    }

    // Maintain ncPowerSet_ — also for same-type state changes (observer pulse, rail power).
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
        // Mirrors BlockObserver.breakBlock: an observer removed (pushed by a piston,
        // overwritten, ...) mid-pulse — i.e. it was powered and still had its follow-up tick
        // scheduled — fires one last updateNeighborsInFront so its front neighbor still sees
        // the pulse complete, instead of the cycle silently vanishing.
        if (metaBit(oldState, 3) && isUpdateScheduled(pos, BLOCK_OBSERVER)) {
            notifyObserverFront(pos, oldState);
        }
    }
    if (oldId != newId && (newId == BLOCK_PISTON || newId == BLOCK_STICKY_PISTON) && !hasMovingAt(pos)) {
        checkForMove(pos, state);
    }
    if (oldId != newId && isRailBlock(oldId)) {
        // Mirrors BlockRailBase.breakBlock: fires whenever a rail is replaced by anything
        // else (broken by the unsupported-check, pushed away, overwritten, ...), not just
        // along the support-check path above. Detector/golden/activator rails are all
        // "isPowered" in vanilla (only plain rail isn't), so they additionally poke their
        // own position and the block below.
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
    for (std::vector<World::MovingBlock>& bucket : world_.movingBuckets) {
        auto oldSize = bucket.size();
        bucket.erase(std::remove_if(bucket.begin(), bucket.end(),
            [pos](const World::MovingBlock& moving) {
                return samePos(moving.pos, pos);
            }), bucket.end());
        if (trace_ != nullptr && bucket.size() != oldSize) {
            trace_->log(world_, "te.rem", &pos);
        }
    }
}

bool Simulator::clearMovingAt(BlockPos pos) {
    for (std::vector<World::MovingBlock>& bucket : world_.movingBuckets) {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (!samePos(it->pos, pos)) {
                continue;
            }
            World::MovingBlock moving = *it;
            bucket.erase(it);
            if (trace_ != nullptr) {
                trace_->log(world_, "te.rem", &pos);
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
    PROF_SCOPE("hasMovingAt");
    for (const std::vector<World::MovingBlock>& bucket : world_.movingBuckets) {
        for (const World::MovingBlock& moving : bucket) {
            if (samePos(moving.pos, pos)) {
                return true;
            }
        }
    }
    return false;
}

bool Simulator::isExtendingMovingAt(BlockPos pos, int facing) const {
    for (const std::vector<World::MovingBlock>& bucket : world_.movingBuckets) {
        for (const World::MovingBlock& moving : bucket) {
            if (samePos(moving.pos, pos) && moving.facing == facing && moving.extending) {
                return true;
            }
        }
    }
    return false;
}

Simulator::ShiftCycle* Simulator::detectShiftCycle(int maxTicks, ShiftCycle& out) {
    std::unordered_map<StateKey, SeenState, StateKeyHash> seen;
    seen.reserve(256);  // avoid early rehashes; most machines cycle within 256 ticks
    BlockPos anchor;
    StateKey key = stateKey(anchor);
    seen.emplace(key, SeenState{0, anchor});
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
        } else {
            seen.emplace(key, SeenState{tick, anchor});
        }
    }
    return nullptr;
}

Simulator::StateKey Simulator::stateKey(BlockPos& anchor) const {
    PROF_SCOPE("stateKey");

    const auto& blockEntries = world_.blocks.entries();

    // --- Determine block-anchor and block hash ---
    // If the incremental cache is clean we have blockA/blockB already.
    // We still scan events/ticks/moving to find the overall anchor, but skip
    // the block-entry scan entirely when the cache says the block anchor is stable.

    auto anchorBeats = [&](BlockPos pos, BlockPos cur, bool hasCur) {
        if (!hasCur) return true;
        if (pos.y != cur.y) return pos.y < cur.y;
        if (pos.z != cur.z) return pos.z < cur.z;
        return pos.x < cur.x;
    };

    // Block anchor (from cache or full scan)
    BlockPos blockAnchor = {0, 0, 0};
    bool haveBlockAnchor = false;
    std::uint64_t blockA, blockB;

    if (!bhc_.dirty && bhc_.hasAnchor) {
        // Fast path: use cached anchor and hash
        blockAnchor    = bhc_.anchor;
        haveBlockAnchor = true;
        blockA = bhc_.blockA;
        blockB = bhc_.blockB;
    } else {
        // Slow path: full scan — O(n) but sequential
        blockA = 0x9e3779b97f4a7c15ULL;
        blockB = 0xc2b2ae3d27d4eb4fULL;
        for (const auto& e : blockEntries) {
            BlockPos pos = unpackPos(e.key);
            if (anchorBeats(pos, blockAnchor, haveBlockAnchor)) {
                blockAnchor    = pos;
                haveBlockAnchor = true;
            }
        }
        // Second pass: hash relative to blockAnchor
        if (haveBlockAnchor) {
            for (const auto& e : blockEntries) {
                BlockPos pos = unpackPos(e.key);
                std::uint64_t enc = bhcEncodeRel(pos.x - blockAnchor.x,
                                                  pos.y - blockAnchor.y,
                                                  pos.z - blockAnchor.z)
                                  ^ (static_cast<std::uint64_t>(e.state) << 17);
                std::uint64_t m = bhcMix64(enc);
                blockA += m;
                blockB ^= (m << 27) | (m >> 37);
            }
        }
        // Update cache
        bhc_.dirty     = false;
        bhc_.hasAnchor = haveBlockAnchor;
        bhc_.anchor    = blockAnchor;
        bhc_.blockA    = blockA;
        bhc_.blockB    = blockB;
    }

    // --- Overall anchor: extend block-anchor with events/ticks/moving ---
    anchor = blockAnchor;
    bool haveAnchor = haveBlockAnchor;
    auto consider = [&](BlockPos pos) {
        if (anchorBeats(pos, anchor, haveAnchor)) {
            anchor     = pos;
            haveAnchor = true;
        }
    };
    for (const auto& queue : world_.blockEvents) {
        for (const World::BlockEvent& event : queue) consider(event.pos);
    }
    for (const World::ScheduledTick& tick : world_.scheduledTicks) consider(tick.pos);
    for (const auto& bucket : world_.movingBuckets) {
        for (const World::MovingBlock& moving : bucket) consider(moving.pos);
    }
    if (!haveAnchor) {
        anchor = BlockPos{0, 0, 0};
    }

    // If the overall anchor differs from blockAnchor (an event/tick/moving beat it),
    // we must recompute the block hash with the new anchor.
    if (haveAnchor && haveBlockAnchor && !samePos(anchor, blockAnchor)) {
        blockA = 0x9e3779b97f4a7c15ULL;
        blockB = 0xc2b2ae3d27d4eb4fULL;
        for (const auto& e : blockEntries) {
            BlockPos pos = unpackPos(e.key);
            std::uint64_t enc = bhcEncodeRel(pos.x - anchor.x,
                                              pos.y - anchor.y,
                                              pos.z - anchor.z)
                              ^ (static_cast<std::uint64_t>(e.state) << 17);
            std::uint64_t m = bhcMix64(enc);
            blockA += m;
            blockB ^= (m << 27) | (m >> 37);
        }
        // Don't update cache here: the non-block anchor is transient
    }

    std::uint64_t eventA = 0x165667b19e3779f9ULL;
    std::uint64_t eventB = 0x85ebca77c2b2ae63ULL;
    std::uint64_t eventCount = 0;
    for (const auto& queue : world_.blockEvents) {
        for (const World::BlockEvent& event : queue) {
            std::uint64_t encoded = bhcEncodeRel(event.pos.x - anchor.x, event.pos.y - anchor.y, event.pos.z - anchor.z)
                ^ (static_cast<std::uint64_t>(event.blockId & 0xfff) << 13)
                ^ (static_cast<std::uint64_t>(event.eventId & 0xff) << 25)
                ^ (static_cast<std::uint64_t>(event.eventParam & 0xff) << 33);
            std::uint64_t mixed = bhcMix64(encoded);
            eventA += mixed;
            eventB ^= (mixed << 31) | (mixed >> 33);
            ++eventCount;
        }
    }

    std::uint64_t tickA = 0xd6e8feb86659fd93ULL;
    std::uint64_t tickB = 0xa5a3564e27f88621ULL;
    for (const World::ScheduledTick& tick : world_.scheduledTicks) {
        std::uint64_t encoded = bhcEncodeRel(tick.pos.x - anchor.x, tick.pos.y - anchor.y, tick.pos.z - anchor.z)
            ^ (static_cast<std::uint64_t>(tick.blockId & 0xfff) << 11)
            ^ (static_cast<std::uint64_t>(tick.time - world_.time) << 32);
        std::uint64_t mixed = bhcMix64(encoded);
        tickA += mixed;
        tickB ^= (mixed << 43) | (mixed >> 21);
    }

    std::uint64_t movingA = 0x94d049bb133111ebULL;
    std::uint64_t movingB = 0xbf58476d1ce4e5b9ULL;
    std::uint64_t movingCount = 0;
    for (std::size_t phase = 0; phase < world_.movingBuckets.size(); ++phase) {
        int bucketIndex = ((world_.movingPtr - static_cast<int>(phase)) % 3 + 3) % 3;
        const auto& bucket = world_.movingBuckets[static_cast<std::size_t>(bucketIndex)];
        for (const World::MovingBlock& moving : bucket) {
            std::uint64_t encoded = bhcEncodeRel(moving.pos.x - anchor.x, moving.pos.y - anchor.y, moving.pos.z - anchor.z)
                ^ (static_cast<std::uint64_t>(moving.pistonState) << 9)
                ^ (static_cast<std::uint64_t>(phase) << 29)
                ^ (static_cast<std::uint64_t>(moving.facing) << 33)
                ^ (moving.extending ? 0x100000001b3ULL : 0ULL)
                ^ (moving.shouldHeadBeRendered ? 0x9e3779b97f4a7c15ULL : 0ULL);
            std::uint64_t mixed = bhcMix64(encoded);
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

std::string Simulator::debugPistonHelper(const Candidate& candidate) {
    Result error;
    error.id = candidate.id;
    try {
        loadCandidate(candidate);
        std::uint32_t state = world_.getBlock(candidate.trigger);
        int id = blockId(state);
        if (!isPistonBlock(id)) {
            error.ok = false;
            error.errorCode = "trigger_not_piston";
            error.error = "debug piston helper requires trigger to point at piston or sticky piston";
            return error.toJson();
        }

        const Facing& facing = facingByIndex(facingMeta(state));
        PistonStructureHelper helper(world_, candidate.trigger, facing, true);
        bool canMoveResult = helper.canMove();

        std::ostringstream out;
        out << '{'
            << "\"id\":" << candidate.id
            << ",\"ok\":true"
            << ",\"mode\":\"piston_helper\""
            << ",\"piston\":{\"x\":" << candidate.trigger.x
            << ",\"y\":" << candidate.trigger.y
            << ",\"z\":" << candidate.trigger.z
            << ",\"state\":" << state
            << ",\"facing\":" << facing.index
            << ",\"sticky\":" << (isStickyPistonBlock(id) ? "true" : "false") << '}'
            << ",\"canMove\":" << (canMoveResult ? "true" : "false")
            << ",\"move\":[";
        for (int i = 0; i < helper.moveCount(); ++i) {
            if (i != 0) {
                out << ',';
            }
            BlockPos pos = helper.moveAt(i);
            out << "{\"x\":" << pos.x << ",\"y\":" << pos.y << ",\"z\":" << pos.z << '}';
        }
        out << "],\"destroy\":[";
        for (int i = 0; i < helper.destroyCount(); ++i) {
            if (i != 0) {
                out << ',';
            }
            BlockPos pos = helper.destroyAt(i);
            out << "{\"x\":" << pos.x << ",\"y\":" << pos.y << ",\"z\":" << pos.z << '}';
        }
        out << "]}";
        return out.str();
    } catch (const std::exception& ex) {
        error.ok = false;
        error.errorCode = "piston_helper_error";
        error.error = ex.what();
        return error.toJson();
    }
}

std::string Simulator::debugPistonMove(const Candidate& candidate) {
    Result error;
    error.id = candidate.id;
    try {
        loadCandidate(candidate);
        std::uint32_t before = world_.getBlock(candidate.trigger);
        int id = blockId(before);
        if (!isPistonBlock(id)) {
            error.ok = false;
            error.errorCode = "trigger_not_piston";
            error.error = "debug piston move requires trigger to point at piston or sticky piston";
            return error.toJson();
        }

        const Facing& facing = facingByIndex(facingMeta(before));
        bool sticky = isStickyPistonBlock(id);
        bool moved = pistonDoMove(world_, candidate.trigger, facing, true, sticky);

        std::ostringstream out;
        out << '{'
            << "\"id\":" << candidate.id
            << ",\"ok\":true"
            << ",\"mode\":\"piston_move_immediate\""
            << ",\"warning\":\"settles blocks immediately; moving piston tile entity timing is not ported yet\""
            << ",\"moved\":" << (moved ? "true" : "false")
            << ",\"pistonStateBefore\":" << before
            << ",\"pistonStateAfter\":" << world_.getBlock(candidate.trigger)
            << ",\"liveBlocks\":" << world_.liveBlocks()
            << '}';
        return out.str();
    } catch (const std::exception& ex) {
        error.ok = false;
        error.errorCode = "piston_move_error";
        error.error = ex.what();
        return error.toJson();
    }
}

void Simulator::loadCandidate(const Candidate& candidate) {
    bhc_ = BlockHashCache{};  // reset cache
    world_.reset();
    world_.blocks.reserve(candidate.blocks.size());

    for (const BlockEntry& block : candidate.blocks) {
        if (block.state != 0) {
            BlockPos pos{block.x, block.y, block.z};
            world_.setBlock(pos, block.state);
            if (trace_ != nullptr) {
                trace_->log(world_, "cpp.load", &pos, static_cast<int>(block.state), 0);
            }
        }
    }

    // Build watchSet_ / observerSet_ / ncPowerSet_ from loaded blocks.
    // world_.setBlock() bypasses setBlockState(), so scan entries once after load.
    observerSet_.clear();
    watchSet_.clear();
    ncPowerSet_.clear();
    for (const auto& e : world_.blocks.entries()) {
        int id = blockId(e.state);
        if (id == BLOCK_OBSERVER) observerSet_.insert(e.key);
        if (isWatchedBlock(id)) watchSet_.insert(e.key);
        if (isNcPowerSource(id, e.state)) ncPowerSet_.insert(e.key);
    }

    if (trace_ != nullptr) {
        trace_->log(world_, "cpp.trigger", &candidate.trigger, 0, 0);
    }
}

} // namespace mcp1122
