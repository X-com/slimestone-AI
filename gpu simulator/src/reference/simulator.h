#pragma once

#include "facing.h"
#include "json_stream.h"  // mcp1122::Candidate, reused unmodified from cpp extract
#include "piston.h"
#include "trace.h"
#include "world.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mcp1122gpu {

using mcp1122::Facing;

// Mirrors cpp extract's Simulator::StateKey exactly (same word layout, same hash algorithm)
// so the two are diffable word-for-word.
struct StateKey {
    std::array<std::uint64_t, 12> words{};
    bool operator==(const StateKey& o) const noexcept { return words == o.words; }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& k) const noexcept {
        std::uint64_t h = k.words[4] ^ k.words[5] ^ k.words[6] ^ k.words[7]
                        ^ k.words[8] ^ k.words[9] ^ k.words[10] ^ k.words[11];
        return static_cast<std::size_t>(h);
    }
};

struct Result {
    std::int64_t id = 0;
    bool ok = false;
    bool working = false;
    int ticks = 0;
    int start = 0;
    int end = 0;
    int period = 0;
    BlockPos shift;
    bool cycles = false;
    bool settled = false;
    bool validCycle = false;
    BlockPos finalShift;
    std::int64_t elapsedNs = 0;
    std::string errorCode;
    std::string error;

    std::string toJson() const;
};

// Full port of cpp extract's Simulator, adapted to FixedWorld. FixedWorld::entries() is now an
// incrementally maintained dense list (O(live count) to read, not O(kExtent^3)), and
// watchSet_/observerSet_/ncPowerSet_ are fixed-capacity PosKeySet, not std::unordered_set - so
// this class no longer allocates or scans the whole cube on the hot path. One deliberate
// simplification remains, performance-only (never correctness-affecting): no incremental
// block-hash cache (cpp extract's BlockHashCache/bhc_) - stateKey() still rehashes every live
// block from scratch each tick rather than applying a per-setBlockState delta. That's fine
// while live counts stay in the tens-to-low-thousands range fixtures/flyers.data exercise;
// revisit only if per-tick cost becomes the bottleneck again.
class Simulator {
public:
    explicit Simulator(Trace* trace) : trace_(trace) {}
    Result simulate(const mcp1122::Candidate& candidate);

private:
    struct ShiftCycle {
        int start = 0;
        int end = 0;
        int period = 0;
        BlockPos shift;
    };

    struct SeenState {
        int tick = 0;
        BlockPos anchor;
    };

    FixedWorld world_;
    Trace* trace_ = nullptr;
    PosKeySet watchSet_;
    PosKeySet observerSet_;
    PosKeySet ncPowerSet_;

    // Reused scratch buffers for draining scheduledTicks/blockEvents[idx]/movingBuckets[idx]
    // each tick without swapping capacity out to a fresh local (which would free it on scope
    // exit and force a realloc on the container's next use, every few ticks, for any candidate
    // with ongoing piston/event activity) - mirrors cpp extract's pendingDue_ pattern.
    std::vector<ScheduledTick> pendingDue_;
    std::vector<BlockEvent> pendingBlockEvents_;
    std::vector<MovingBlock> pendingDoneMoving_;

    // Trigger pulse / burnout state (see Simulator::trigger).
    BlockPos triggerPos_;
    bool triggerCharged_ = false;
    std::int64_t triggerEndTick_ = -1;
    bool triggerDisabled_ = false;

    void loadCandidate(const mcp1122::Candidate& candidate);
    void trigger(BlockPos pos);
    void tickWorld();
    void tickScheduledUpdates();
    void observerUpdateTick(BlockPos pos, std::uint32_t state);
    void notifyObserverFront(BlockPos pos, std::uint32_t state);
    void sendQueuedBlockEvents();
    bool fireBlockEvent(const BlockEvent& event);
    void updateEntities();
    void settleMovingBlock(const MovingBlock& moving);
    void addMovingBlock(const MovingBlock& block);
    void lampUpdateTick(BlockPos pos, std::uint32_t state);
    void scheduleUpdate(BlockPos pos, int blockIdValue, int delay);
    bool isUpdateScheduled(BlockPos pos, int blockIdValue) const;
    void addBlockEvent(BlockPos pos, int blockIdValue, int eventId, int eventParam);
    void neighborChanged(BlockPos pos, int sourceBlockId, BlockPos fromPos);
    void neighborChangedImpl(BlockPos pos, std::uint64_t key, int sourceBlockId, BlockPos fromPos);
    void observedNeighborChanged(BlockPos pos, int changedBlockId, BlockPos changedPos);
    void notifyNeighbors(BlockPos pos, int sourceBlockId, bool updateObservers);
    void notifyNeighborsExcept(BlockPos pos, int sourceBlockId, int skipFacing);
    void updateObservingBlocksAt(BlockPos pos, int sourceBlockId);
    void checkForMove(BlockPos pos, std::uint32_t state);
    bool shouldPistonBeExtended(BlockPos pos, std::uint32_t state) const;
    void railNeighborChanged(BlockPos pos, std::uint32_t state);
    void updateRailPowerState(BlockPos pos, std::uint32_t state, int railId, int shape);
    bool findPoweredRailSignal(int railId, BlockPos pos, std::uint32_t state, bool forward, int distance) const;
    bool isSameRailWithPower(int railId, BlockPos pos, bool forward, int distance, int expectedShape) const;
    bool isTopSolid(BlockPos pos) const;
    bool isBlockPowered(BlockPos pos) const;
    bool isSidePowered(BlockPos pos, const Facing& side) const;
    int redstonePower(BlockPos pos, const Facing& side) const;
    int weakPower(std::uint32_t state, const Facing& side) const;
    int strongPower(BlockPos pos, const Facing& side) const;
    int strongPowerAll(BlockPos pos) const;
    bool pistonEventReceived(BlockPos pos, std::uint32_t state, int id, int param);
    bool doPistonMove(BlockPos pos, const Facing& direction, bool extending, bool sticky);
    void setBlockState(BlockPos pos, std::uint32_t state, int flags);
    void setBlockToAir(BlockPos pos);
    void removeMovingAt(BlockPos pos);
    bool clearMovingAt(BlockPos pos);
    bool hasMovingAt(BlockPos pos) const;
    bool isExtendingMovingAt(BlockPos pos, int facing) const;
    ShiftCycle* detectShiftCycle(int maxTicks, ShiftCycle& out);
    bool isQuiescent() const;
    bool compareFinalToInitial(const mcp1122::Candidate& candidate, BlockPos& outShift) const;
    StateKey stateKey(BlockPos& anchor) const;
    static bool samePos(BlockPos a, BlockPos b);
};

} // namespace mcp1122gpu
