#pragma once

#include "json_stream.h"
#include "facing.h"
#include "trace.h"
#include "world.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

namespace mcp1122 {

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

class Simulator {
public:
    explicit Simulator(Trace* trace) : trace_(trace) {}
    Result simulate(const Candidate& candidate);
    std::string debugPistonHelper(const Candidate& candidate);
    std::string debugPistonMove(const Candidate& candidate);

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

    struct StateKey {
        std::array<std::uint64_t, 12> words{};
        bool operator==(const StateKey& o) const noexcept { return words == o.words; }
    };

    struct StateKeyHash {
        std::size_t operator()(const StateKey& k) const noexcept {
            // XOR the hash words (indices 4-11); counts in 0-3 are already captured there
            std::uint64_t h = k.words[4] ^ k.words[5] ^ k.words[6] ^ k.words[7]
                            ^ k.words[8] ^ k.words[9] ^ k.words[10] ^ k.words[11];
            return static_cast<std::size_t>(h);
        }
    };

    // Incremental block-hash cache.
    // Maintained in setBlockState(); used by stateKey() to skip the full
    // entries() scan when the block anchor has not changed.
    struct BlockHashCache {
        bool dirty      = true;   // must recompute from scratch
        bool hasAnchor  = false;
        BlockPos anchor = {0, 0, 0};
        std::uint64_t blockA = 0x9e3779b97f4a7c15ULL;
        std::uint64_t blockB = 0xc2b2ae3d27d4eb4fULL;
    };

    World world_;
    Trace* trace_ = nullptr;
    std::vector<World::ScheduledTick> pendingDue_;
    mutable BlockHashCache bhc_;
    // Positions of blocks that react to neighborChanged (pistons, piston heads, fence gates).
    // Maintained in setBlockState() so neighborChanged() can skip the main hash table lookup
    // for the overwhelming majority of positions that hold air or inert blocks.
    std::unordered_set<std::uint64_t> watchSet_;
    // Positions of observer blocks — same purpose for observedNeighborChanged().
    std::unordered_set<std::uint64_t> observerSet_;
    // Positions of blocks that can contribute strong power *through* a normal cube:
    // powered observers (metaBit 3) and powered detector rails (metaBit 3).
    // Redstone blocks are excluded because they supply weak power directly and never
    // cause strongPowerAll() to return non-zero.
    // When this set is empty (no observers or rails are currently active), every
    // strongPowerAll() call can be skipped entirely — the common case for machines
    // that use only redstone blocks as power sources.
    std::unordered_set<std::uint64_t> ncPowerSet_;

    // Trigger pulse / burnout state (see Simulator::trigger).
    BlockPos triggerPos_;
    bool triggerCharged_ = false;
    std::int64_t triggerEndTick_ = -1;
    bool triggerDisabled_ = false;

    void loadCandidate(const Candidate& candidate);
    void trigger(BlockPos pos);
    void tickWorld();
    void tickScheduledUpdates();
    void observerUpdateTick(BlockPos pos, std::uint32_t state);
    void notifyObserverFront(BlockPos pos, std::uint32_t state);
    void sendQueuedBlockEvents();
    bool fireBlockEvent(const World::BlockEvent& event);
    void updateEntities();
    void settleMovingBlock(const World::MovingBlock& moving);
    void addMovingBlock(const World::MovingBlock& block);
    void lampUpdateTick(BlockPos pos, std::uint32_t state);
    void scheduleUpdate(BlockPos pos, int blockId, int delay);
    bool isUpdateScheduled(BlockPos pos, int blockId) const;
    void addBlockEvent(BlockPos pos, int blockId, int eventId, int eventParam);
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
    bool compareFinalToInitial(const Candidate& candidate, BlockPos& outShift) const;
    StateKey stateKey(BlockPos& anchor) const;
    static bool samePos(BlockPos a, BlockPos b);
};

} // namespace mcp1122
