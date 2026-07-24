#pragma once

#include "json_stream.h"
#include "facing.h"
#include "sim_event_log.h"
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
    explicit Simulator(Trace* trace, SimEventLog* eventLog = nullptr) : trace_(trace), eventLog_(eventLog) {}
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
    // Optional binary event log ("simulation_data"). Null = disabled; every hook is gated on it,
    // and the two bookkeeping maps below are only mutated when it is non-null.
    SimEventLog* eventLog_ = nullptr;
    // current position -> stable original-block id (packPos at load), re-keyed through every piston
    // move so a block's log run survives arbitrarily many pushes.
    std::unordered_map<std::uint64_t, std::uint64_t> originalIdOf_;
    // observer position -> pending fire cause (SEC_*), stashed when a pulse is scheduled/placed and
    // consumed when the observer actually fires in observerUpdateTick.
    std::unordered_map<std::uint64_t, int> observerFireCause_;
    // While a single block is relocating inside doPistonMove, its old and new positions both map to
    // this stable id - so setBlockState hooks firing mid-move (redstone/observer vacating the old
    // cell, appearing in the new) resolve to the block's stable log-run id regardless of whether
    // originalIdOf_ has been re-keyed yet.
    bool currentMoveActive_ = false;
    BlockPos currentMoveOldPos_;
    BlockPos currentMoveNewPos_;
    std::uint64_t currentMoveId_ = 0;
    std::vector<World::ScheduledTick> pendingDue_;
    // Reused scratch buffers for draining world_.blockEvents[idx] / movingBuckets[idx] each
    // tick without swapping capacity out to a fresh local (which would free it on scope exit
    // and force a realloc on the bucket's next use, every few ticks, for any candidate with
    // ongoing piston/event activity).
    std::vector<World::BlockEvent> pendingBlockEvents_;
    std::vector<World::MovingBlock> pendingDoneMoving_;
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

    // Structural (piston-usage) verification - alternative to burnout/compareFinalToInitial,
    // toggled by MCP1122_CPP_STRUCTURAL_VERIFY (see Simulator::simulate). Only touched/consulted
    // when structuralVerifyEnabled_ is true.
    bool structuralVerifyEnabled_ = false;
    std::unordered_set<std::uint64_t> validPistons_;  // positions of pistons not yet "spent"
    std::unordered_set<std::uint64_t> unmovedBlocks_;  // original blocks never yet displaced
    bool structuralShortCircuited_ = false;
    // Rail block count right after the candidate is placed. A rail losing support mid-move
    // (its supporting block briefly vacated while both are carried by the same push) is
    // normal and self-corrects once the moving-block queue settles - so this is checked
    // once, against the settled count at the confirmed cycle, rather than flagging the
    // first transient "unsupported" reading. A genuine drop means a rail was dragged
    // somewhere with nothing to land on and broke for good.
    int initialRailCount_ = 0;

    void loadCandidate(const Candidate& candidate);
    void trigger(BlockPos pos);
    bool triggerStructural(BlockPos pos);
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
    // simulation_data event-log helpers (only called when eventLog_ != nullptr).
    // Maps a live position to the stable original-block id that owns that block's log run, so a
    // block carried to a new position still logs under one blockKey. Falls back to the raw packed
    // position for anything not tracked.
    std::uint64_t stableKey(BlockPos pos) const;
    void logPistonQueued(BlockPos pistonPos, int direction, bool extend);
    void logObserverActivations(BlockPos observerPos, std::uint32_t state);
    void logRedstonePistonScan(BlockPos redstonePos, bool activating);
    bool shouldPistonBeExtended(BlockPos pos, std::uint32_t state) const;
    void railNeighborChanged(BlockPos pos, std::uint32_t state);
    void updateRailPowerState(BlockPos pos, std::uint32_t state, int railId, int shape);
    // Mirrors BlockRailBase.onBlockAdded / BlockRailBase.Rail: recomputes a rail's own shape
    // (straight, curve, or ascending) from its live neighbors whenever it's placed via
    // setBlockState with a block-type change (piston settling a moved rail, most commonly),
    // then cascades the same recompute to the up-to-two rails it newly connects to.
    void railOnBlockAdded(BlockPos pos);
    void railPlace(BlockPos pos, bool poweredHint);
    void railConnectTo(BlockPos ownPos, const BlockPos existingConn[], int existingCount,
                        BlockPos newTarget, bool hasCurves);
    bool railFindRailAt(BlockPos pos, BlockPos& outPos, int& outShape) const;
    void railConnectedPositions(BlockPos pos, int shape, BlockPos out[2], int& count) const;
    void railRemoveSoftConnections(BlockPos ownPos, BlockPos conn[], int& count) const;
    bool railHasNeighborRail(BlockPos thisPos, BlockPos dirPos) const;
    int railApplyAscending(int dir, BlockPos north, BlockPos south, BlockPos west, BlockPos east) const;
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
