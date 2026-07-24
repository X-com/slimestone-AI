#pragma once

#include "packed_pos.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace mcp1122 {

// Optional, per-candidate binary event log ("simulation_data") feeding RL training. Unlike Trace
// (text, incremental), this buffers fixed-width records in RAM during the run and, at close(),
// groups them per subject block into contiguous runs so a reader can go from one block straight to
// that block's complete, in-order, self-contained history - no chain-walking, no joins. See the
// plan / verify_simulation_data.py for the authoritative on-disk layout (both sides must stay in
// sync). Everything here is skipped entirely when the owning Simulator's eventLog_ pointer is null.

enum SimEventKind : std::uint8_t {
    PistonQueued = 0,              // piston (subject) had an extend/retract queued
    PistonMoveExecuted = 1,        // piston (subject) executed a move; flags bit1=0 if blocked
    BlockPushed = 2,               // subject block carried by a piston; actorKey=piston, targetKey=dest
    ObserverFired = 3,             // observer (subject) pulsed; flags bits2-3 = cause
    ObserverActivated = 4,         // observer (subject) pulse reached targetKey; flags bit4 = target is piston
    RedstoneBlockAppeared = 5,     // redstone block (subject) placed at a position
    RedstoneBlockRemoved = 6,      // redstone block (subject) removed
    RedstoneActivatedPiston = 7,   // redstone block (subject) powers targetKey piston
    RedstoneDeactivatedPiston = 8, // redstone block (subject) removal unpowers targetKey piston
};

// flags bits
constexpr std::uint8_t SEF_EXTEND       = 1 << 0; // set = extend, clear = retract
constexpr std::uint8_t SEF_SUCCESS      = 1 << 1; // set = executed/moved, clear = blocked
constexpr std::uint8_t SEF_TARGET_PISTON = 1 << 4; // ObserverActivated: target is a piston (else observer)

// ObserverFired cause, stored in flags bits 2-3.
constexpr std::uint8_t SEC_SCHEDULED       = 0; // generic scheduled pulse
constexpr std::uint8_t SEC_FACING_CHANGED  = 1; // the block the observer faces changed
constexpr std::uint8_t SEC_OBSERVER_MOVED  = 2; // the observer itself was moved
inline std::uint8_t observerCauseFlags(std::uint8_t cause) { return static_cast<std::uint8_t>((cause & 0x3) << 2); }

constexpr std::uint8_t SE_NO_DIRECTION = 0xFF;

#pragma pack(push, 1)
struct SimEvent {
    std::uint64_t blockKey = 0;          // subject: whose log this record belongs to (stable original id)
    std::uint64_t actorKey = 0;          // piston/observer/redstone that caused it (raw pos; 0 = n/a)
    std::uint64_t targetKey = 0;         // pulse target / pushed-block destination (raw pos; 0 = n/a)
    std::int64_t  activationTick = 0;    // tick the event became relevant
    std::int64_t  scheduledTick = 0;     // tick the piston move was queued (= activationTick otherwise)
    std::int64_t  executedTick = 0;      // tick it actually moved/fired (= activationTick otherwise)
    std::uint32_t activationSubtick = 0; // global monotonic order at activation
    std::uint32_t scheduledSubtick = 0;
    std::uint32_t executedSubtick = 0;
    std::uint32_t pushGroupId = 0;       // shared by every event from one doPistonMove call (0 = n/a)
    std::uint8_t  kind = 0;
    std::uint8_t  direction = SE_NO_DIRECTION; // Facing::index 0-5, 0xFF = n/a
    std::uint8_t  flags = 0;
    std::uint8_t  attemptedAmount = 0;
    std::uint8_t  actualAmount = 0;
    std::uint8_t  reserved0 = 0;
    std::uint16_t reserved1 = 0;
};

struct BlockIndexEntry {
    std::uint64_t originalKey = 0;   // packPos() at load - the stable subject id
    std::uint64_t currentKey = 0;    // where the block ended up
    std::uint32_t firstEventIdx = 0; // start of this block's contiguous run
    std::uint32_t eventCount = 0;
    std::uint32_t originalState = 0; // block state at load (type/meta, no second lookup)
    std::uint32_t reserved = 0;
};

struct SimLogFooter {
    char          magic[4] = {'S', 'D', 'L', '2'};
    std::uint32_t version = 2;
    std::uint64_t eventCount = 0;
    std::uint64_t blockIndexOffset = 0;
    std::uint32_t blockCount = 0;
    std::uint32_t eventRecSize = sizeof(SimEvent);
    std::uint32_t blockRecSize = sizeof(BlockIndexEntry);
    std::uint32_t reserved = 0;
    std::uint64_t reserved2 = 0;
};
#pragma pack(pop)

static_assert(sizeof(SimEvent) == 72, "SimEvent must be 72 bytes");
static_assert(sizeof(BlockIndexEntry) == 32, "BlockIndexEntry must be 32 bytes");
static_assert(sizeof(SimLogFooter) == 48, "SimLogFooter must be 48 bytes");
static_assert(std::is_standard_layout<SimEvent>::value, "SimEvent must be standard-layout");

struct QueueInfo {
    std::int64_t tick = 0;
    std::uint32_t subtick = 0;
    bool found = false;
};

class SimEventLog {
public:
    void open(const std::string& path);   // closes previous if open, resets buffers
    bool enabled() const { return out_.is_open(); }
    void close();                           // group by block, write events + index + footer

    ~SimEventLog() { if (enabled()) close(); }

    void registerOriginalBlock(std::uint64_t originalKey, std::uint32_t originalState);
    void setCurrentKey(std::uint64_t originalKey, std::uint64_t currentKey);

    std::uint32_t nextOrder() { return nextOrder_++; }
    std::uint32_t nextPushGroupId() { return ++nextPushGroupId_; }

    void noteQueued(std::uint64_t pistonKey, bool extend, std::int64_t tick, std::uint32_t subtick);
    QueueInfo takeQueued(std::uint64_t pistonKey, bool extend);

    void push(const SimEvent& ev) { buffer_.push_back(ev); }

    // Self-check: round-trips synthetic interleaved events through a temp file + inline reader,
    // asserts each block's run reconstructs complete and in order. Returns true on PASS.
    static bool selfTest();

private:
    std::ofstream out_;
    std::uint32_t nextOrder_ = 0;
    std::uint32_t nextPushGroupId_ = 0;
    std::vector<SimEvent> buffer_;
    std::vector<BlockIndexEntry> blockIndex_;
    std::unordered_map<std::uint64_t, std::size_t> indexOf_;      // originalKey -> blockIndex_ slot
    std::unordered_map<std::uint64_t, QueueInfo> pendingQueue_;   // (pistonKey<<1|extend) -> queue timing

    static std::uint64_t queueKey(std::uint64_t pistonKey, bool extend) {
        return (pistonKey << 1) | (extend ? 1u : 0u);
    }
};

} // namespace mcp1122
