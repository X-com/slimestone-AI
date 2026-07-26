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
    // SDL3 additions (appended, never renumbered):
    PistonExtendBlocked = 9,       // piston (subject) wanted to extend but could not (see failureReason)
    PistonRetractBlocked = 10,     // piston (subject) wanted to retract but could not
    BlockLeftBehind = 11,          // sticky pull failed / block detached from its group (reserved, not yet emitted)
    BlockDestroyed = 12,           // block pushed into a destroying condition (reserved, not yet emitted)
    ComponentSplit = 13,           // a connected group tore apart (reserved, not yet emitted)
    ObserverSuppressed = 14,       // observer fired but had no effect (reserved, not yet emitted)
    // piston (subject) was notified because SOME neighboring block changed (the generic mechanism
    // every block placement uses to poke its 6 neighbors - notifyNeighbors/neighborChangedImpl).
    // This is the catch-all cause: a piston's own head appearing next to its base, a rail forming,
    // a fence gate toggling nearby, etc. all route through here even when no more specific kind
    // (ObserverActivated / Redstone*Piston / BlockPushed-self) applies. actorKey = stableKey of the
    // position that changed (fromPos); reserved0 carries the raw block id that WAS there before the
    // change (sourceBlockId, 0-255, fits in a byte) so a reader isn't forced to infer it.
    PistonNeighborNotified = 15,
};

// SimEvent.failureReason (0 = success/none). Populated on the *Blocked and PistonMoveExecuted(blocked)
// records, and mirrored onto the PushGroupRecord.
enum SimFailureReason : std::uint8_t {
    FAIL_NONE                 = 0,
    FAIL_PUSH_LIMIT_EXCEEDED  = 1,  // push group would exceed 12; PushGroupRecord.attemptedCount carries the size
    FAIL_IMMOVABLE_IN_PATH    = 2,  // an immovable block (obsidian, tile entity, extended piston) blocks the path
    FAIL_NO_SPACE_TO_EXTEND   = 3,
    FAIL_BLOCK_CANNOT_BE_PUSHED = 4,
    FAIL_ALREADY_IN_TARGET_STATE = 5,
    FAIL_NOT_POWERED          = 6,
    FAIL_OUT_OF_BOUNDS        = 7,
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

// InitialBlockState.movabilityClass / stickinessClass
constexpr std::uint8_t MOVABILITY_MOVABLE   = 0;
constexpr std::uint8_t MOVABILITY_IMMOVABLE = 1;
constexpr std::uint8_t MOVABILITY_POPS       = 2; // pushReaction::Destroy
constexpr std::uint8_t STICKINESS_NONE       = 0;
constexpr std::uint8_t STICKINESS_ALL        = 1; // slime
constexpr std::uint8_t STICKINESS_ALL_EXCEPT_SLIME = 2; // honey (not present in this registry, reserved)
// PushReaction::PushOnly blocks (glazed terracotta): pushable, but never drags a neighbor and
// never gets dragged itself, even by an adjacent slime block - matches canPush()'s own
// `case PushReaction::PushOnly: return facing.index == pushFacing.index;` rule (piston.cpp),
// which already refuses any perpendicular/pull movement at runtime. This class exists so the
// static t=0 sticky-component computation (loadCandidate) agrees with that runtime rule instead
// of gluing these blocks into a component the real physics would never actually drag together.
constexpr std::uint8_t STICKINESS_NEVER       = 3;

// RunSummary.terminationReason
enum SimTermination : std::uint8_t {
    TERM_CYCLE_DETECTED     = 0,
    TERM_TICK_BUDGET        = 1,
    TERM_NOTHING_HAPPENED   = 2,
    TERM_STRUCTURE_DESTROYED = 3,
    TERM_OUT_OF_BOUNDS      = 4,
    TERM_INTERNAL_ERROR     = 5,
};

#pragma pack(push, 1)
struct SimEvent {
    std::uint64_t blockKey = 0;          // subject: whose log this record belongs to (stable original id)
    std::uint64_t actorKey = 0;          // piston/observer/redstone that caused it (raw pos; 0 = n/a)
    std::uint64_t targetKey = 0;         // pulse target / pushed-block destination (raw pos; 0 = n/a)
    std::uint64_t globalSeq = 0;         // monotonic across the whole run, never reused (== activationSubtick here)
    std::int64_t  activationTick = 0;    // tick the event became relevant
    std::int64_t  scheduledTick = 0;     // tick the piston move was queued (= activationTick otherwise)
    std::int64_t  executedTick = 0;      // tick it actually moved/fired (= activationTick otherwise)
    std::uint32_t activationSubtick = 0; // global monotonic order at activation
    std::uint32_t scheduledSubtick = 0;
    std::uint32_t executedSubtick = 0;
    std::uint32_t pushGroupId = 0;       // shared by every event from one doPistonMove call (0 = n/a)
    std::int16_t  fromX = 0, fromY = 0, fromZ = 0; // subject position before (== to for non-move events)
    std::int16_t  toX = 0, toY = 0, toZ = 0;       // subject position after
    std::uint8_t  kind = 0;
    std::uint8_t  direction = SE_NO_DIRECTION; // Facing::index 0-5, 0xFF = n/a
    std::uint8_t  flags = 0;
    std::uint8_t  attemptedAmount = 0;
    std::uint8_t  actualAmount = 0;
    std::uint8_t  failureReason = 0;     // SimFailureReason; 0 = success
    std::uint8_t  reserved0 = 0;
    std::uint8_t  reserved1 = 0;
    std::uint32_t reserved2 = 0;
};

struct BlockIndexEntry {
    std::uint64_t originalKey = 0;   // packPos() at load - the stable subject id
    std::uint64_t currentKey = 0;    // where the block ended up
    std::uint32_t firstEventIdx = 0; // start of this block's contiguous run
    std::uint32_t eventCount = 0;
    std::uint32_t originalState = 0; // block state at load (type/meta, no second lookup)
    std::uint32_t reserved = 0;
};

// One record per piston firing ATTEMPT (success or fail). Members index into the flat pushMembers[]
// array via memberOffset/memberCount. Failed attempts are recorded with full would-be membership.
// Reused as-is (same type, separate section + separate member array) for the STATIC push-group
// preview: one speculative PistonStructureHelper::canMove() call per piston at load time, for
// whichever action its current extended state implies is next (extend if retracted, retract if
// extended) - so every piston has a group-size feature available even if it never actually fires
// during the observed run. In that static section, tick/subtick/globalSeq are not meaningful (always
// 0) - the section itself (not a field) tells the reader "this is a t=0 prediction, not an event".
struct PushGroupRecord {
    std::uint64_t globalSeq = 0;
    std::uint64_t pistonKey = 0;     // stable id of the acting piston
    std::int32_t  tick = 0;
    std::uint16_t subtick = 0;
    std::uint8_t  direction = SE_NO_DIRECTION;
    std::uint8_t  succeeded = 0;     // 0/1
    std::uint8_t  failureReason = 0; // SimFailureReason
    std::uint8_t  pad[3] = {0, 0, 0};
    std::uint32_t memberCount = 0;
    std::uint32_t memberOffset = 0;  // index into pushMembers[]
    std::uint32_t attemptedCount = 0; // real group size (13/14... for over-limit), not the limit
    std::uint64_t reserved = 0;
};

// The model's input: one per original block, complete and self-contained (no source-JSON needed).
struct InitialBlockState {
    std::uint64_t stableKey = 0;
    std::int16_t  x = 0, y = 0, z = 0;
    std::uint16_t blockTypeId = 0;
    std::uint8_t  facing = SE_NO_DIRECTION;
    std::uint8_t  stateFlags = 0;      // bit0 extended, bit1 powered, bit2 open (best-effort)
    std::uint8_t  movabilityClass = 0; // MOVABILITY_*
    std::uint8_t  stickinessClass = 0; // STICKINESS_*
    std::int16_t  componentId = -1;
    std::uint8_t  isTrigger = 0;
    std::uint8_t  pad = 0;
    std::uint32_t rawState = 0;        // full state word (type|meta), no second lookup
    std::uint32_t reserved = 0;
};

// Static, ahead-of-time relation: sourceKey statically powers pistonKey given the t=0 board, computed
// once at load via the simulator's own power-resolution code (not reimplemented) - independent of
// whether that activation ever actually happens during the observed run. viaQC distinguishes a direct
// adjacency from a quasi-connectivity path (through the block above the piston) - the model needs this
// distinction (see sim_event_log.h's PistonNeighborNotified doc / the QC design discussion): QC powers
// but never itself fires an update, so `would_power` must expose it or the model sees uncaused pistons.
struct WouldPowerEdge {
    std::uint64_t sourceKey = 0;   // stable id of the power-providing block (redstone/observer/etc.)
    std::uint64_t pistonKey = 0;   // stable id of the piston it would power
    std::uint8_t  viaQC = 0;       // 0 = direct adjacency, 1 = via quasi-connectivity
    std::uint8_t  pad[7] = {0, 0, 0, 0, 0, 0, 0};
};

// Connected sticky group at t=0. Members index into the flat componentMembers[] array.
struct ComponentRecord {
    std::int16_t  componentId = 0;
    std::uint8_t  containsImmovable = 0;
    std::uint8_t  pad = 0;
    std::uint32_t memberCount = 0;
    std::int16_t  bboxMin[3] = {0, 0, 0};
    std::int16_t  bboxMax[3] = {0, 0, 0};
    std::uint32_t memberOffset = 0;    // index into componentMembers[]
    std::uint64_t reserved = 0;
};

// One per run. Also carries the initial-state "header" (bbox / trigger / travel axis) so no separate
// header section is needed. Simulator sets the logical fields; close() fills the measured-from-log
// fields (totalEvents, distinctBlocksWithEvents, maxPushGroupSize, pushLimitFailureCount).
struct RunSummary {
    std::uint8_t  terminationReason = TERM_INTERNAL_ERROR;
    std::uint8_t  validCycle = 0;
    std::int8_t   travelAxis = -1;     // 0=x 1=y 2=z, -1 unknown
    std::uint8_t  pad = 0;
    std::int32_t  totalTicks = 0;
    std::int32_t  period = 0;          // 0 if none
    std::int16_t  netShift[3] = {0, 0, 0};
    std::int16_t  bboxMin[3] = {0, 0, 0};
    std::int16_t  bboxMax[3] = {0, 0, 0};
    std::int16_t  triggerPos[3] = {0, 0, 0};
    std::uint32_t totalEvents = 0;
    std::uint32_t distinctBlocksWithEvents = 0;
    std::uint32_t maxObserverChainDepth = 0; // not computed yet (0); upgrade if the model needs it
    std::uint32_t maxPushGroupSize = 0;
    std::uint32_t pushLimitFailureCount = 0;
    std::uint32_t blockCount = 0;      // number of original blocks (== InitialBlockState count)
    std::uint32_t reserved = 0;
};

// EOF footer. Readers seek from the end. Carries offset+count for every section. Bumped SDL3->SDL4
// (magic + formatVersion both change) because the footer's own byte size grows here - this codebase's
// established convention (see SDL2->SDL3) is a new magic per footer-layout change, so a fixed-size
// footer can always be read with a single EOF-relative seek instead of a reader having to dispatch on
// formatVersion to learn how big the footer even is.
struct SimLogFooter {
    char          magic[4] = {'S', 'D', 'L', '4'};
    std::uint32_t formatVersion = 4;
    std::uint64_t simulatorBuildHash = 0;
    std::uint64_t generatorSeed = 0;
    std::uint64_t eventCount = 0;
    std::uint64_t blockIndexOffset = 0;
    std::uint32_t blockCount = 0;
    std::uint32_t eventRecSize = sizeof(SimEvent);
    std::uint32_t blockRecSize = sizeof(BlockIndexEntry);
    std::uint64_t pushGroupOffset = 0;
    std::uint32_t pushGroupCount = 0;
    std::uint32_t pushGroupRecSize = sizeof(PushGroupRecord);
    std::uint64_t pushMemberOffset = 0;
    std::uint32_t pushMemberCount = 0;
    std::uint32_t pad0 = 0;
    std::uint64_t initialOffset = 0;
    std::uint32_t initialCount = 0;
    std::uint32_t initialRecSize = sizeof(InitialBlockState);
    std::uint64_t componentOffset = 0;
    std::uint32_t componentCount = 0;
    std::uint32_t componentRecSize = sizeof(ComponentRecord);
    std::uint64_t componentMemberOffset = 0;
    std::uint32_t componentMemberCount = 0;
    std::uint32_t summaryRecSize = sizeof(RunSummary);
    std::uint64_t summaryOffset = 0;
    // SDL4 additions:
    std::uint64_t wouldPowerOffset = 0;
    std::uint32_t wouldPowerCount = 0;
    std::uint32_t wouldPowerRecSize = sizeof(WouldPowerEdge);
    // Static push-group preview reuses PushGroupRecord (see its doc comment) but is a distinct
    // section + distinct member array from the dynamic pushGroup*/pushMember* fields above.
    std::uint64_t staticPushGroupOffset = 0;
    std::uint32_t staticPushGroupCount = 0;
    std::uint32_t staticPushMemberCount = 0;
    std::uint64_t staticPushMemberOffset = 0;
    std::uint64_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(SimEvent) == 96, "SimEvent must be 96 bytes");
static_assert(sizeof(BlockIndexEntry) == 32, "BlockIndexEntry must be 32 bytes");
static_assert(sizeof(PushGroupRecord) == 48, "PushGroupRecord must be 48 bytes");
static_assert(sizeof(InitialBlockState) == 32, "InitialBlockState must be 32 bytes");
static_assert(sizeof(ComponentRecord) == 32, "ComponentRecord must be 32 bytes");
static_assert(sizeof(RunSummary) == 64, "RunSummary must be 64 bytes");
static_assert(sizeof(WouldPowerEdge) == 24, "WouldPowerEdge must be 24 bytes");
static_assert(sizeof(SimLogFooter) == 188, "SimLogFooter must be 188 bytes");
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
    void close();                           // group by block, write all sections + footer

    ~SimEventLog() { if (enabled()) close(); }

    void registerOriginalBlock(std::uint64_t originalKey, std::uint32_t originalState);
    void setCurrentKey(std::uint64_t originalKey, std::uint64_t currentKey);

    std::uint32_t nextOrder() { return nextOrder_++; }
    std::uint32_t nextPushGroupId() { return ++nextPushGroupId_; }

    void noteQueued(std::uint64_t pistonKey, bool extend, std::int64_t tick, std::uint32_t subtick);
    QueueInfo takeQueued(std::uint64_t pistonKey, bool extend);

    // globalSeq always mirrors activationSubtick here (one monotonic counter for the whole run) -
    // set centrally so every call site gets it for free instead of repeating it at each emit.
    void push(SimEvent ev) { ev.globalSeq = ev.activationSubtick; buffer_.push_back(ev); }

    // SDL3 section setters, called by the simulator.
    void addPushGroup(std::uint64_t globalSeq, std::uint64_t pistonKey, std::int32_t tick,
                      std::uint16_t subtick, std::uint8_t direction, bool succeeded,
                      std::uint8_t failureReason, std::uint32_t attemptedCount,
                      const std::vector<std::uint64_t>& members);
    void setInitialState(std::vector<InitialBlockState> initial) { initial_ = std::move(initial); }
    void setComponents(std::vector<ComponentRecord> components, std::vector<std::uint64_t> members) {
        components_ = std::move(components);
        componentMembers_ = std::move(members);
    }
    void setSummary(const RunSummary& summary) { summary_ = summary; hasSummary_ = true; }

    // SDL4 section setters.
    void setWouldPower(std::vector<WouldPowerEdge> edges) { wouldPower_ = std::move(edges); }
    void setStaticPushPreview(std::vector<PushGroupRecord> records, std::vector<std::uint64_t> members) {
        staticPushGroups_ = std::move(records);
        staticPushMembers_ = std::move(members);
    }

    // Self-check: round-trips synthetic events + one of every new section through a temp file + inline
    // reader, asserts each reconstructs correctly. Returns true on PASS.
    static bool selfTest();

private:
    std::ofstream out_;
    std::uint32_t nextOrder_ = 0;
    std::uint32_t nextPushGroupId_ = 0;
    std::vector<SimEvent> buffer_;
    std::vector<BlockIndexEntry> blockIndex_;
    std::vector<PushGroupRecord> pushGroups_;
    std::vector<std::uint64_t> pushMembers_;
    std::vector<InitialBlockState> initial_;
    std::vector<ComponentRecord> components_;
    std::vector<std::uint64_t> componentMembers_;
    std::vector<WouldPowerEdge> wouldPower_;
    std::vector<PushGroupRecord> staticPushGroups_;
    std::vector<std::uint64_t> staticPushMembers_;
    RunSummary summary_;
    bool hasSummary_ = false;
    std::unordered_map<std::uint64_t, std::size_t> indexOf_;      // originalKey -> blockIndex_ slot
    std::unordered_map<std::uint64_t, QueueInfo> pendingQueue_;   // (pistonKey<<1|extend) -> queue timing

    static std::uint64_t queueKey(std::uint64_t pistonKey, bool extend) {
        return (pistonKey << 1) | (extend ? 1u : 0u);
    }
};

} // namespace mcp1122
