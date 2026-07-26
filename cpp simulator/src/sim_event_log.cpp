#include "sim_event_log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace mcp1122 {

void SimEventLog::open(const std::string& path) {
    if (out_.is_open()) {
        close();
    }
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    buffer_.clear();
    buffer_.reserve(4096);
    blockIndex_.clear();
    pushGroups_.clear();
    pushMembers_.clear();
    initial_.clear();
    components_.clear();
    componentMembers_.clear();
    wouldPower_.clear();
    staticPushGroups_.clear();
    staticPushMembers_.clear();
    summary_ = RunSummary{};
    hasSummary_ = false;
    indexOf_.clear();
    pendingQueue_.clear();
    nextOrder_ = 0;
    nextPushGroupId_ = 0;
    out_.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
}

void SimEventLog::addPushGroup(std::uint64_t globalSeq, std::uint64_t pistonKey, std::int32_t tick,
                               std::uint16_t subtick, std::uint8_t direction, bool succeeded,
                               std::uint8_t failureReason, std::uint32_t attemptedCount,
                               const std::vector<std::uint64_t>& members) {
    PushGroupRecord rec;
    rec.globalSeq = globalSeq;
    rec.pistonKey = pistonKey;
    rec.tick = tick;
    rec.subtick = subtick;
    rec.direction = direction;
    rec.succeeded = succeeded ? 1 : 0;
    rec.failureReason = failureReason;
    rec.attemptedCount = attemptedCount;
    rec.memberOffset = static_cast<std::uint32_t>(pushMembers_.size());
    rec.memberCount = static_cast<std::uint32_t>(members.size());
    pushMembers_.insert(pushMembers_.end(), members.begin(), members.end());
    pushGroups_.push_back(rec);
}

void SimEventLog::registerOriginalBlock(std::uint64_t originalKey, std::uint32_t originalState) {
    if (indexOf_.count(originalKey)) {
        return;
    }
    BlockIndexEntry entry;
    entry.originalKey = originalKey;
    entry.currentKey = originalKey;
    entry.originalState = originalState;
    indexOf_[originalKey] = blockIndex_.size();
    blockIndex_.push_back(entry);
}

void SimEventLog::setCurrentKey(std::uint64_t originalKey, std::uint64_t currentKey) {
    auto it = indexOf_.find(originalKey);
    if (it != indexOf_.end()) {
        blockIndex_[it->second].currentKey = currentKey;
    }
}

void SimEventLog::noteQueued(std::uint64_t pistonKey, bool extend, std::int64_t tick, std::uint32_t subtick) {
    pendingQueue_[queueKey(pistonKey, extend)] = QueueInfo{tick, subtick, true};
}

QueueInfo SimEventLog::takeQueued(std::uint64_t pistonKey, bool extend) {
    auto it = pendingQueue_.find(queueKey(pistonKey, extend));
    if (it == pendingQueue_.end()) {
        return QueueInfo{};  // found == false
    }
    QueueInfo info = it->second;
    pendingQueue_.erase(it);
    return info;
}

void SimEventLog::close() {
    if (!out_.is_open()) {
        return;
    }

    // Group by subject block: sort by (blockKey, activationSubtick) so each block's events are
    // contiguous and in simulation order. stable_sort keeps insertion order for equal keys as a
    // belt-and-suspenders tiebreak (subtick is already unique).
    std::stable_sort(buffer_.begin(), buffer_.end(),
        [](const SimEvent& a, const SimEvent& b) {
            if (a.blockKey != b.blockKey) return a.blockKey < b.blockKey;
            return a.activationSubtick < b.activationSubtick;
        });

    // Assign each block's contiguous run into its index entry. An event whose blockKey was never
    // registered (shouldn't happen - every subject is an original block) gets an entry on the fly.
    std::size_t i = 0;
    while (i < buffer_.size()) {
        std::uint64_t key = buffer_[i].blockKey;
        std::size_t start = i;
        while (i < buffer_.size() && buffer_[i].blockKey == key) {
            ++i;
        }
        auto it = indexOf_.find(key);
        if (it == indexOf_.end()) {
            BlockIndexEntry entry;
            entry.originalKey = key;
            entry.currentKey = key;
            indexOf_[key] = blockIndex_.size();
            blockIndex_.push_back(entry);
            it = indexOf_.find(key);
        }
        blockIndex_[it->second].firstEventIdx = static_cast<std::uint32_t>(start);
        blockIndex_[it->second].eventCount = static_cast<std::uint32_t>(i - start);
    }

    // Block index sorted by originalKey for binary search on the read side.
    std::sort(blockIndex_.begin(), blockIndex_.end(),
        [](const BlockIndexEntry& a, const BlockIndexEntry& b) { return a.originalKey < b.originalKey; });

    // Fill the measured-from-log summary fields (the simulator set the logical ones via setSummary).
    RunSummary footerSummary = summary_;
    footerSummary.totalEvents = static_cast<std::uint32_t>(buffer_.size());
    std::uint32_t distinct = 0;
    for (std::size_t j = 0; j < buffer_.size();) {
        std::uint64_t key = buffer_[j].blockKey;
        ++distinct;
        while (j < buffer_.size() && buffer_[j].blockKey == key) ++j;
    }
    footerSummary.distinctBlocksWithEvents = distinct;
    std::uint32_t maxGroup = 0, pushLimitFails = 0;
    for (const PushGroupRecord& g : pushGroups_) {
        maxGroup = std::max(maxGroup, g.attemptedCount);
        if (g.failureReason == FAIL_PUSH_LIMIT_EXCEEDED) ++pushLimitFails;
    }
    footerSummary.maxPushGroupSize = maxGroup;
    footerSummary.pushLimitFailureCount = pushLimitFails;
    if (footerSummary.totalEvents == 0) {
        // Zero events is the generator's free rejection filter regardless of why the simulator
        // stopped (tick budget vs. a same-position "cycle") - nothing worth training on happened.
        footerSummary.terminationReason = TERM_NOTHING_HAPPENED;
    }

    // Write every section in footer order, tracking byte offsets as we go.
    SimLogFooter footer;
    auto writeVec = [&](const void* data, std::size_t count, std::size_t recSize) -> std::uint64_t {
        std::uint64_t off = static_cast<std::uint64_t>(out_.tellp());
        if (count > 0) {
            out_.write(reinterpret_cast<const char*>(data),
                       static_cast<std::streamsize>(count * recSize));
        }
        return off;
    };

    writeVec(buffer_.data(), buffer_.size(), sizeof(SimEvent));  // events start at offset 0
    footer.eventCount = buffer_.size();
    footer.blockIndexOffset = writeVec(blockIndex_.data(), blockIndex_.size(), sizeof(BlockIndexEntry));
    footer.blockCount = static_cast<std::uint32_t>(blockIndex_.size());
    footer.pushGroupOffset = writeVec(pushGroups_.data(), pushGroups_.size(), sizeof(PushGroupRecord));
    footer.pushGroupCount = static_cast<std::uint32_t>(pushGroups_.size());
    footer.pushMemberOffset = writeVec(pushMembers_.data(), pushMembers_.size(), sizeof(std::uint64_t));
    footer.pushMemberCount = static_cast<std::uint32_t>(pushMembers_.size());
    footer.initialOffset = writeVec(initial_.data(), initial_.size(), sizeof(InitialBlockState));
    footer.initialCount = static_cast<std::uint32_t>(initial_.size());
    footer.componentOffset = writeVec(components_.data(), components_.size(), sizeof(ComponentRecord));
    footer.componentCount = static_cast<std::uint32_t>(components_.size());
    footer.componentMemberOffset =
        writeVec(componentMembers_.data(), componentMembers_.size(), sizeof(std::uint64_t));
    footer.componentMemberCount = static_cast<std::uint32_t>(componentMembers_.size());
    footer.summaryOffset = static_cast<std::uint64_t>(out_.tellp());
    out_.write(reinterpret_cast<const char*>(&footerSummary), sizeof(RunSummary));
    footer.wouldPowerOffset = writeVec(wouldPower_.data(), wouldPower_.size(), sizeof(WouldPowerEdge));
    footer.wouldPowerCount = static_cast<std::uint32_t>(wouldPower_.size());
    footer.staticPushMemberOffset =
        writeVec(staticPushMembers_.data(), staticPushMembers_.size(), sizeof(std::uint64_t));
    footer.staticPushMemberCount = static_cast<std::uint32_t>(staticPushMembers_.size());
    footer.staticPushGroupOffset =
        writeVec(staticPushGroups_.data(), staticPushGroups_.size(), sizeof(PushGroupRecord));
    footer.staticPushGroupCount = static_cast<std::uint32_t>(staticPushGroups_.size());

    out_.write(reinterpret_cast<const char*>(&footer), sizeof(footer));
    out_.close();

    buffer_.clear();
    blockIndex_.clear();
    pushGroups_.clear();
    pushMembers_.clear();
    initial_.clear();
    components_.clear();
    componentMembers_.clear();
    wouldPower_.clear();
    staticPushGroups_.clear();
    staticPushMembers_.clear();
    summary_ = RunSummary{};
    hasSummary_ = false;
    indexOf_.clear();
    pendingQueue_.clear();
}

bool SimEventLog::selfTest() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "mcp1122_simdata_selftest.bin").string();

    const std::uint64_t kA = packPos(BlockPos{1, 2, 3});
    const std::uint64_t kB = packPos(BlockPos{4, 5, 6});

    {
        SimEventLog log;
        log.open(path);
        if (!log.enabled()) {
            std::cerr << "selftest: could not open " << path << '\n';
            return false;
        }
        log.registerOriginalBlock(kA, 165);
        log.registerOriginalBlock(kB, 33);

        // Interleave the two blocks' events in emission order so grouping actually has to reorder.
        SimEvent e1;
        e1.blockKey = kA; e1.kind = BlockPushed; e1.activationSubtick = log.nextOrder();
        e1.pushGroupId = 10; e1.activationTick = 5;
        log.push(e1);

        SimEvent e2;
        e2.blockKey = kB; e2.kind = ObserverFired; e2.activationSubtick = log.nextOrder();
        e2.activationTick = 5;
        log.push(e2);

        SimEvent e3;
        e3.blockKey = kA; e3.kind = BlockPushed; e3.activationSubtick = log.nextOrder();
        e3.pushGroupId = 11; e3.activationTick = 18;
        log.push(e3);

        SimEvent e4;
        e4.blockKey = kB; e4.kind = ObserverActivated; e4.activationSubtick = log.nextOrder();
        e4.activationTick = 18;
        log.push(e4);

        // One failed push group with membership (the informative case).
        log.addPushGroup(/*globalSeq*/ 99, /*pistonKey*/ kA, /*tick*/ 18, /*subtick*/ 4,
                         /*direction*/ 0, /*succeeded*/ false, FAIL_PUSH_LIMIT_EXCEEDED,
                         /*attemptedCount*/ 13, std::vector<std::uint64_t>{kA, kB});

        // One initial-state record and one component record referencing both blocks.
        InitialBlockState ib;
        ib.stableKey = kA; ib.x = 1; ib.y = 2; ib.z = 3; ib.blockTypeId = 165;
        ib.movabilityClass = MOVABILITY_MOVABLE; ib.stickinessClass = STICKINESS_ALL; ib.componentId = 0;
        log.setInitialState(std::vector<InitialBlockState>{ib});

        ComponentRecord cr;
        cr.componentId = 0; cr.memberCount = 2; cr.memberOffset = 0;
        log.setComponents(std::vector<ComponentRecord>{cr}, std::vector<std::uint64_t>{kA, kB});

        RunSummary rs;
        rs.terminationReason = TERM_CYCLE_DETECTED; rs.validCycle = 1; rs.period = 13;
        rs.netShift[0] = 1; rs.blockCount = 2;
        log.setSummary(rs);

        // One would-power edge (kA statically powers kB, via QC) and one static push-group preview
        // (a piston that never actually fires during a run still gets a group-size feature).
        WouldPowerEdge wp;
        wp.sourceKey = kA; wp.pistonKey = kB; wp.viaQC = 1;
        log.setWouldPower(std::vector<WouldPowerEdge>{wp});

        PushGroupRecord staticPreview;
        staticPreview.pistonKey = kB; staticPreview.succeeded = 1; staticPreview.attemptedCount = 2;
        staticPreview.memberCount = 2; staticPreview.memberOffset = 0;
        log.setStaticPushPreview(std::vector<PushGroupRecord>{staticPreview}, std::vector<std::uint64_t>{kA, kB});

        log.close();
    }

    // Inline reader: footer from EOF, block index, then per-block contiguous runs.
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "selftest: could not reopen " << path << '\n';
        return false;
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    if (size < static_cast<std::streamoff>(sizeof(SimLogFooter))) {
        std::cerr << "selftest: file too small\n";
        return false;
    }
    SimLogFooter footer;
    in.seekg(size - static_cast<std::streamoff>(sizeof(SimLogFooter)), std::ios::beg);
    in.read(reinterpret_cast<char*>(&footer), sizeof(footer));
    if (std::memcmp(footer.magic, "SDL5", 4) != 0 || footer.eventRecSize != sizeof(SimEvent)) {
        std::cerr << "selftest: bad footer\n";
        return false;
    }

    std::vector<BlockIndexEntry> index(footer.blockCount);
    in.seekg(static_cast<std::streamoff>(footer.blockIndexOffset), std::ios::beg);
    if (footer.blockCount > 0) {
        in.read(reinterpret_cast<char*>(index.data()),
                static_cast<std::streamsize>(footer.blockCount * sizeof(BlockIndexEntry)));
    }

    auto readRun = [&](std::uint64_t key, std::vector<SimEvent>& out) -> bool {
        for (const BlockIndexEntry& e : index) {
            if (e.originalKey == key) {
                out.resize(e.eventCount);
                if (e.eventCount > 0) {
                    in.seekg(static_cast<std::streamoff>(e.firstEventIdx) * sizeof(SimEvent), std::ios::beg);
                    in.read(reinterpret_cast<char*>(out.data()),
                            static_cast<std::streamsize>(e.eventCount * sizeof(SimEvent)));
                }
                return true;
            }
        }
        return false;
    };

    std::vector<SimEvent> runA, runB;
    if (!readRun(kA, runA) || !readRun(kB, runB)) {
        std::cerr << "selftest: block missing from index\n";
        return false;
    }
    bool ok = runA.size() == 2 && runB.size() == 2
        && runA[0].pushGroupId == 10 && runA[1].pushGroupId == 11
        && runA[0].activationSubtick < runA[1].activationSubtick
        && runB[0].kind == ObserverFired && runB[1].kind == ObserverActivated
        && runB[0].activationSubtick < runB[1].activationSubtick;

    // Push groups: one failed record with full membership.
    std::vector<PushGroupRecord> groups(footer.pushGroupCount);
    if (footer.pushGroupCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.pushGroupOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(groups.data()),
                static_cast<std::streamsize>(footer.pushGroupCount * sizeof(PushGroupRecord)));
    }
    std::vector<std::uint64_t> pushMembers(footer.pushMemberCount);
    if (footer.pushMemberCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.pushMemberOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(pushMembers.data()),
                static_cast<std::streamsize>(footer.pushMemberCount * sizeof(std::uint64_t)));
    }
    ok = ok && groups.size() == 1 && groups[0].succeeded == 0
        && groups[0].failureReason == FAIL_PUSH_LIMIT_EXCEEDED && groups[0].attemptedCount == 13
        && groups[0].memberCount == 2 && pushMembers.size() == 2
        && pushMembers[groups[0].memberOffset] == kA && pushMembers[groups[0].memberOffset + 1] == kB;

    // Initial state + components.
    std::vector<InitialBlockState> initial(footer.initialCount);
    if (footer.initialCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.initialOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(initial.data()),
                static_cast<std::streamsize>(footer.initialCount * sizeof(InitialBlockState)));
    }
    std::vector<ComponentRecord> comps(footer.componentCount);
    if (footer.componentCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.componentOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(comps.data()),
                static_cast<std::streamsize>(footer.componentCount * sizeof(ComponentRecord)));
    }
    ok = ok && initial.size() == 1 && initial[0].stableKey == kA && initial[0].blockTypeId == 165
        && comps.size() == 1 && comps[0].memberCount == 2;

    // RunSummary: measured fields must have been filled in by close(), logical fields preserved.
    RunSummary rs;
    in.seekg(static_cast<std::streamoff>(footer.summaryOffset), std::ios::beg);
    in.read(reinterpret_cast<char*>(&rs), sizeof(rs));
    ok = ok && rs.terminationReason == TERM_CYCLE_DETECTED && rs.validCycle == 1 && rs.period == 13
        && rs.totalEvents == 4 && rs.distinctBlocksWithEvents == 2
        && rs.maxPushGroupSize == 13 && rs.pushLimitFailureCount == 1;

    // Would-power edge (static, SDL4).
    std::vector<WouldPowerEdge> wouldPower(footer.wouldPowerCount);
    if (footer.wouldPowerCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.wouldPowerOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(wouldPower.data()),
                static_cast<std::streamsize>(footer.wouldPowerCount * sizeof(WouldPowerEdge)));
    }
    ok = ok && wouldPower.size() == 1 && wouldPower[0].sourceKey == kA && wouldPower[0].pistonKey == kB
        && wouldPower[0].viaQC == 1;

    // Static push-group preview (SDL4) - a separate section + member array from the dynamic ones.
    std::vector<PushGroupRecord> staticGroups(footer.staticPushGroupCount);
    if (footer.staticPushGroupCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.staticPushGroupOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(staticGroups.data()),
                static_cast<std::streamsize>(footer.staticPushGroupCount * sizeof(PushGroupRecord)));
    }
    std::vector<std::uint64_t> staticMembers(footer.staticPushMemberCount);
    if (footer.staticPushMemberCount > 0) {
        in.seekg(static_cast<std::streamoff>(footer.staticPushMemberOffset), std::ios::beg);
        in.read(reinterpret_cast<char*>(staticMembers.data()),
                static_cast<std::streamsize>(footer.staticPushMemberCount * sizeof(std::uint64_t)));
    }
    ok = ok && staticGroups.size() == 1 && staticGroups[0].pistonKey == kB
        && staticGroups[0].succeeded == 1 && staticGroups[0].attemptedCount == 2
        && staticMembers.size() == 2 && staticMembers[0] == kA && staticMembers[1] == kB;

    if (!ok) {
        std::cerr << "selftest: SDL4 sections did not round-trip\n";
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (!ok) {
        std::cerr << "selftest: reconstructed runs did not match\n";
        return false;
    }
    return true;
}

} // namespace mcp1122
