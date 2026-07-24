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
    indexOf_.clear();
    pendingQueue_.clear();
    nextOrder_ = 0;
    nextPushGroupId_ = 0;
    out_.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
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

    if (!buffer_.empty()) {
        out_.write(reinterpret_cast<const char*>(buffer_.data()),
                   static_cast<std::streamsize>(buffer_.size() * sizeof(SimEvent)));
    }
    SimLogFooter footer;
    footer.eventCount = buffer_.size();
    footer.blockIndexOffset = buffer_.size() * sizeof(SimEvent);
    footer.blockCount = static_cast<std::uint32_t>(blockIndex_.size());
    if (!blockIndex_.empty()) {
        out_.write(reinterpret_cast<const char*>(blockIndex_.data()),
                   static_cast<std::streamsize>(blockIndex_.size() * sizeof(BlockIndexEntry)));
    }
    out_.write(reinterpret_cast<const char*>(&footer), sizeof(footer));
    out_.close();

    buffer_.clear();
    blockIndex_.clear();
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

        log.close();
    }

    // Inline reader: footer from EOF-48, block index, then per-block contiguous runs.
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
    if (std::memcmp(footer.magic, "SDL2", 4) != 0 || footer.eventRecSize != sizeof(SimEvent)) {
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

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (!ok) {
        std::cerr << "selftest: reconstructed runs did not match\n";
        return false;
    }
    return true;
}

} // namespace mcp1122
