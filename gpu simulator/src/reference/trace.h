#pragma once

#include "packed_pos.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace mcp1122gpu {

using mcp1122::BlockPos;

// Same line format as cpp extract's mcp1122::Trace ("worldTime tag x,y,z [blockId[:meta]]
// [a,b]"), so output diffs directly against cpp extract traces with compare_traces.py. Kept
// as a separate small class instead of reusing mcp1122::Trace because that class takes a
// mcp1122::World& (for world.time) - coupling it to this world type would mean either
// reaching into cpp extract's PosStateMap-backed World or duplicating cpp extract's World
// struct just to satisfy a signature. Taking worldTime directly avoids both.
class Trace {
public:
    void open(const std::string& path);
    bool enabled() const { return out_.is_open(); }
    void log(std::int64_t worldTime, const char* tag, const BlockPos* pos, int a = 0, int b = 0);
    void logBlock(std::int64_t worldTime, const char* tag, const BlockPos* pos, int blockId, int a = 0, int b = 0);
    void logState(std::int64_t worldTime, const char* tag, const BlockPos* pos, std::uint32_t state, int a = 0, int b = 0);
    // Matches mcp1122::Trace::logHash's format exactly - diffed against cpp extract's
    // "b.initkey" line to verify the ported state-key hash.
    void logHash(std::int64_t worldTime, const char* tag, std::uint64_t count, std::uint64_t a, std::uint64_t b);

private:
    std::ofstream out_;
    void writePrefix(std::int64_t worldTime, const char* tag, const BlockPos* pos);
    void writePayload(int a, int b);
};

} // namespace mcp1122gpu
