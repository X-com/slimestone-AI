#pragma once

#include "packed_pos.h"
#include "world.h"

#include <fstream>
#include <string>

namespace mcp1122 {

class Trace {
public:
    void open(const std::string& path);
    bool enabled() const { return out_.is_open(); }
    void log(const World& world, const char* tag, const BlockPos* pos, int a = 0, int b = 0);
    void logBlock(const World& world, const char* tag, const BlockPos* pos, int blockId, int a = 0, int b = 0);
    void logState(const World& world, const char* tag, const BlockPos* pos, std::uint32_t state, int a = 0, int b = 0);
    // For dumping wide hash words (e.g. StateKey) that would lose precision through the int a/b
    // payload the other log*() methods use - no BlockPos, since a state-key hash isn't tied to
    // one position.
    void logHash(const World& world, const char* tag, std::uint64_t a, std::uint64_t b, std::uint64_t c = 0);

private:
    std::ofstream out_;
    void writePrefix(const World& world, const char* tag, const BlockPos* pos);
    void writePayload(int a, int b);
};

} // namespace mcp1122
