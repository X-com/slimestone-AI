#include "trace.h"

#include "block_registry.h"

#include <filesystem>

namespace mcp1122 {

void Trace::open(const std::string& path) {
    // open() is called once per candidate (see tracePathForCandidate() in main.cpp) so each
    // flying machine gets its own file instead of every candidate scrambling into one shared
    // stream. std::ofstream::open() on an already-open stream silently fails and leaves the
    // previous file open, so the old file must be closed first.
    if (out_.is_open()) {
        out_.close();
    }
    std::filesystem::path tracePath(path);
    if (tracePath.has_parent_path()) {
        std::filesystem::create_directories(tracePath.parent_path());
    }
    out_.open(path.c_str(), std::ios::out | std::ios::trunc);
}

void Trace::log(const World& world, const char* tag, const BlockPos* pos, int a, int b) {
    if (!out_.is_open()) {
        return;
    }

    writePrefix(world, tag, pos);
    writePayload(a, b);
    out_ << '\n';
}

void Trace::logBlock(const World& world, const char* tag, const BlockPos* pos, int blockId, int a, int b) {
    if (!out_.is_open()) {
        return;
    }
    writePrefix(world, tag, pos);
    out_ << ' ' << blockId;
    writePayload(a, b);
    out_ << '\n';
}

void Trace::logState(const World& world, const char* tag, const BlockPos* pos, std::uint32_t state, int a, int b) {
    if (!out_.is_open()) {
        return;
    }
    writePrefix(world, tag, pos);
    out_ << ' ' << blockId(state);
    int meta = blockMeta(state);
    out_ << ':' << meta;
    writePayload(a, b);
    out_ << '\n';
}

void Trace::writePrefix(const World& world, const char* tag, const BlockPos* pos) {
    out_ << world.time << ' ' << tag << ' ';
    if (pos == nullptr) {
        out_ << "?,?,";
    } else {
        out_ << pos->x << ',' << pos->y << ',' << pos->z;
    }
}

void Trace::writePayload(int a, int b) {
    if (a != 0 || b != 0) {
        out_ << ' ' << a << ',' << b;
    }
}

} // namespace mcp1122
