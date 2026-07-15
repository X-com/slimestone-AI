#include "trace.h"

#include "block_registry.h"

#include <filesystem>

namespace mcp1122gpu {

void Trace::open(const std::string& path) {
    if (out_.is_open()) {
        out_.close();
    }
    std::filesystem::path tracePath(path);
    if (tracePath.has_parent_path()) {
        std::filesystem::create_directories(tracePath.parent_path());
    }
    out_.open(path.c_str(), std::ios::out | std::ios::trunc);
}

void Trace::log(std::int64_t worldTime, const char* tag, const BlockPos* pos, int a, int b) {
    if (!out_.is_open()) {
        return;
    }
    writePrefix(worldTime, tag, pos);
    writePayload(a, b);
    out_ << '\n';
}

void Trace::logBlock(std::int64_t worldTime, const char* tag, const BlockPos* pos, int blockId, int a, int b) {
    if (!out_.is_open()) {
        return;
    }
    writePrefix(worldTime, tag, pos);
    out_ << ' ' << blockId;
    writePayload(a, b);
    out_ << '\n';
}

void Trace::logState(std::int64_t worldTime, const char* tag, const BlockPos* pos, std::uint32_t state, int a, int b) {
    if (!out_.is_open()) {
        return;
    }
    writePrefix(worldTime, tag, pos);
    out_ << ' ' << mcp1122::blockId(state);
    int meta = mcp1122::blockMeta(state);
    out_ << ':' << meta;
    writePayload(a, b);
    out_ << '\n';
}

void Trace::logHash(std::int64_t worldTime, const char* tag, std::uint64_t count, std::uint64_t a, std::uint64_t b) {
    if (!out_.is_open()) {
        return;
    }
    out_ << worldTime << ' ' << tag << ' ' << count << ' ' << a << ' ' << b << '\n';
}

void Trace::writePrefix(std::int64_t worldTime, const char* tag, const BlockPos* pos) {
    out_ << worldTime << ' ' << tag << ' ';
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

} // namespace mcp1122gpu
