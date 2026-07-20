#include "json_stream.h"

#include <cstdlib>
#include <stdexcept>

namespace mcp1122 {

namespace {

// Every candidate is raised by this many blocks on load so the simulated world never sees a
// negative y - the real-Minecraft (mcp1122) engine's chunk storage is a hard 0-255 array and
// crashes on negative y, so all three engines apply this identical offset for consistency.
// Nothing un-shifts it back: results (working/period/shift/ticks) are offset-invariant since
// shift is a delta between two anchors, and trace logs simply show the raised coordinates.
//
// The simulator itself enforces the matching upper bound (Simulator::setBlockState/
// scheduleUpdate silently no-op for y < 0 or y >= 256, mirroring the real engine's array
// bounds) - so a candidate whose own real height already exceeds roughly 191 blocks (255 - 64)
// gets its upper portion silently dropped once shifted, with no error, just wrong results. A
// candidate that's already entirely non-negative doesn't need the offset at all;
// MCP1122_CPP_NO_Y_OFFSET=1 skips it for exactly that case (verification/debugging only - a
// candidate that genuinely has negative y still needs the default offset).
int candidateYOffset() {
    if (const char* env = std::getenv("MCP1122_CPP_NO_Y_OFFSET")) {
        if (std::atoi(env) != 0) {
            return 0;
        }
    }
    return 64;
}

template <typename T>
bool readRaw(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return in.gcount() == static_cast<std::streamsize>(sizeof(T));
}

} // namespace

bool readCandidateCompact(std::istream& in, Candidate& out) {
    std::int32_t id = 0;
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    if (in.gcount() == 0) {
        return false; // clean EOF between records - not an error, just "no more candidates"
    }
    if (in.gcount() != static_cast<std::streamsize>(sizeof(id))) {
        throw std::runtime_error("truncated compact candidate: id");
    }

    std::int32_t tx = 0, ty = 0, tz = 0;
    std::uint32_t blockCount = 0;
    if (!readRaw(in, tx) || !readRaw(in, ty) || !readRaw(in, tz) || !readRaw(in, blockCount)) {
        throw std::runtime_error("truncated compact candidate: header");
    }

    int yOffset = candidateYOffset();
    out = Candidate{};
    out.id = id;
    out.trigger = BlockPos{tx, ty + yOffset, tz};
    out.blocks.reserve(blockCount);

    for (std::uint32_t i = 0; i < blockCount; ++i) {
        std::int32_t x = 0, y = 0, z = 0;
        std::uint32_t state = 0;
        if (!readRaw(in, x) || !readRaw(in, y) || !readRaw(in, z) || !readRaw(in, state)) {
            throw std::runtime_error("truncated compact candidate: block");
        }
        BlockEntry block;
        block.x = x;
        block.y = y + yOffset;
        block.z = z;
        block.state = state;
        out.blocks.push_back(block);
    }

    return true;
}

std::string quoteJson(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

} // namespace mcp1122
