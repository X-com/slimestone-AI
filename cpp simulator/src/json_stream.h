#pragma once

#include "packed_pos.h"

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace mcp1122 {

struct BlockEntry {
    int x = 0;
    int y = 0;
    int z = 0;
    std::uint32_t state = 0;
};

struct Candidate {
    std::int64_t id = 0;
    BlockPos trigger;
    std::vector<BlockEntry> blocks;
};

// Reads one candidate from the compact binary format (see genetic_ml.compact_format on the
// Python side for the authoritative layout - both sides must stay in sync):
//   int32 id | int32 trigger_x,trigger_y,trigger_z | uint32 block_count
//   block_count x { int32 x,y,z | uint32 state }
// Used for both the stdin stream (the live GA) and file mode (named fixture files) - the same
// format either way. Returns false on a clean EOF between records; throws std::runtime_error
// if the stream is cut off partway through a record.
bool readCandidateCompact(std::istream& in, Candidate& out);

std::string quoteJson(const std::string& value);

} // namespace mcp1122
