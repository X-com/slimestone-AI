#pragma once

// Device-compatible port of ../reference (and cpp extract)'s packed_pos.h. Only change:
// packSigned21 can't throw on device. It masks to 21 bits instead of throwing - unreachable in
// practice, not just "shouldn't happen": every position this is called on already passed
// FixedWorld::inRange (extent=128 around origin), so it's always far inside the +-1048576
// 21-bit range those bounds are nested in. No caller needs to observe an overflow that only a
// future FixedWorld::kExtent bump far past 2^20 could ever trigger.

#include "gpu_common.cuh"

#include <cstdint>

namespace mcp1122gpu {

struct BlockPos {
    int x = 0;
    int y = 0;
    int z = 0;
};

MCPGPU_HD inline std::uint64_t packSigned21(int value) {
    return static_cast<std::uint64_t>(value & 0x1fffff);
}

MCPGPU_HD inline std::uint64_t packPos(int x, int y, int z) {
    return packSigned21(x) | (packSigned21(y) << 21) | (packSigned21(z) << 42);
}

MCPGPU_HD inline std::uint64_t packPos(BlockPos pos) {
    return packPos(pos.x, pos.y, pos.z);
}

} // namespace mcp1122gpu
