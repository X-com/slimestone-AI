#pragma once

// Device-compatible port of ../reference/facing.h (which itself mirrors cpp extract's). No
// behavior change - the original was already exception-free and allocation-free, only the
// `static const` function-local table (fine on host, not guaranteed visible/safe the same way
// across CUDA translation) becomes a plain device-visible constant array.

#include "gpu_common.cuh"
#include "packed_pos.cuh"

namespace mcp1122gpu {

enum class Axis { X, Y, Z };

struct Facing {
    int index;
    int opposite;
    Axis axis;
    int dx;
    int dy;
    int dz;
};

// DOWN, UP, NORTH, SOUTH, WEST, EAST - same order/indices as cpp extract's facings().
MCPGPU_HD inline const Facing& facingByIndex(int index) {
    static const Facing kFacings[6] = {
        {0, 1, Axis::Y, 0, -1, 0},
        {1, 0, Axis::Y, 0, 1, 0},
        {2, 3, Axis::Z, 0, 0, -1},
        {3, 2, Axis::Z, 0, 0, 1},
        {4, 5, Axis::X, -1, 0, 0},
        {5, 4, Axis::X, 1, 0, 0},
    };
    int wrapped = index % 6;
    if (wrapped < 0) wrapped = -wrapped;
    return kFacings[wrapped];
}

MCPGPU_HD inline const Facing& facings(int i) { return facingByIndex(i); }

MCPGPU_HD inline const Facing& opposite(const Facing& facing) { return facingByIndex(facing.opposite); }

MCPGPU_HD inline BlockPos offset(BlockPos pos, const Facing& facing, int distance = 1) {
    return BlockPos{pos.x + facing.dx * distance, pos.y + facing.dy * distance, pos.z + facing.dz * distance};
}

} // namespace mcp1122gpu
