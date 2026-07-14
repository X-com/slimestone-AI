#pragma once

#include "packed_pos.h"

#include <array>

namespace mcp1122 {

enum class Axis {
    X,
    Y,
    Z
};

struct Facing {
    int index;
    int opposite;
    Axis axis;
    int dx;
    int dy;
    int dz;
};

inline const std::array<Facing, 6>& facings() {
    static const std::array<Facing, 6> values{{
        {0, 1, Axis::Y,  0, -1,  0}, // DOWN
        {1, 0, Axis::Y,  0,  1,  0}, // UP
        {2, 3, Axis::Z,  0,  0, -1}, // NORTH
        {3, 2, Axis::Z,  0,  0,  1}, // SOUTH
        {4, 5, Axis::X, -1,  0,  0}, // WEST
        {5, 4, Axis::X,  1,  0,  0}, // EAST
    }};
    return values;
}

inline const Facing& facingByIndex(int index) {
    int wrapped = index % 6;
    if (wrapped < 0) {
        wrapped = -wrapped;
    }
    return facings()[static_cast<std::size_t>(wrapped)];
}

inline const Facing& opposite(const Facing& facing) {
    return facings()[static_cast<std::size_t>(facing.opposite)];
}

inline BlockPos offset(BlockPos pos, const Facing& facing, int distance = 1) {
    return BlockPos{
        pos.x + facing.dx * distance,
        pos.y + facing.dy * distance,
        pos.z + facing.dz * distance
    };
}

} // namespace mcp1122
