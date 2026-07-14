#pragma once

#include <cstdint>
#include <stdexcept>

namespace mcp1122 {

struct BlockPos {
    int x = 0;
    int y = 0;
    int z = 0;
};

inline std::uint64_t packSigned21(int value) {
    if (value < -1048576 || value > 1048575) {
        throw std::out_of_range("position coordinate is outside signed 21-bit range");
    }
    return static_cast<std::uint64_t>(value & 0x1fffff);
}

inline int unpackSigned21(std::uint64_t value) {
    int out = static_cast<int>(value & 0x1fffff);
    if ((out & 0x100000) != 0) {
        out |= ~0x1fffff;
    }
    return out;
}

inline std::uint64_t packPos(int x, int y, int z) {
    return packSigned21(x) | (packSigned21(y) << 21) | (packSigned21(z) << 42);
}

inline std::uint64_t packPos(BlockPos pos) {
    return packPos(pos.x, pos.y, pos.z);
}

inline BlockPos unpackPos(std::uint64_t key) {
    return BlockPos{
        unpackSigned21(key),
        unpackSigned21(key >> 21),
        unpackSigned21(key >> 42)
    };
}

} // namespace mcp1122
