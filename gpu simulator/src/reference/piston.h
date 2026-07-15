#pragma once

// Ported from ../../cpp extract/src/piston.h, adapted to FixedWorld instead of
// mcp1122::World's hash map. The algorithm itself (canPush, doMove, PistonStructureHelper)
// is already fixed-array-shaped in cpp extract - std::array<BlockPos,12>/<BlockPos,4> bounded
// by the vanilla piston push limit - so this port changes only the World type, not the logic.

#include "facing.h"
#include "world.h"

#include <array>
#include <cstdint>

namespace mcp1122gpu {

using mcp1122::Facing;

bool canPush(std::uint32_t state, const FixedWorld& world, BlockPos pos, const Facing& facing,
             bool destroyBlocks, const Facing& pushFacing);

class PistonStructureHelper {
public:
    PistonStructureHelper(const FixedWorld& world, BlockPos pistonPos, const Facing& pistonFacing, bool extending);

    bool canMove();
    int moveCount() const { return moveCount_; }
    int destroyCount() const { return destroyCount_; }
    BlockPos moveAt(int index) const { return toMove_[static_cast<std::size_t>(index)]; }
    BlockPos destroyAt(int index) const { return toDestroy_[static_cast<std::size_t>(index)]; }

private:
    const FixedWorld& world_;
    BlockPos pistonPos_;
    BlockPos blockToMove_;
    const Facing& moveDirection_;
    std::array<BlockPos, 12> toMove_{};
    int moveCount_ = 0;
    std::array<BlockPos, 4> toDestroy_{};
    int destroyCount_ = 0;
    std::array<BlockPos, 12> reorderTmp_{};

    bool addBlockLine(BlockPos origin, const Facing& branchFacing);
    bool addBranchingBlocks(BlockPos fromPos);
    void reorderListAtCollision(int movedInLine, int collisionIndex);
    int indexOfToMove(BlockPos pos) const;
};

} // namespace mcp1122gpu
