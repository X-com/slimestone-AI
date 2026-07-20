#pragma once

#include "facing.h"
#include "world.h"

#include <array>
#include <cstdint>

namespace mcp1122 {

bool canPush(std::uint32_t state, const World& world, BlockPos pos, const Facing& facing,
             bool destroyBlocks, const Facing& pushFacing);

bool pistonDoMove(World& world, BlockPos pos, const Facing& direction, bool extending, bool sticky);

class PistonStructureHelper {
public:
    PistonStructureHelper(const World& world, BlockPos pistonPos, const Facing& pistonFacing, bool extending);

    bool canMove();
    int moveCount() const { return moveCount_; }
    int destroyCount() const { return destroyCount_; }
    BlockPos moveAt(int index) const { return toMove_[static_cast<std::size_t>(index)]; }
    BlockPos destroyAt(int index) const { return toDestroy_[static_cast<std::size_t>(index)]; }

private:
    const World& world_;
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

} // namespace mcp1122
