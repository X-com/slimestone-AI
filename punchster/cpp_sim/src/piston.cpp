#include "piston.h"

#include "block_registry.h"

namespace mcp1122 {

namespace {

bool samePos(BlockPos a, BlockPos b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

} // namespace

bool canPush(std::uint32_t state, const World&, BlockPos pos, const Facing& facing,
             bool destroyBlocks, const Facing& pushFacing) {
    int id = blockId(state);
    if (id == BLOCK_OBSIDIAN) {
        return false;
    }
    if (pos.y < 0 || (facing.index == 0 && pos.y == 0)) {
        return false;
    }
    if (pos.y > 255 || (facing.index == 1 && pos.y == 255)) {
        return false;
    }

    const BlockData& data = blockData(id);
    if (!isPistonBlock(id)) {
        if (data.hardness == -1.0f) {
            return false;
        }
        switch (data.pushReaction) {
            case PushReaction::Block:
                return false;
            case PushReaction::Destroy:
                return destroyBlocks;
            case PushReaction::PushOnly:
                return facing.index == pushFacing.index;
            case PushReaction::Normal:
                break;
        }
    } else if (metaBit(state, 3)) {
        return false;
    }

    return !data.hasTileEntity;
}

bool pistonDoMove(World& world, BlockPos pos, const Facing& direction, bool extending, bool sticky) {
    if (!extending) {
        world.removeBlock(offset(pos, direction));
    }

    PistonStructureHelper helper(world, pos, direction, extending);
    if (!helper.canMove()) {
        return false;
    }

    std::array<std::uint32_t, 12> movedStates{};
    for (int i = 0; i < helper.moveCount(); ++i) {
        movedStates[static_cast<std::size_t>(i)] = world.getBlock(helper.moveAt(i));
    }

    const Facing& moveFacing = extending ? direction : opposite(direction);

    for (int i = helper.destroyCount() - 1; i >= 0; --i) {
        world.removeBlock(helper.destroyAt(i));
    }

    for (int i = helper.moveCount() - 1; i >= 0; --i) {
        BlockPos source = helper.moveAt(i);
        BlockPos target = offset(source, moveFacing);
        world.removeBlock(source);
        world.setBlock(target, movedStates[static_cast<std::size_t>(i)]);
    }

    if (extending) {
        int headMeta = direction.index | (sticky ? 8 : 0);
        world.setBlock(offset(pos, direction), makeState(BLOCK_PISTON_HEAD, headMeta));
        world.setBlock(pos, setMetaBit(world.getBlock(pos), 3, true));
    } else {
        world.setBlock(pos, setMetaBit(world.getBlock(pos), 3, false));
    }

    return true;
}

PistonStructureHelper::PistonStructureHelper(const World& world, BlockPos pistonPos,
                                             const Facing& pistonFacing, bool extending)
    : world_(world),
      pistonPos_(pistonPos),
      blockToMove_(extending ? offset(pistonPos, pistonFacing) : offset(pistonPos, pistonFacing, 2)),
      moveDirection_(extending ? pistonFacing : opposite(pistonFacing)) {
}

bool PistonStructureHelper::canMove() {
    moveCount_ = 0;
    destroyCount_ = 0;

    std::uint32_t state = world_.getBlock(blockToMove_);
    if (!canPush(state, world_, blockToMove_, moveDirection_, false, moveDirection_)) {
        if (blockData(blockId(state)).pushReaction == PushReaction::Destroy) {
            toDestroy_[static_cast<std::size_t>(destroyCount_++)] = blockToMove_;
            return true;
        }
        return false;
    }

    if (!addBlockLine(blockToMove_, moveDirection_)) {
        return false;
    }

    for (int i = 0; i < moveCount_; ++i) {
        BlockPos pos = toMove_[static_cast<std::size_t>(i)];
        if (blockId(world_.getBlock(pos)) == BLOCK_SLIME && !addBranchingBlocks(pos)) {
            return false;
        }
    }
    return true;
}

bool PistonStructureHelper::addBlockLine(BlockPos origin, const Facing& branchFacing) {
    std::uint32_t state = world_.getBlock(origin);
    int id = blockId(state);

    if (id == BLOCK_AIR) {
        return true;
    }
    if (!canPush(state, world_, origin, moveDirection_, false, branchFacing)) {
        return true;
    }
    if (samePos(origin, pistonPos_)) {
        return true;
    }
    if (indexOfToMove(origin) >= 0) {
        return true;
    }

    int i = 1;
    if (i + moveCount_ > 12) {
        return false;
    }

    const Facing& opp = opposite(moveDirection_);
    while (id == BLOCK_SLIME) {
        BlockPos scan = offset(origin, opp, i);
        state = world_.getBlock(scan);
        id = blockId(state);

        if (id == BLOCK_AIR
                || !canPush(state, world_, scan, moveDirection_, false, opp)
                || samePos(scan, pistonPos_)) {
            break;
        }

        ++i;
        if (i + moveCount_ > 12) {
            return false;
        }
    }

    int movedInLine = 0;
    for (int j = i - 1; j >= 0; --j) {
        toMove_[static_cast<std::size_t>(moveCount_++)] = offset(origin, opp, j);
        ++movedInLine;
    }

    int forwardDistance = 1;
    while (true) {
        BlockPos forward = offset(origin, moveDirection_, forwardDistance);
        int collision = indexOfToMove(forward);
        if (collision > -1) {
            reorderListAtCollision(movedInLine, collision);
            for (int l = 0; l <= collision + movedInLine; ++l) {
                BlockPos moved = toMove_[static_cast<std::size_t>(l)];
                if (blockId(world_.getBlock(moved)) == BLOCK_SLIME && !addBranchingBlocks(moved)) {
                    return false;
                }
            }
            return true;
        }

        state = world_.getBlock(forward);
        id = blockId(state);
        if (id == BLOCK_AIR) {
            return true;
        }

        if (!canPush(state, world_, forward, moveDirection_, true, moveDirection_)
                || samePos(forward, pistonPos_)) {
            return false;
        }

        if (blockData(id).pushReaction == PushReaction::Destroy) {
            toDestroy_[static_cast<std::size_t>(destroyCount_++)] = forward;
            return true;
        }

        if (moveCount_ >= 12) {
            return false;
        }

        toMove_[static_cast<std::size_t>(moveCount_++)] = forward;
        ++movedInLine;
        ++forwardDistance;
    }
}

bool PistonStructureHelper::addBranchingBlocks(BlockPos fromPos) {
    for (const Facing& facing : facings()) {
        if (facing.axis != moveDirection_.axis) {
            if (!addBlockLine(offset(fromPos, facing), facing)) {
                return false;
            }
        }
    }
    return true;
}

void PistonStructureHelper::reorderListAtCollision(int movedInLine, int collisionIndex) {
    int size = moveCount_;
    int out = 0;
    for (int i = 0; i < collisionIndex; ++i) {
        reorderTmp_[static_cast<std::size_t>(out++)] = toMove_[static_cast<std::size_t>(i)];
    }
    for (int i = size - movedInLine; i < size; ++i) {
        reorderTmp_[static_cast<std::size_t>(out++)] = toMove_[static_cast<std::size_t>(i)];
    }
    for (int i = collisionIndex; i < size - movedInLine; ++i) {
        reorderTmp_[static_cast<std::size_t>(out++)] = toMove_[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < out; ++i) {
        toMove_[static_cast<std::size_t>(i)] = reorderTmp_[static_cast<std::size_t>(i)];
    }
}

int PistonStructureHelper::indexOfToMove(BlockPos pos) const {
    for (int i = 0; i < moveCount_; ++i) {
        if (samePos(toMove_[static_cast<std::size_t>(i)], pos)) {
            return i;
        }
    }
    return -1;
}

} // namespace mcp1122
