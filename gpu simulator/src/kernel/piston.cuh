#pragma once

// Device-compatible port of ../reference/piston.h/.cpp, itself ported near-verbatim from cpp
// extract - the algorithm is already fixed-array-shaped there (std::array<BlockPos,12>/<4>
// bounded by the vanilla piston push limit), so this only changes World type and the
// `for (const Facing& f : facings())` range-for (no array-returning facings() on device) to an
// indexed loop over facingByIndex(0..5). No exceptions here in the original either - nothing to
// convert.

#include "block_registry.cuh"
#include "facing.cuh"
#include "gpu_common.cuh"
#include "world.cuh"

namespace mcp1122gpu {

MCPGPU_HD inline bool samePosPiston(BlockPos a, BlockPos b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

MCPGPU_HD inline bool canPush(std::uint32_t state, const FixedWorld&, BlockPos pos, const Facing& facing,
                               bool destroyBlocks, const Facing& pushFacing) {
    int id = blockId(state);
    if (id == BLOCK_OBSIDIAN) return false;
    if (pos.y < 0 || (facing.index == 0 && pos.y == 0)) return false;
    if (pos.y > 255 || (facing.index == 1 && pos.y == 255)) return false;

    const BlockData& data = blockData(id);
    if (!isPistonBlock(id)) {
        if (data.hardness == -1.0f) return false;
        switch (data.pushReaction) {
            case PushReaction::Block: return false;
            case PushReaction::Destroy: return destroyBlocks;
            case PushReaction::PushOnly: return facing.index == pushFacing.index;
            case PushReaction::Normal: break;
        }
    } else if (metaBit(state, 3)) {
        return false;
    }

    return !data.hasTileEntity;
}

class PistonStructureHelper {
public:
    MCPGPU_HD PistonStructureHelper(const FixedWorld& world, BlockPos pistonPos, const Facing& pistonFacing,
                                     bool extending)
        : world_(world),
          pistonPos_(pistonPos),
          blockToMove_(extending ? offset(pistonPos, pistonFacing) : offset(pistonPos, pistonFacing, 2)),
          moveDirection_(extending ? pistonFacing : opposite(pistonFacing)) {}

    MCPGPU_HD bool canMove() {
        moveCount_ = 0;
        destroyCount_ = 0;

        std::uint32_t state = world_.getBlock(blockToMove_);
        if (!canPush(state, world_, blockToMove_, moveDirection_, false, moveDirection_)) {
            if (blockData(blockId(state)).pushReaction == PushReaction::Destroy) {
                toDestroy_[destroyCount_++] = blockToMove_;
                return true;
            }
            return false;
        }

        if (!addBlockLine(blockToMove_, moveDirection_)) return false;

        for (int i = 0; i < moveCount_; ++i) {
            BlockPos pos = toMove_[i];
            if (blockId(world_.getBlock(pos)) == BLOCK_SLIME && !addBranchingBlocks(pos)) return false;
        }
        return true;
    }

    MCPGPU_HD int moveCount() const { return moveCount_; }
    MCPGPU_HD int destroyCount() const { return destroyCount_; }
    MCPGPU_HD BlockPos moveAt(int index) const { return toMove_[index]; }
    MCPGPU_HD BlockPos destroyAt(int index) const { return toDestroy_[index]; }

private:
    const FixedWorld& world_;
    BlockPos pistonPos_;
    BlockPos blockToMove_;
    const Facing& moveDirection_;
    BlockPos toMove_[12]{};
    int moveCount_ = 0;
    BlockPos toDestroy_[4]{};
    int destroyCount_ = 0;
    BlockPos reorderTmp_[12]{};

    MCPGPU_HD bool addBlockLine(BlockPos origin, const Facing& branchFacing) {
        std::uint32_t state = world_.getBlock(origin);
        int id = blockId(state);

        if (id == BLOCK_AIR) return true;
        if (!canPush(state, world_, origin, moveDirection_, false, branchFacing)) return true;
        if (samePosPiston(origin, pistonPos_)) return true;
        if (indexOfToMove(origin) >= 0) return true;

        int i = 1;
        if (i + moveCount_ > 12) return false;

        const Facing& opp = opposite(moveDirection_);
        while (id == BLOCK_SLIME) {
            BlockPos scan = offset(origin, opp, i);
            state = world_.getBlock(scan);
            id = blockId(state);

            if (id == BLOCK_AIR || !canPush(state, world_, scan, moveDirection_, false, opp) ||
                samePosPiston(scan, pistonPos_)) {
                break;
            }

            ++i;
            if (i + moveCount_ > 12) return false;
        }

        int movedInLine = 0;
        for (int j = i - 1; j >= 0; --j) {
            toMove_[moveCount_++] = offset(origin, opp, j);
            ++movedInLine;
        }

        int forwardDistance = 1;
        while (true) {
            BlockPos forward = offset(origin, moveDirection_, forwardDistance);
            int collision = indexOfToMove(forward);
            if (collision > -1) {
                reorderListAtCollision(movedInLine, collision);
                for (int l = 0; l <= collision + movedInLine; ++l) {
                    BlockPos moved = toMove_[l];
                    if (blockId(world_.getBlock(moved)) == BLOCK_SLIME && !addBranchingBlocks(moved)) return false;
                }
                return true;
            }

            state = world_.getBlock(forward);
            id = blockId(state);
            if (id == BLOCK_AIR) return true;

            if (!canPush(state, world_, forward, moveDirection_, true, moveDirection_) ||
                samePosPiston(forward, pistonPos_)) {
                return false;
            }

            if (blockData(id).pushReaction == PushReaction::Destroy) {
                toDestroy_[destroyCount_++] = forward;
                return true;
            }

            if (moveCount_ >= 12) return false;

            toMove_[moveCount_++] = forward;
            ++movedInLine;
            ++forwardDistance;
        }
    }

    MCPGPU_HD bool addBranchingBlocks(BlockPos fromPos) {
        for (int i = 0; i < 6; ++i) {
            const Facing& facing = facingByIndex(i);
            if (facing.axis != moveDirection_.axis) {
                if (!addBlockLine(offset(fromPos, facing), facing)) return false;
            }
        }
        return true;
    }

    MCPGPU_HD void reorderListAtCollision(int movedInLine, int collisionIndex) {
        int size = moveCount_;
        int out = 0;
        for (int i = 0; i < collisionIndex; ++i) reorderTmp_[out++] = toMove_[i];
        for (int i = size - movedInLine; i < size; ++i) reorderTmp_[out++] = toMove_[i];
        for (int i = collisionIndex; i < size - movedInLine; ++i) reorderTmp_[out++] = toMove_[i];
        for (int i = 0; i < out; ++i) toMove_[i] = reorderTmp_[i];
    }

    MCPGPU_HD int indexOfToMove(BlockPos pos) const {
        for (int i = 0; i < moveCount_; ++i) {
            if (samePosPiston(toMove_[i], pos)) return i;
        }
        return -1;
    }
};

} // namespace mcp1122gpu
