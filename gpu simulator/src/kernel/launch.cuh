#pragma once

// Host-facing entry point into the CUDA kernel (Milestone H: naive one-thread-per-candidate
// dispatch, no persistent-kernel/work-queue scheduling yet - that's Milestone I). Everything
// device-shaped lives behind this one function so src/host/main_gpu.cu doesn't need to know
// about cudaMalloc/kernel launch syntax - it just builds a batch and reads results back.

#include "simulator.cuh"

#include <cstdint>
#include <vector>

namespace mcp1122gpu {

// One candidate's blocks as flattened host-side input; runBatch() pools these into one device
// allocation (CSR-style: offsets into a single concatenated block array) before launching.
struct HostCandidate {
    std::int64_t id = 0;
    BlockPos trigger;
    std::vector<GpuBlockEntry> blocks;
};

// Uploads the block-data lookup table to device __constant__ memory. Call once before the first
// runBatch().
void initBlockRegistryDevice();

// Runs every candidate in `batch` as one CUDA thread each, blocking until the whole batch
// completes, and returns one GpuResult per candidate in the same order.
std::vector<GpuResult> runBatch(const std::vector<HostCandidate>& batch, int maxTicks);

} // namespace mcp1122gpu
