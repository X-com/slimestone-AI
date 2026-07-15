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

// Runs every candidate in `batch` through the persistent work-queue kernel, blocking until the
// whole batch completes, and returns one GpuResult per candidate in the same order.
// threadsPerBlock controls the launch config for simulateQueueKernel - see gpu_kernel.cu's
// runBatch() for how this trades off against per-block resource reservation. 64 is verified
// stable on this project's own dev card (a memory-constrained 2GB GTX 960); a newer/higher-VRAM
// GPU may tolerate a larger value for better latency-hiding, but bump it incrementally and
// re-verify against a full run (output diffed against reference/) rather than assuming a bigger
// number is free - 128 silently ran the dev card out of memory a few batches into a real run,
// not at launch time, so a quick smoke test isn't enough to catch a bad value.
std::vector<GpuResult> runBatch(const std::vector<HostCandidate>& batch, int maxTicks, int threadsPerBlock);

} // namespace mcp1122gpu
