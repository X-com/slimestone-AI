#include "launch.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace mcp1122gpu {

namespace {

#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                      \
        cudaError_t _status = (expr);                                                         \
        if (_status != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(_status));                                         \
            std::exit(1);                                                                      \
        }                                                                                       \
    } while (0)

// Milestone I: persistent-kernel/work-queue dispatch. Milestone H launched exactly one thread
// per candidate, so a warp stuck with even one slow candidate stalled every other lane in that
// warp for the slow candidate's whole run (SIMT lockstep - a warp doesn't retire until all its
// lanes finish, and an idle finished lane can't just go grab different work independently).
// Here a small, fixed pool of `numWorkers` threads stays resident for the whole launch; each one
// loops pulling the next unclaimed candidate index off a shared atomic counter until the queue
// is empty. A thread that finishes a short candidate immediately starts pulling more work
// instead of idling - the warp as a whole still only moves at the pace of its slowest currently-
// active lane per round, but "per round" is now one candidate instead of the whole batch, so
// imbalance averages out over many draws instead of being fixed by the initial static
// assignment. Decoupling worker count from candidate count is also why arbitrarily large batches
// no longer need arbitrarily large device memory (Milestone H's 1 Simulator + 1 cube per
// candidate); see FixedWorld::clearLiveCells() for how a worker's cube gets reused correctly
// across candidates instead of needing a fresh cudaMemset-sized slot per candidate.
__global__ void simulateQueueKernel(Simulator* workerStates, const GpuCandidateView* candidates,
                                     GpuResult* results, std::uint32_t* cellsPool,
                                     std::int32_t* entryIndexPool, int numCandidates, int numWorkers,
                                     int maxTicks, int* nextIndex) {
    int workerId = blockIdx.x * blockDim.x + threadIdx.x;
    if (workerId >= numWorkers) return;

    Simulator& sim = workerStates[workerId];
    std::size_t cellOffset = static_cast<std::size_t>(workerId) * FixedWorld::kCellCount;
    sim.bindCube(cellsPool + cellOffset, entryIndexPool + cellOffset);

    while (true) {
        int idx = atomicAdd(nextIndex, 1);
        if (idx >= numCandidates) break;
        results[idx] = sim.simulate(candidates[idx], maxTicks);
        sim.clearWorldCube();
    }
}

} // namespace

void initBlockRegistryDevice() {
    std::array<BlockData, 256> table = buildBlockRegistryTable();
    CUDA_CHECK(cudaMemcpyToSymbol(g_blockRegistry, table.data(), sizeof(BlockData) * 256));
}

namespace {

// Picks a worker count that fits comfortably in whatever device memory is actually free right
// now, instead of a hardcoded constant that would silently over- or under-subscribe different
// GPUs (this project's own dev box is a 2GB card with ~700MB typically free alongside the
// desktop compositor - a fixed "256 workers" guess would OOM there). Leaves a 30% headroom
// margin below free memory for the block/candidate/result pools and driver overhead.
int pickWorkerCount(int maxUseful) {
    std::size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    std::size_t perWorkerBytes =
        sizeof(Simulator) + sizeof(std::uint32_t) * FixedWorld::kCellCount + sizeof(std::int32_t) * FixedWorld::kCellCount;
    std::size_t budget = static_cast<std::size_t>(static_cast<double>(freeBytes) * 0.7);
    int fromMemory = static_cast<int>(budget / perWorkerBytes);
    int workers = std::max(1, std::min(fromMemory, maxUseful));
    return workers;
}

} // namespace

std::vector<GpuResult> runBatch(const std::vector<HostCandidate>& batch, int maxTicks) {
    int numCandidates = static_cast<int>(batch.size());
    std::vector<GpuResult> results(static_cast<std::size_t>(numCandidates));
    if (numCandidates == 0) return results;

    // Pool every candidate's blocks into one flat device array (CSR layout: per-candidate
    // offset + count into the shared pool) instead of one cudaMalloc per candidate.
    std::vector<GpuBlockEntry> pooledBlocks;
    std::vector<int> offsets(static_cast<std::size_t>(numCandidates));
    std::vector<int> counts(static_cast<std::size_t>(numCandidates));
    for (int i = 0; i < numCandidates; ++i) {
        offsets[static_cast<std::size_t>(i)] = static_cast<int>(pooledBlocks.size());
        counts[static_cast<std::size_t>(i)] = static_cast<int>(batch[static_cast<std::size_t>(i)].blocks.size());
        for (const GpuBlockEntry& b : batch[static_cast<std::size_t>(i)].blocks) pooledBlocks.push_back(b);
    }

    GpuBlockEntry* d_blocks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(GpuBlockEntry) * (pooledBlocks.empty() ? 1 : pooledBlocks.size())));
    if (!pooledBlocks.empty()) {
        CUDA_CHECK(cudaMemcpy(d_blocks, pooledBlocks.data(), sizeof(GpuBlockEntry) * pooledBlocks.size(),
                               cudaMemcpyHostToDevice));
    }

    std::vector<GpuCandidateView> views(static_cast<std::size_t>(numCandidates));
    for (int i = 0; i < numCandidates; ++i) {
        views[static_cast<std::size_t>(i)] = GpuCandidateView{
            batch[static_cast<std::size_t>(i)].id, batch[static_cast<std::size_t>(i)].trigger,
            d_blocks + offsets[static_cast<std::size_t>(i)], counts[static_cast<std::size_t>(i)]};
    }

    GpuCandidateView* d_candidates = nullptr;
    CUDA_CHECK(cudaMalloc(&d_candidates, sizeof(GpuCandidateView) * static_cast<std::size_t>(numCandidates)));
    CUDA_CHECK(cudaMemcpy(d_candidates, views.data(), sizeof(GpuCandidateView) * static_cast<std::size_t>(numCandidates),
                           cudaMemcpyHostToDevice));

    GpuResult* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(GpuResult) * static_cast<std::size_t>(numCandidates)));

    int* d_nextIndex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nextIndex, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nextIndex, 0, sizeof(int)));

    // Worker pool is sized independently of numCandidates now (Milestone I) - never bigger than
    // there is work for, never bigger than device memory allows.
    int numWorkers = pickWorkerCount(numCandidates);
    std::fprintf(stderr, "gpu_kernel: %d candidate(s), %d worker(s)\n", numCandidates, numWorkers);

    Simulator* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(Simulator) * static_cast<std::size_t>(numWorkers)));

    std::uint32_t* d_cellsPool = nullptr;
    std::int32_t* d_entryIndexPool = nullptr;
    std::size_t cellsPerWorker = static_cast<std::size_t>(FixedWorld::kCellCount);
    std::size_t cellsBytes = sizeof(std::uint32_t) * cellsPerWorker * static_cast<std::size_t>(numWorkers);
    std::size_t entryIndexBytes = sizeof(std::int32_t) * cellsPerWorker * static_cast<std::size_t>(numWorkers);
    CUDA_CHECK(cudaMalloc(&d_cellsPool, cellsBytes));
    CUDA_CHECK(cudaMalloc(&d_entryIndexPool, entryIndexBytes));
    // One-shot bulk zero-init per worker slot (see world.cuh's FixedWorld::reset()/
    // clearLiveCells() comments - every use after this first candidate is cleared incrementally
    // by clearLiveCells() instead of re-memsetting here).
    CUDA_CHECK(cudaMemset(d_cellsPool, 0x00, cellsBytes));
    CUDA_CHECK(cudaMemset(d_entryIndexPool, 0xFF, entryIndexBytes));

    // Default per-thread stack (~1KB) is too small for this call graph's depth (rail power
    // search recurses up to 8 deep via findPoweredRailSignal<->isSameRailWithPower, plus the
    // neighbor-propagation/piston-move chain nests several calls deep). 16KB/thread is
    // generous headroom now that the >100KB scratch snapshot buffers moved off the stack (see
    // simulator.cuh's *Scratch_ members).
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));

    int threadsPerBlock = 32;
    int blocks = (numWorkers + threadsPerBlock - 1) / threadsPerBlock;
    simulateQueueKernel<<<blocks, threadsPerBlock>>>(d_states, d_candidates, d_results, d_cellsPool,
                                                       d_entryIndexPool, numCandidates, numWorkers, maxTicks,
                                                       d_nextIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(results.data(), d_results, sizeof(GpuResult) * static_cast<std::size_t>(numCandidates),
                           cudaMemcpyDeviceToHost));

    cudaFree(d_blocks);
    cudaFree(d_candidates);
    cudaFree(d_results);
    cudaFree(d_nextIndex);
    cudaFree(d_states);
    cudaFree(d_cellsPool);
    cudaFree(d_entryIndexPool);

    return results;
}

} // namespace mcp1122gpu
