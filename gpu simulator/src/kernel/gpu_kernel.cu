#include "launch.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
                                     GpuResult* results, std::uint64_t* hashKeysPool,
                                     std::int32_t* hashSlotPool, int numCandidates, int numWorkers,
                                     int maxTicks, int* nextIndex) {
    int workerId = blockIdx.x * blockDim.x + threadIdx.x;
    if (workerId >= numWorkers) return;

    Simulator& sim = workerStates[workerId];
    std::size_t hashOffset = static_cast<std::size_t>(workerId) * FixedWorld::kHashCapacity;
    sim.bindCube(hashKeysPool + hashOffset, hashSlotPool + hashOffset);

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

// Set MCP1122_GPU_PROFILE=1 to print per-batch phase timings (alloc/upload, worker-pool
// setup, kernel, download) to stderr via CUDA events. Off by default - cudaEventRecord/
// Synchronize add real overhead (forces a sync point CUDA would otherwise pipeline around), so
// this should stay opt-in rather than always-on instrumentation. Exists to answer, with actual
// numbers instead of guesses, how much of a batch's wall time is malloc/memset/free churn (the
// question that decides whether persistent-buffer reuse across batches - see runBatch()'s
// per-batch cudaMalloc calls - is worth the added complexity here).
bool profilingEnabled() {
    static const bool enabled = std::getenv("MCP1122_GPU_PROFILE") != nullptr;
    return enabled;
}

// Picks a worker count that fits comfortably in whatever device memory is actually free right
// now, instead of a hardcoded constant that would silently over- or under-subscribe different
// GPUs (this project's own dev box is a 2GB card with ~700MB typically free alongside the
// desktop compositor - a fixed "256 workers" guess would OOM there). Leaves a 30% headroom
// margin below free memory for the block/candidate/result pools and driver overhead.
int pickWorkerCount(int maxUseful) {
    std::size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    std::size_t perWorkerBytes =
        sizeof(Simulator) + sizeof(std::uint64_t) * FixedWorld::kHashCapacity + sizeof(std::int32_t) * FixedWorld::kHashCapacity;
    std::size_t budget = static_cast<std::size_t>(static_cast<double>(freeBytes) * 0.7);
    int fromMemory = static_cast<int>(budget / perWorkerBytes);
    int workers = std::max(1, std::min(fromMemory, maxUseful));
    return workers;
}

// Persistent, grow-only device buffers for runBatch()'s per-call inputs/outputs (blocks,
// candidate views, results, the atomic work-queue counter), reused across batches instead of
// torn down and rebuilt every call. MCP1122_GPU_PROFILE (see profilingEnabled() above) was used
// to find the real cost here, and it disproved two hypotheses in a row - worth keeping both, so
// nobody re-tries them without a profiler in hand:
//
// 1. "It's the four per-batch cudaMalloc calls" - looked obvious before measuring ("upload" was
//    ~75-100ms of a ~110ms batch, dwarfing the worker pool's ~2ms). Reusing these buffers (this
//    struct) removed all four mallocs, and "upload" didn't move.
// 2. "It's WDDM's implicit staging copy for pageable host memory" - the two H2D cudaMemcpy
//    calls were ~75ms combined for a payload of a few tens of KB, vastly more than raw PCIe
//    bandwidth would predict, and lopsided against the D2H download (comparable size, <0.3ms) -
//    a classic pageable-vs-pinned-memory symptom. Staged the same transfers through
//    cudaHostAlloc'd pinned buffers instead of plain heap memory; "upload" still didn't move.
//
// hostMarshalMs in runBatch() ruled out the CPU-side vector-building loop too (<1ms). Net: this
// buffer reuse is still worth keeping (real cudaMalloc/cudaFree calls removed, verified
// byte-identical against reference/ across two full runs), but it is NOT what makes "upload"
// slow - that ~75ms/batch looks like a fixed cost of synchronous CUDA calls under WDDM on this
// card that neither allocation reuse nor pinned memory touches. Next step, if pursued: Nsight
// Systems' timeline view to see whether it's the memcpy call itself or something serializing
// around it (the profiler this needs, not the CUDA-event timing already in this file - see the
// PROFILING note in the project write-up). Capacities only ever grow (never shrink/free
// mid-run): candidates/results-per-batch are bounded by --batch-size (constant for the whole
// process), and blocks-per-batch varies with candidate content but a high-water-mark buffer
// handles that the same way. Left out of scope deliberately: the worker pool (d_states/hash
// pools) stays exactly as before, freshly sized from live cudaMemGetInfo() every batch - it
// isn't the bottleneck the profiler found, and reusing it would reintroduce the risk flagged
// earlier (a fixed allocation stops adapting to this GPU's fluctuating free memory as the
// desktop compositor competes for it, since this is the display-driving GPU, not a headless
// compute card).
struct PersistentBuffers {
    GpuBlockEntry* blocks = nullptr;
    std::size_t blocksCapacity = 0;
    GpuCandidateView* candidates = nullptr;
    std::size_t candidatesCapacity = 0;
    GpuResult* results = nullptr;
    std::size_t resultsCapacity = 0;
    int* nextIndex = nullptr;
};

PersistentBuffers& persistentBuffers() {
    static PersistentBuffers buffers;
    return buffers;
}

template <typename T>
void ensureCapacity(T** ptr, std::size_t* capacity, std::size_t needed) {
    if (needed <= *capacity) return;
    if (*ptr != nullptr) cudaFree(*ptr);
    CUDA_CHECK(cudaMalloc(ptr, sizeof(T) * needed));
    *capacity = needed;
}

} // namespace

std::vector<GpuResult> runBatch(const std::vector<HostCandidate>& batch, int maxTicks, int threadsPerBlock) {
    int numCandidates = static_cast<int>(batch.size());
    std::vector<GpuResult> results(static_cast<std::size_t>(numCandidates));
    if (numCandidates == 0) return results;

    bool profiling = profilingEnabled();
    cudaEvent_t evStart, evUploaded, evPoolReady, evKernelDone, evDownloaded;
    if (profiling) {
        cudaEventCreate(&evStart);
        cudaEventCreate(&evUploaded);
        cudaEventCreate(&evPoolReady);
        cudaEventCreate(&evKernelDone);
        cudaEventCreate(&evDownloaded);
        cudaEventRecord(evStart);
    }

    auto hostMarshalStart = std::chrono::steady_clock::now();

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

    PersistentBuffers& persist = persistentBuffers();

    std::size_t blocksNeeded = pooledBlocks.empty() ? 1 : pooledBlocks.size();
    ensureCapacity(&persist.blocks, &persist.blocksCapacity, blocksNeeded);
    GpuBlockEntry* d_blocks = persist.blocks;
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

    double hostMarshalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostMarshalStart).count();

    ensureCapacity(&persist.candidates, &persist.candidatesCapacity, static_cast<std::size_t>(numCandidates));
    GpuCandidateView* d_candidates = persist.candidates;
    CUDA_CHECK(cudaMemcpy(d_candidates, views.data(), sizeof(GpuCandidateView) * static_cast<std::size_t>(numCandidates),
                           cudaMemcpyHostToDevice));

    ensureCapacity(&persist.results, &persist.resultsCapacity, static_cast<std::size_t>(numCandidates));
    GpuResult* d_results = persist.results;

    if (persist.nextIndex == nullptr) {
        CUDA_CHECK(cudaMalloc(&persist.nextIndex, sizeof(int)));
    }
    int* d_nextIndex = persist.nextIndex;
    CUDA_CHECK(cudaMemset(d_nextIndex, 0, sizeof(int)));

    if (profiling) cudaEventRecord(evUploaded);

    // Worker pool is sized independently of numCandidates now (Milestone I) - never bigger than
    // there is work for, never bigger than device memory allows.
    int numWorkers = pickWorkerCount(numCandidates);
    std::fprintf(stderr, "gpu_kernel: %d candidate(s), %d worker(s)\n", numCandidates, numWorkers);

    Simulator* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(Simulator) * static_cast<std::size_t>(numWorkers)));

    std::uint64_t* d_hashKeysPool = nullptr;
    std::int32_t* d_hashSlotPool = nullptr;
    std::size_t slotsPerWorker = static_cast<std::size_t>(FixedWorld::kHashCapacity);
    std::size_t hashKeysBytes = sizeof(std::uint64_t) * slotsPerWorker * static_cast<std::size_t>(numWorkers);
    std::size_t hashSlotBytes = sizeof(std::int32_t) * slotsPerWorker * static_cast<std::size_t>(numWorkers);
    CUDA_CHECK(cudaMalloc(&d_hashKeysPool, hashKeysBytes));
    CUDA_CHECK(cudaMalloc(&d_hashSlotPool, hashSlotBytes));
    // One-shot bulk init per worker slot (see world.cuh's FixedWorld::reset()/clearLiveCells()
    // comments - every use after this first candidate is cleared incrementally by
    // clearLiveCells() instead of re-memsetting here). 0xFF over int32 slot values is -1
    // ("empty"), which is the sentinel findSlot()/hashInsert() rely on; the keys pool's initial
    // contents are never read for an empty slot, so a plain zero-fill is fine there.
    CUDA_CHECK(cudaMemset(d_hashKeysPool, 0x00, hashKeysBytes));
    CUDA_CHECK(cudaMemset(d_hashSlotPool, 0xFF, hashSlotBytes));

    // Default per-thread stack (~1KB) is too small for this call graph's depth (rail power
    // search recurses up to 8 deep via findPoweredRailSignal<->isSameRailWithPower, plus the
    // neighbor-propagation/piston-move chain nests several calls deep). 16KB/thread is
    // generous headroom now that the >100KB scratch snapshot buffers moved off the stack (see
    // simulator.cuh's *Scratch_ members).
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));

    if (profiling) cudaEventRecord(evPoolReady);

    // threadsPerBlock is caller-supplied (main_gpu.cu's --threads-per-block, default 64) rather
    // than hardcoded, because the right value is hardware-dependent, not a constant this file can
    // know. 32 (one warp/block) under-uses the scheduler's ability to hide memory latency by
    // keeping more warps resident per SM - each worker thread here is scalar, branchy,
    // pointer-chasing code with no block-level cooperation (no __syncthreads, no shared memory),
    // so nothing about this kernel benefits from small blocks in principle. In practice this
    // project's own dev card's implicit per-block resource reservation (registers/local-memory
    // backing this call graph's depth, amplified by cudaDeviceSetLimit(cudaLimitStackSize)
    // above) grows with block size in a way pickWorkerCount()'s cudaMemGetInfo budget can't see
    // ahead of time: 128 threads/block reliably ran that 2GB card out of memory a few batches
    // into a full flyers.data run (cudaMemset failing with "unknown error", the classic symptom
    // of an earlier kernel launch silently exceeding device memory) - not a hang, an outright
    // crash, and not at launch time, so a quick smoke test wouldn't have caught it either. 64 was
    // verified stable and byte-identical against reference/ across two full 12,800-candidate
    // runs on that card, hence the default. A newer/higher-VRAM GPU may tolerate more - but bump
    // it incrementally and re-verify the same way (a full run to completion, output diffed
    // against reference/), not by reasoning that a bigger number "should" be fine.
    threadsPerBlock = std::max(1, std::min(threadsPerBlock, 1024));
    int blocks = (numWorkers + threadsPerBlock - 1) / threadsPerBlock;
    simulateQueueKernel<<<blocks, threadsPerBlock>>>(d_states, d_candidates, d_results, d_hashKeysPool,
                                                       d_hashSlotPool, numCandidates, numWorkers, maxTicks,
                                                       d_nextIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiling) cudaEventRecord(evKernelDone);

    CUDA_CHECK(cudaMemcpy(results.data(), d_results, sizeof(GpuResult) * static_cast<std::size_t>(numCandidates),
                           cudaMemcpyDeviceToHost));
    if (profiling) cudaEventRecord(evDownloaded);

    // d_blocks/d_candidates/d_results/d_nextIndex are NOT freed here - they're the persistent,
    // grow-only buffers in PersistentBuffers (see its comment above), reused by the next
    // runBatch() call instead of being torn down and rebuilt every time. Only the worker pool
    // (sized fresh from live free memory every batch - see pickWorkerCount()) still gets freed.
    cudaFree(d_states);
    cudaFree(d_hashKeysPool);
    cudaFree(d_hashSlotPool);

    if (profiling) {
        cudaEventSynchronize(evDownloaded);
        float uploadMs = 0, poolMs = 0, kernelMs = 0, downloadMs = 0, totalMs = 0;
        cudaEventElapsedTime(&uploadMs, evStart, evUploaded);
        cudaEventElapsedTime(&poolMs, evUploaded, evPoolReady);
        cudaEventElapsedTime(&kernelMs, evPoolReady, evKernelDone);
        cudaEventElapsedTime(&downloadMs, evKernelDone, evDownloaded);
        cudaEventElapsedTime(&totalMs, evStart, evDownloaded);
        std::fprintf(stderr,
                     "gpu_kernel: phase timings (ms) - upload=%.2f (hostMarshal=%.2f) workerPoolAlloc=%.2f "
                     "kernel=%.2f download=%.2f total=%.2f\n",
                     uploadMs, hostMarshalMs, poolMs, kernelMs, downloadMs, totalMs);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evUploaded);
        cudaEventDestroy(evPoolReady);
        cudaEventDestroy(evKernelDone);
        cudaEventDestroy(evDownloaded);
    }

    return results;
}

} // namespace mcp1122gpu
