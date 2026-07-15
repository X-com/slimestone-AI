# MCP1122 GPU Extract

This folder is the GPU port of `../cpp extract`, which is itself the C++ port of `../extract`.

`../cpp extract` remains the oracle until this port matches it. `../extract` (Java) is the
original oracle behind that.

## Why this is structured differently from a normal GPU project

A single candidate's tick loop (hash lookups, piston structure helper, redstone propagation,
block-event queues) is a branchy, pointer-chasing state machine — a bad fit for parallelizing
*inside* one candidate across GPU threads. The real parallelism is across candidates: run one
candidate per thread, thousands of threads at once, instead of one candidate per CPU process
the way `genetic_ml/simulator_pool.py` does it today.

That means every candidate's world state has to become fixed-size and allocation-free
(bounded bounding box, flat arrays instead of `std::unordered_map`/`std::unordered_set`,
fixed-capacity ring buffers for scheduled ticks / block events / moving blocks) before any of
this can run on a device.

## Layout

```text
src/
  reference/   plain single-threaded CPU code using the exact flat-array, fixed-capacity
               layout the kernel will use. No STL containers, no exceptions, no recursion.
               This is the debug environment: build and validate the GPU-shaped algorithm
               here, on a real CPU debugger, before touching a device.
  kernel/      the actual GPU kernel(s) - one thread per candidate, persistent work-queue
               dispatch to avoid warp divergence from candidates finishing at wildly
               different tick counts.
  host/        device buffer setup, candidate batch marshalling, result gather.
```

## Porting Milestones

1. Keep `../cpp extract` fixture results and traces (`../extract/outlog/stream-input/*.json`)
   as the oracle.
2. Design the fixed-capacity per-candidate world layout (bounded bounding box, packed pos
   indexing, fixed ring buffers for ticks/events/moving blocks).
3. Port state-key/cycle detector into `reference/` against the flat layout.
4. Port scheduled tick and block-event queues into `reference/`.
5. Port observer behavior into `reference/`.
6. Port piston structure helper into `reference/` using fixed-size arrays (already bounded by
   Minecraft's push limit).
7. Port piston block-event handling and `doMove` into `reference/`.
8. Port moving block settlement into `reference/`.
9. Diff `reference/` output against `../cpp extract` fixture results, tick-by-tick trace diff
   for any mismatch, same shape as `../cpp extract/VERIFICATION.md`.
10. Only once `reference/` matches exactly: translate it into a single-candidate-per-thread
    kernel with naive one-thread-per-candidate dispatch. Validate identical output on a
    handful of candidates.
11. Add persistent-kernel / work-queue dispatch so idle threads pull the next candidate
    instead of sitting idle until the slowest lane in the batch finishes.
12. Benchmark candidates/sec against the current CPU multi-process pool
    (`genetic_ml/simulator_pool.py`). This is the number that decides whether the port paid
    for itself - do this before any further kernel optimization.

Do not treat anything under `kernel/` as correctness-tested until milestone 9 has passed for
`reference/` first. A bug found on the device should be reproduced in `reference/` before
being debugged - not chased directly in device code.

## Status

Milestones 1-9 done (see `VERIFICATION.md` for the full checklist and numbers). `reference/`
is now a complete, verified port of `cpp extract`'s `Simulator`:

- `FixedWorld` (fixed 128^3 flat cube addressed by `worldPos - origin`) plus fixed-capacity
  queues: `blockEvents[2]` (cap 512), `scheduledTicks` (cap 512), `movingBuckets[3]` (cap 16384
  each) - all push operations throw `std::out_of_range` on overflow instead of growing or
  truncating silently.
- `piston.h/.cpp` ported near-verbatim from `cpp extract` (the algorithm was already
  fixed-array-shaped - `std::array<BlockPos,12>`/`<BlockPos,4>` bounded by the vanilla push
  limit - only the `World` type changed).
- The full tick loop: scheduled ticks, block events, observer behavior, redstone power
  (including rail power relay), piston structure helper, piston movement, moving-block
  settlement, and translation-invariant cycle detection (`stateKey()`/`detectShiftCycle()`).
- `FixedWorld::entries()` is an incrementally maintained dense list (swap-remove on delete,
  same technique as cpp extract's `PosStateMap::entries()`) - O(1) amortized to keep in sync
  from `setBlock()`, O(live count) to read, not an O(kExtent^3) cube scan.
- `watchSet_`/`observerSet_`/`ncPowerSet_` are `PosKeySet` (`world.h`) - fixed-capacity
  (4096), linear-scan-membership arrays, not `std::unordered_set`.
- One deliberate, documented simplification remains, performance-only: no incremental
  block-hash cache (`stateKey()` still rehashes every live block from scratch each tick rather
  than applying a per-`setBlockState` delta like cpp extract's `bhc_`). Fine while live counts
  stay small; see the `Simulator` class doc comment in `simulator.h`.

Verified against all 37 `../extract/outlog/stream-input/**/*.json` fixtures: 35 in-range
candidates produce **byte-identical full tick-by-tick traces** against `cpp extract --trace`
(not just matching final results) across runs up to 6000 ticks, and identical final
`ok`/`working`/`ticks`/`start`/`end`/`period`/`shift`. The 2 known oversized fixtures still
fail loudly and cleanly with `errorCode: "simulation_error"` (cube too small), matching cpp
extract's own error-handling shape (single catch around `loadCandidate`+`trigger`+
`detectShiftCycle`, same as cpp extract's `simulate()`).

Also verified against the full `genetic_ml/data/compact-working/flyers.data` corpus (12,800
real GA-archived candidates): 12,800/12,800 `ok:true`/`working:true`, zero errors. That run
took ~14.5s after the `entries()`/`PosKeySet` optimization above, down from ~7 minutes before
it (~25x on a 200-candidate timing sample: 7.76s -> 0.31s) - the O(kExtent^3) full-cube scan
was the dominant per-tick cost, exactly as flagged before this milestone.

Milestone 10 (kernel port) done - see VERIFICATION.md's Milestone H entry. `src/kernel/` now
holds a device-compatible, near-verbatim mechanical translation of `reference/` (fixed-capacity
arrays instead of `std::vector`, a sticky error flag instead of exceptions - see world.cuh's file
header for why that's behavior-identical for every fixture that doesn't hit a capacity ceiling)
plus a naive one-thread-per-candidate `__global__` kernel. `src/host/main_gpu.cu` is the CUDA
counterpart to `reference/main.cpp` - same compact-binary stdin protocol, same JSON output shape.

Milestone 11 (persistent-kernel/work-queue dispatch) done too - see VERIFICATION.md's Milestone I
entry. A small, memory-sized pool of worker threads now pulls candidates off a shared atomic
counter instead of Milestone 10's one-thread-per-candidate fixed assignment, and device memory
is sized by worker count (auto-picked from free VRAM) instead of batch size, so a batch no longer
needs to be kept small just to fit in memory.

Milestone 12 (throughput benchmark against `genetic_ml/simulator_pool.py`) done too - see
`genetic_ml/benchmark_gpu_vs_pool.py` and VERIFICATION.md's Milestone J entry. Short version: on
this hardware (GTX 960, a weak/old 8-SM WDDM-mode consumer card), the kernel is currently ~15%
*slower* than the existing 12-process CPU pool on the real `flyers.data` GA population, memory-
limited to only ~50-60 concurrent workers rather than compute-limited - not the throughput win
the porting effort was ultimately aiming for, reported here as-measured rather than glossed over.

Also still open, and now better understood rather than fixed by Milestone 11: Windows WDDM GPUs
(not TCC) kill any single kernel launch that runs past the driver's ~2s timeout (TDR), and
work-queue dispatch only helps *batch-level* imbalance (idle lanes waiting on the batch's
slowest candidate) - it can't shorten one candidate's own serial tick loop, which is where the
large `dont-fly/` fixtures (thousand-block machines like `tank.json`, multi-thousand-tick runs)
actually spend their time. Confirmed by testing: even a TDR-safe small batch containing only
medium candidates (the largest requiring 528 ticks) still hits the timeout once those candidates
carry thousands of live blocks each, since `stateKey()`'s per-tick cost is O(live block count) -
the same rehash-every-tick simplification `reference/`'s own doc comment already flags, now
compounded by a GPU thread being much slower per-scalar-op than a CPU core. Validated correct
(byte-identical results against `reference/`) on every candidate in
`genetic_ml/data/compact-working/flyers.data` (1584 candidates from a real GA run, ~1.4s total)
and every fast-resolving fixture in `../extract/outlog/stream-input/dont-fly/`; the slow/large
fixtures there remain correctness-untested on the kernel path because no batch containing them
finishes before TDR kills it. Fixing that for real is either a system-level TDR timeout change
(out of scope here) or checkpointed multi-launch ticking (a real, separate engineering task, not
attempted).

Also cross-validated against three *independently implemented* engines, not just against
`reference/`'s own history: `genetic_ml/run_four_way_compare.py` (uses `four_way_compare.py`,
formerly `three_way_compare.py` before this GPU engine joined) feeds every one of the 12,800
`flyers.data` candidates through the C++ engine, the Java `extract` engine, the real-Minecraft
`mcp1122` engine, and this project's `reference/` build (`gpu_simulator_stream.exe` - the CPU-
shaped oracle, not `gpu_kernel_stream.exe`, since `kernel/` isn't trusted as its own oracle per
this file's own rule above) and checks all four agree on `working`/`shift`/`period`. Re-ran it as
part of this milestone: **0 mismatches across all 12,800 candidates.** `genetic_ml/
four_way_pool.py`'s `FourWaySimulatorPool` (the same cross-check wired live into the GA loop via
`main_ga_4way_test.py`) was also smoke-tested directly and works.

## Running the CUDA kernel

```powershell
.\build-cuda.bat        # builds build\gpu_kernel_stream.exe (nvcc + MSVC host compiler)
.\test-cuda.bat          # runs it against genetic_ml/data/compact-working/flyers.data
```

Mirrors `build.bat`/`test.bat` for `reference/` - same idea, pointed at the CUDA kernel instead.

## Build

```powershell
.\build-msys2.ps1
```

or `build.bat` (double-clickable, calls the same script and pauses on completion).

Builds `gpu_simulator_stream.exe`, reusing `../cpp extract/src/json_stream.cpp` and
`block_registry.cpp` unmodified rather than duplicating them - only the world/trace/simulator
layers differ.

## Run

Same compact-binary stdin protocol as `cpp extract`:

```powershell
Get-Content -Raw '..\..\genetic-ml\data\compact-working\flyers.data' |
  .\build\gpu_simulator_stream.exe --trace outlog\gpu-ref-trace.log
```

Or `test.bat` - builds if needed, then runs the full `flyers.data` corpus in file mode and
reports the exit code (mirrors `cpp extract`'s `run-flyers-data.bat`).
