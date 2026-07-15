# Verification Plan

`../cpp extract` remains the behavior oracle until `reference/` matches it exactly. Only after
`reference/` matches does `kernel/` become something to trust.

## Reference Commands

Generate cpp extract fixture results to diff against (from `../cpp extract`):

```powershell
Get-ChildItem '..\extract\outlog\stream-input' -Filter '*.json' |
  Sort-Object Name |
  Get-Content |
  .\build\mcp1122_cpp_stream.exe --trace outlog\cpp-update-trace.log
```

`reference/` should use the same `Trace` line shape (`worldTime tag x,y,z optionalA,optionalB`)
so traces diff directly against cpp extract's.

## Status

Milestones A through G done and verified (full engine port). `src/kernel`, `src/host` are
still empty - no GPU code written yet.

Checked by encoding every fixture under `../extract/outlog/stream-input/**/*.json` to the
compact format and running both `cpp extract --trace` and `gpu_simulator_stream --trace` on
each:

```text
fixtures=37
full tick-by-tick trace files byte-identical (diff -q)=35
final result (ok/working/ticks/start/end/period/shift) identical=35
cube-too-small failures (expected, both engines agree it's out of scope)=2
mismatches=0
```

"Full tick-by-tick trace files byte-identical" means the entire `--trace` output file for
each candidate - every `h.tick`/`w.beq`/`p.chk`/`p.mv+`/`te.*`/etc. line across the whole run,
not just the load-time lines - diffs to nothing, including candidates that ran the full
1797-tick or 6000-tick(cap) simulation. That's a much stronger check than matching final
results alone: two engines could reach the same `working`/`shift` by different tick-by-tick
paths, and a full trace diff would catch that; this one doesn't need to, because there is no
divergence anywhere in the run.

The 2 oversized fixtures ("1797 onepointfive.json", "1 wide trencher.json", spans of 266/206
blocks) fail with `errorCode: "simulation_error"` and no trace output - cpp extract's own
error-handling shape (`simulate()` wraps `loadCandidate()`+`trigger()`+`detectShiftCycle()` in
one `catch`, so cpp extract's own bounds violations - if it had any - would report the same
way). Not a defect; revisit `FixedWorld::kExtent` if these two need to be in scope later.

## One-off spot check: `cpp extract/flying-json/1-wide with rails.json`

A third real fixture, outside both the `../extract/outlog/stream-input` set above and
`flyers.data` - picked to exercise rail power propagation (`railNeighborChanged`/
`updateRailPowerState`/`findPoweredRailSignal`), a code path the other two suites don't hit as
heavily. Confirmed by the person who built it to actually fly in real Minecraft - a genuine
correctness check, not just an engine-to-engine agreement check.

**First pass, misleading:** same span problem as the two oversized fixtures above (206 blocks
tall, needs extent 238) - `reference/`'s `FixedWorld::kExtent` was temporarily bumped to 256
(host RAM only, no GPU/device code touched) so it could load at all. Both `cpp extract` and
`gpu extract` then agreed with each other - byte-identical full traces again - but both reported
`working:false` (ran the full 6000-tick cap, no cycle found). That agreement looked like
confirmation, but it wasn't: this machine is known to actually work, so two engines agreeing on
a wrong answer is worse than useless if left unquestioned.

**Root cause, found by bringing in a fourth, independent engine** (`extract`, the original Java
port, and `mcp1122`, the real-Minecraft engine - see `genetic_ml/four_way_compare.py`): all
three "shaped" engines (`cpp`, `gpu`, `extract`) apply a shared `CANDIDATE_Y_OFFSET = 64` to
every candidate on load (`cpp extract/src/json_stream.cpp`; `gpu extract` reuses that same file;
`extract`'s Java reader independently reimplements the identical +64), meant to keep small
candidates' local Y coordinates non-negative. This fixture spans y=1..206 locally - after +64
that's y=65..270, and **36 of its 1080 blocks (the top ~15 layers) land at y>=256**, past real
Minecraft's actual world-height ceiling. Every engine's `setBlockState` has a shared guard
mirroring that real limit (`if (pos.y < 0 || pos.y >= 256) return;`), so those 36 blocks get
placed during load (which bypasses that check) but can never have their redstone/piston logic
updated afterward - silently inert in `cpp`/`gpu`/`extract` alike. `mcp1122` doesn't degrade
silently the same way - backed by a real chunk array instead of a lenient map/cube, it crashes
outright (`ArrayIndexOutOfBoundsException`) trying to place a block that high, which is what
made the bug visible instead of just another quiet agreement.

**Second pass, corrected:** re-tested with the fixture's blocks shifted down by 15 (so
`max_y(206) + 64 - 15 = 255`, exactly under the ceiling) - a fixture-only workaround, no change
to the shared `CANDIDATE_Y_OFFSET` logic itself (that's a separate, larger fix touching the
whole pipeline, not done here). All three engines now agree on a real result:

```text
cpp:     ok=true working=true ticks=3570 start=1352 end=3570 period=2218 shift={0,0,1}
gpu:     ok=true working=true ticks=3570 start=1352 end=3570 period=2218 shift={0,0,1}
extract: ok=true working=true ticks=3570 start=1352 end=3570 period=2218 shift={0,0,1}
```

`cpp` vs `gpu` full traces are **byte-identical** again - 3,972,896 lines, `diff -q` reports no
difference, confirmed a second way with `compare_traces.py`. `extract`'s trace isn't diffable
line-for-line against `cpp`/`gpu` (different tag vocabulary - `extract` only mirrors `mcp1122`'s
tags since it's instrumented Minecraft source, and it has no per-block load-time trace lines at
all), which is expected and by design: `genetic_ml/four_way_compare.py` itself only ever
compares final results against `extract`, never full traces, for exactly this reason. Final
results matching exactly across all three independently-implemented engines is the right and
complete check at that level.

Net: `cpp` vs `gpu` remains proven byte-identical on this fixture either way (both the misleading
`working:false` first pass and the corrected `working:true` second pass agree perfectly between
those two). The bug this surfaced - `CANDIDATE_Y_OFFSET=64` silently clipping tall machines - is
not a `cpp`-vs-`gpu` port defect; it predates the GPU port entirely and affects the whole shared
pipeline (`main_ga.py`, `four_way_compare.py`, every engine). Flagged here, not fixed - a proper
fix (offset only by as much as each candidate actually needs, not a flat +64) is a separate,
larger change across `json_stream.cpp`, `extract`'s reader, and whatever `mcp1122`-side code
applies the same constant, and wasn't requested as part of this verification pass.

One capacity constant needed correcting during this port: `kMaxMovingPerBucket` started at
256 and was hit by two large fixtures (`complex_machine.json`, 3154 blocks; `tank.json`, 4731
blocks) with `errorCode: "simulation_error"`. Raised to 16384 and re-verified full-trace-match
on both - confirming it was an undersized constant, not a stuck/leaking moving-block bug
(which would have kept failing at any capacity).

## Performance Pass: entries() and PosKeySet (post-Milestone G)

`FixedWorld::entries()` and the `watchSet_`/`observerSet_`/`ncPowerSet_` sets were flagged in
the `Simulator` class doc comment as real gaps before kernel work: `entries()` was an
O(kExtent^3) full-cube scan called every tick (via `stateKey()`), and the three sets were
`std::unordered_set` rather than fixed-capacity. Both fixed:

- `entries()` is now an incrementally maintained dense list, updated in `setBlock()` with the
  same swap-remove technique as cpp extract's `PosStateMap::entries()` - O(1) amortized to
  maintain, O(live count) to read.
- The three sets became `PosKeySet` (`world.h`): fixed-capacity (4096) linear-scan arrays,
  throwing `std::out_of_range` on overflow like every other fixed-capacity structure in this
  port.

Re-verified full correctness after the change - same fixture and trace-diff results as
Milestone G (35/37 byte-identical, same 2 known oversized failures, 0 mismatches) - then
measured the effect:

```text
200-candidate sample from flyers.data: 7.76s -> 0.31s (~25x)
full flyers.data (12,800 candidates): ~7 min -> 14.5s
full flyers.data result: 12,800/12,800 ok:true, working:true, 0 errors
```

The O(kExtent^3) scan (2M cells at `kExtent=128`) was the dominant per-tick cost even for the
tiny (~7-block average) `flyers.data` candidates, since it ran regardless of how few blocks
actually existed. This is now closed; the remaining documented simplification (no incremental
block-hash cache in `stateKey()` - see `simulator.h`) is a much smaller cost at these live
counts and has not needed revisiting.

`b.initkey` is a trace line (`worldTime tag blockCount blockA blockB`, decimal) added via
`Trace::logHash()` in both engines' trace classes - cpp extract's is a small additive change
(new method + one call site in `loadCandidate()`, both gated behind `trace_ != nullptr`, no
behavior change with tracing off). It dumps the block-only portion of `stateKey()` right after
`loadCandidate()` returns, before `trigger()`/`tickWorld()` have run.

## Milestone Checks

Mirrors `../cpp extract/VERIFICATION.md` milestones 1-7, but the oracle is `../cpp extract`
instead of the Java `../extract`, and every milestone is implemented in `reference/` (flat
fixed-capacity layout, no STL containers) before it is ever attempted in `kernel/`.

### Milestone A: Fixed-Capacity World Layout

Expected:

- `reference/` parses the same compact/JSON candidate stream as `cpp extract`.
- Candidate blocks load into a fixed-size flat array (bounded bounding box), not a hash map.
- Bounding-box overflow is a hard, explicit error - not a silent truncation - since a
  candidate that doesn't fit the fixed layout must be flagged, not mis-simulated.

### Milestone B: Initial State Key

Compare `reference/`'s initial packed state key against `cpp extract`'s for every fixture.

### Milestone C: Tick Queues And Block Events — done

Fixed-capacity scheduled-tick and block-event queues (`FixedWorld::pushScheduledTick`/
`pushBlockEvent`). `h.tick`/`w.beq` lines verified byte-identical across full traces.

### Milestone D: Observer Behavior — done

Observer schedule tick, powered/unpowered state id changes, neighbor update order all
verified byte-identical (`o.tick`/`o.obs`/`o.src` lines).

### Milestone E: Piston Structure Helper — done

`piston.h/.cpp` ported near-verbatim (already fixed-array-shaped in cpp extract). Move count,
destroy count, and positions in order verified via full trace match (`p.chk`/`p.q+`/`p.q-`).

### Milestone F: Piston Movement — done

Piston block event handling, movement, and moving-block settlement ported
(`pistonEventReceived`/`doPistonMove`/`settleMovingBlock`). Verified via `p.ev`/`p.mv+`/
`p.mv-`/`te.*` lines matching across full traces, including the vanilla `aiblockstate`
scratch-array quirk (see the comment in `simulator.cpp`'s `doPistonMove`).

### Milestone G: Cycle Detection — done

Final results (`ok`, `working`, `ticks`, `start`, `end`, `period`, `shift`) verified identical
on all 35 in-range fixtures, including ones that ran the full 1797-tick or 6000-tick(cap)
simulation.

`reference/` now matches `cpp extract` on every in-range fixture in
`../extract/outlog/stream-input/*.json`.

### Milestone H: Kernel Port — done

Mechanically translated `reference/` into a one-thread-per-candidate CUDA kernel under
`src/kernel/` (`world.cuh`, `piston.cuh`, `simulator.cuh`, `gpu_kernel.cu`) plus a host driver
(`src/host/main_gpu.cu`), built with `build-cuda.bat` (nvcc + MSVC host compiler). Two structural
changes from `reference/`, neither a game-logic change - see world.cuh's and simulator.cuh's file
header comments for the full reasoning:

- `std::vector`/`std::array`-of-vector queues become fixed-capacity C arrays + counts, and large
  scratch buffers that were function-local in `reference/` (e.g. `updateEntities()`'s moving-block
  snapshot) became `Simulator` member fields instead - a GPU thread's stack is a few KB by
  default, so anything bigger has to live in the same per-candidate global-memory allocation as
  everything else, not on the call stack.
- Exceptions become a sticky `error_` flag checked once per tick boundary instead of unwinding
  mid-tick - behavior-identical for every fixture that never crosses a capacity ceiling (all of
  them below), since a would-be exception's outcome (abort, report `simulation_error`) is
  preserved either way.

Two real correctness bugs were caught and fixed during this port via the same tick-by-tick
state-key diffing technique used above, applied against `reference/` instead of cpp extract:

1. The cycle-detection hash table's insert path recorded a new state as `used` but never wrote
   its `tick`/`anchor` fields, so a later match returned tick 0 / anchor `{0,0,0}` instead of the
   state's real first-seen tick/anchor - silently corrupting `start`/`period`/`shift` on any
   candidate whose cycle didn't start at tick 0.
2. `clearMovingAt()` used swap-with-last removal instead of `reference/`'s order-preserving
   `vector::erase()`. The moving-block bucket's iteration order feeds `settleMovingBlock()`'s
   side-effecting neighbor-update cascade in `updateEntities()`, so reordering it changed which
   neighbor update ran first and diverged the simulation trajectory for any candidate with more
   than one simultaneously-moving block (i.e. most real sticky-piston flying machines).

Verified byte-identical (after stripping the host-only `elapsedNs` timing field) against
`reference/` on:

```text
genetic_ml/data/compact-working/flyers.data: 1584/1584 candidates identical (real GA population)
../extract/outlog/stream-input/dont-fly/: 23/23 fast-resolving fixtures identical
```

Not yet exercised on the kernel path: the slow/large `dont-fly/` fixtures (multi-thousand-tick
runs, `tank.json`'s 4731 blocks) - naive one-thread-per-candidate dispatch on this machine's
WDDM-mode GPU hits the driver's ~2s kernel timeout (TDR) on those before finishing. Not a
correctness defect (the two bugs above were both real logic errors, not a timeout artifact).
Milestone I's persistent-kernel/work-queue dispatch turned out NOT to close this gap - see its
entry below for why (it fixes batch-level imbalance, not one candidate's own serial latency).

### Milestone I: Work-Queue Dispatch — done

`gpu_kernel.cu`'s `simulateQueueKernel` replaced Milestone H's fixed one-thread-per-candidate
assignment with a small, persistent pool of worker threads that pull the next candidate index off
a shared `atomicAdd` counter until the queue is empty, instead of every candidate getting its own
thread for the whole launch. Two supporting changes, neither a game-logic change:

- Worker count (`pickWorkerCount()`) is now chosen from free device memory
  (`cudaMemGetInfo`) instead of equal to the batch size, since a worker now processes many
  candidates back-to-back rather than exactly one - this is what makes it possible to safely
  raise `main_gpu.cu`'s default `--batch-size` from 16 (Milestone H, memory-limited) to 256.
- `FixedWorld::clearLiveCells()` resets a worker's cube between candidates by walking just
  `entries_` (the live block list already maintained incrementally by `setBlock()`) instead of
  the host-only one-shot bulk `cudaMemset` Milestone H relied on - that memset only zeroes each
  worker's cube slice once, before its first candidate, not between every candidate a worker
  processes.

Re-validated byte-identical against `reference/` (correctness unchanged, per the milestone's own
requirement) on the same fixture sets as Milestone H:

```text
genetic_ml/data/compact-working/flyers.data: 1584/1584 candidates identical, ~1.4s total
../extract/outlog/stream-input/dont-fly/: 23/23 fast-resolving fixtures identical
```

Also tested, and instructive rather than a regression: batching in the large/slow `dont-fly/`
fixtures (multi-thousand-tick runs, `tank.json`'s 4731 blocks) still hits the Windows WDDM ~2s
kernel timeout (TDR) even in a batch with no 6000-tick-cap candidates at all (largest was 528
ticks) - work-queue dispatch fixes *batch-level* load imbalance (idle lanes waiting on a batch's
slowest member), not a single candidate's own serial tick-loop latency, which is where these
particular fixtures spend their time (`stateKey()` is O(live block count) per tick, and these
fixtures have thousands of live blocks). See README.md's Milestone 11 note for what closing that
gap for real would take - not attempted here, out of this milestone's scope.

### Milestone J: Throughput Benchmark

### Milestone J: Throughput Benchmark — done

`genetic_ml/benchmark_gpu_vs_pool.py` times the exact same `flyers.data` batch (12,800 real GA
candidates) through both `genetic_ml.simulator_pool.SimulatorPool` (12 `mcp1122_cpp_stream.exe`
worker processes - what `main_ga.py` actually uses) and `gpu_kernel_stream.exe` (one process,
Milestone I persistent work-queue dispatch), reusing each engine's own normal calling convention
rather than a special benchmark-only protocol.

```text
cpu pool (12 worker processes): 12800 results, 12800 ok, 7.17s, 1785.6 candidates/sec
gpu kernel (1 process, persistent work-queue dispatch): 12800 results, 12800 ok, 8.45s, 1515.2 candidates/sec
speedup (cpu_time / gpu_time): 0.85x
```

The answer, honestly: **no speedup on this hardware** - the GPU kernel is about 15% slower than
the 12-core CPU pool, not faster. Diagnosed, not just measured: `gpu_kernel_stream.exe` prints
its actual worker count per launch (`gpu_kernel: <n> candidate(s), <m> worker(s)`), which shows
only ~50-60 workers running concurrently for a 256-candidate batch - `pickWorkerCount()` is
memory-limited (~19.6MB/worker on a 2GB card with typically ~700MB-1GB free alongside the desktop
compositor), not compute-limited, so the kernel never comes close to this GPU's theoretical
concurrent-thread ceiling. Even so, ~50-60 GPU threads land in the same ballpark as 12 CPU
processes - the real story is per-thread cost, not thread count: a GPU thread executing this
branchy, pointer-chasing, warp-divergent tick loop (README.md's own upfront description of why
this workload doesn't parallelize *within* a candidate) is markedly slower per operation than a
CPU core, and an old, weak, WDDM-mode consumer card (GTX 960, 8 SMs, compute capability 5.2) has
less raw throughput to begin with than a modern 12-core CPU. This matches the milestone's purpose
exactly: it's the number that decides whether further kernel optimization is worth pursuing on
*this* hardware, and the honest answer today is that the memory-per-worker cost (shrinking
`FixedWorld::kExtent` or the `kMax*` capacity constants would raise the worker count) would need
to improve substantially before this port pays for itself over the existing CPU pool.
