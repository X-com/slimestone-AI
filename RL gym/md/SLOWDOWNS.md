# Slowdown register

Training speed matters to this project, so every identified performance risk is tracked here rather
than discovered later.

| # | where | cost | status |
|---|---|---|---|
| 1 | file creation, ~1,000/sec | likely the current binding limit on Windows | **fixed** — records to stdout |
| 2 | Python event replay for per-tick state | ~1 ms/machine; ~10 s per 10,000 | **fixed** — C++ emits snapshots |
| 3 | **graph construction in Python** | ~6,000 cell items + ~600 event items + links, per machine | **OPEN — main remaining hotspot** |
| 4 | simulation itself | ~1,000 machines/sec measured | fundamental; 12 parallel processes |
| 5 | cache misses forcing re-simulation | ~1 ms each | controlled by cache size |
| 6 | model updates on GPU | 10-50 ms each | runs concurrently with simulation (point 22) |
| 7 | `canonical_hash` per candidate | sha256 over sorted blocks | small but runs on every candidate |

---

## 1 — File creation (fixed)

With one record file per candidate at ~1,000 candidates/sec, Windows filesystem churn was likely the
real limit rather than simulation. Also a disk-wear concern.

**Fixed by changing the C++ to write records to standard output** — see point 25 in `DECISIONS.md` for
the framing, binary-mode and error-path requirements.

## 2 — Python event replay (fixed)

Rebuilding per-tick board state by replaying events in Python costs roughly 1 ms per small machine,
about 10 seconds per 10,000 machines, repeated on every pass.

**Fixed by having the C++ emit per-tick snapshots behind a flag** — see point 27. The stronger argument
was correctness rather than speed: the Python replay is a reimplementation of logic the C++ already
performs, and two implementations must be kept in step.

## 3 — Graph construction in Python (OPEN)

**The one to watch.** Building the graph means creating thousands of Python objects per machine — cell
items for every cell at every tick, event items, and the links between them — across thousands of
machines.

This is exactly the kind of cost that stays invisible on a small test and dominates at scale.

Escalating options, in increasing order of effort:

1. **Build the graph directly into numeric arrays** rather than Python objects
2. **Cache the built graph rather than the record**, so each machine pays once — already decided in
   point 19, and it removes the cost entirely for anything seen before
3. **Have the C++ emit the graph structure itself**, the same move already made for per-tick snapshots

**Measure this early.** If it dominates, the graph builder's shape should be decided with that in mind
rather than rewritten afterwards.

## 4 — Simulation (fundamental)

Measured at ~880-1,130 candidates/sec through the batched stdin protocol, across a 12-process CPU pool.

**The GPU port is not the lever.** Its own README reports the kernel ~15% *slower* than the CPU pool on
a GTX 960, memory-limited to ~50-60 concurrent workers rather than compute-limited. Redstone is a
branchy sequential state machine, so the only parallelism is across candidates.

## 5 — Cache misses

A machine record is regenerable by re-simulating at roughly 1 ms. Controlled by cache size, trading
storage against simulation throughput. No decision needed beyond sizing it.

## 6 — Model updates

10-50 ms each on GPU. Does not compete with simulation, which is on CPU — point 22 decouples them
entirely so neither waits for the other.

**Required diagnostic: examples generated per model update.** Decoupling allows the two sides to drift
apart. At 1,000 examples per update the model is barely learning from what it generates, and this looks
identical to healthy operation in the logs.

## 7 — Duplicate hashing

`canonical_hash` is sha256 over the sorted block list. Small per candidate, but it runs on every
candidate before simulating, so it is on the hot path. Worth measuring if the loop is ever throughput-
bound for no other visible reason.
