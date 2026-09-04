# Verification

How we tell a **broken implementation** from a **model that is not learning**. Those look
identical from the outside, and `ALPHAZERO.md` Part 7 only covers the second.

## The governing principle, inherited

`util tools/check_state_builder.py` states it in its own docstring:

> *"both re-derived independently here from the raw event stream rather than imported from
> `state.py` or `check_log_completeness.py`, so a bug in one file can't hide behind another"*

Three existing scripts embody it — `compare_java_cpp_simlog.py` (C++ against an independent Java
engine), `check_log_completeness.py`, `check_state_builder.py`. **Differential testing against an
independent path to the same fact.** RL gym extends that rather than inventing a new approach.

The sharpest instance is worth copying wherever it fits: every `BlockStateChanged` carries the
**old** state it overwrites, so replaying and asserting the board already holds that old state
means *a single missing write cannot hide*. The log checks itself.

In practice this means **expected values in `test_unit/` are written out by hand** from
`block_registry.h`, the vanilla meta conventions, and the fixture files — never computed by
calling the thing under test.

## Where the tests live

`RL gym/test_unit/`. One folder, one command. Simulator-driven tests carry `@pytest.mark.slow`
and are **skipped, not failed**, when the C++ binary has not been built.

```
py -m pytest test_unit                 # everything      98 tests,  9.7s
py -m pytest test_unit -m "not slow"   # no simulator    78 tests,  0.17s
```

## Per component, what the oracle is

| component | oracle |
|---|---|
| palette | hand-written literals from `block_registry.h`, cross-checked against corpus counts |
| action enumeration | algebraic identities — cells x 48 + 1, one no-op per cell |
| encoding | struct literals for the wire layout, plus a round trip |
| hashing | invariants — translation, ordering, trigger, rotation |
| `sim.py` | **batch-independence** and determinism |
| the whole Phase A path | **the no-op identity test** |
| `record.py` | struct sizes against the C++ `static_assert`s; summary against the stdout verdict |
| `boards.py` | **the write stream checks itself** — every write states what it overwrites |
| `graph.py` (Phase B) | hand-written semantic assertions; there is no oracle |
| `net.py` (Phase B) | **batch-independence**, masking exactness, gradient flow, overfit-one |
| `search.py` (Phase C) | conservation laws, and the PUCT worked example as a literal test |
| the loop | recall@B **below random is a bug, not a modelling result** |

## The three tests that carry disproportionate weight

### 1. No-op identity (`test_identity.py`)

A placement setting a cell to the block it already holds must produce an identical verdict to
the unmodified machine — through the encoder, through the pipe, through the C++ engine, and
back. One assertion covering candidate construction, the `id | meta<<8` packing, trigger
placement, the stdin protocol, batch framing and `canonical_hash`, **with no expected values
written by hand**.

### 2. Batch-independence (`test_sim_protocol.py`, and `test_net_batching.py` in Phase B)

`main.cpp:57` `processStream` reuses **one `Simulator` instance** across the entire stream, so
cross-candidate state leakage is a live risk. It would be completely invisible: nothing crashes,
the labels are simply slightly wrong, forever. So a candidate is simulated alone and again
buried at position 10 of a long batch, and the verdicts must be identical.

The network half is the same shape: machine A alone versus A concatenated with B must give
**identical** outputs for A. That is the direct test of "no edge crosses between machines" and
"one summary per machine, never shared" — two failures that make training look *better* than it
is, then collapse on a machine evaluated alone.

### 3. The write-stream replay is self-checking (`test_boards.py`)

Every `BlockStateChanged` carries the **old** state it overwrites, so replaying and asserting the
board already holds that value means **a single missing write cannot hide** — the next write at
that cell would disagree. `check_log_completeness.py`'s REPLAY argument, reused unchanged.

This is on **by default in `boards.py`, not only in a test**: it costs one comparison per write
and it is the entire basis for trusting the result. `test_boards.py` also corrupts one event's
old-state field and asserts the check fires, because a self-check that cannot fail is decoration.

A second, independent confirmation: on a machine whose run is exactly one period, the replayed
board must equal the initial board translated by the summary's `netShift` — comparing the event
stream against the `RunSummary`, which different code computes.

**Measured: zero mismatches across all 46 fixtures that produce a record.**

This is what reversed point 27 and removed the last C++ change from the critical path — see
`ALPHAZERO.md`. It also removes the verification problem that change would have created: Java
emits no snapshots, so **nothing could have checked new snapshot code except a Python replay**,
which is what we have instead. The remaining C++ change, records to stdout (point 25), still
needs its own check when it happens — same candidate via file mode and stdout mode, identical
bytes — because point 25 keeps file mode precisely *because* the Java harness reads record files,
so the stdout path would otherwise be uncovered.

## What Phase A actually caught

Verification is only worth the words if it finds things. It found four:

| # | found | it was |
|---|---|---|
| 1 | `assert 1057 in states` failed | **a test bug** — 1057 is `piston \| 4<<8`; the fixture holds `sticky_piston \| 4<<8` = 1053 |
| 2 | truncated-record test read scrambled bytes | **a test bug with a product lesson** — the drain thread and the test were both reading one pipe, interleaving characters. `stdout` must only ever be read through the queue, so `read_raw_line` now exists and reading `process.stdout` directly is documented as forbidden |
| 3 | 212 sims/sec instead of ~1,000 | **a product bug** — a fixed 512 chunk meant 988 candidates became two chunks, so ten of twelve workers sat idle. Chunk size is now derived from the workload; 572 sims/sec |
| 4 | `python_overhead_ms` reported negative | **a product bug** — wall time was compared against the *sum* of `elapsedNs` across parallel workers. The diagnostic was removed rather than corrected, since the two are not comparable once workers > 1 |
| 5 | replaying the write stream matched perfectly on every fixture | **a design error** — point 27's premise did not hold, and a planned C++ change was unnecessary. Reversed; see `ALPHAZERO.md` |

Two of the four were mistakes in the tests themselves, which is the expected ratio and the
reason the expected values are hand-derived: a test that computes its expectation from the code
under test would have passed all four times.

## The honest gap

**The graph's semantic correctness has no oracle.** Invariants catch indices out of range and
wrong counts, never "I connected the observer to the wrong cell".

`test_graph_semantics.py` (Phase B) is the cheapest adequate answer: about a dozen hand-written
statements on `simple_observer_engine` — *"the observer at (1,1,2) facing south has exactly one
`observes` edge, and it targets (1,0,2)"* — rather than transcribing ~40 items and ~150 edges.

**Rule for later:** the first implementation becomes the oracle for the second. Do not write two
now. When `SLOWDOWNS.md` #3 forces the graph builder to be optimised, keep the slow version and
diff against it — the differential pattern applied to our own code.

## Benchmarks

`bench.py` records a JSON baseline and prints a comparison on later runs. **Never fails the
build** — a flaky suite gets ignored.

| number | assumption tested | measured |
|---|---|---|
| simulator calls/sec through the pool | ~1,000/sec | **572/sec** at 12 workers on 988 candidates |
| `elapsedNs` per candidate | ~1 ms | **0.26 ms** on the 6-block base machine |
| graph build ms/machine | `SLOWDOWNS.md` #3 | Phase B |
| forward pass ms/machine | — | Phase B |

Wall-clock throughput and the `elapsedNs` sum are **not** comparable once workers > 1 — the sum
counts every worker's time while the clock counts one — so per-candidate overhead is not derived
by subtracting them.

## Probes — two mechanisms `DEFERRED.md` does not specify

**The probe set is frozen once created.** Adding a probe makes every historical number
incomparable — you can no longer tell whether month 6 beats month 1 or was just measured
differently. New probes start a new named set with its own history.

**Probes are built outside the library's lineage.** A probe derived from our base machine would
very likely be rediscovered and quietly trained on. Building them from unrelated hand-made
machines makes the requirement structural rather than a blocklist that must work perfectly
forever; a hash blocklist stays as a safety net that should almost never fire.

## What runs when

| check | cost | cadence |
|---|---|---|
| `test_unit -m "not slow"` | 0.17 s | every edit |
| `test_unit` | 9.7 s | before every commit |
| mechanic probes | ~20 forward passes | every checkpoint |
| calibration table | reads the attempt log | every checkpoint |
| recall@B vs control | 2 x B simulator calls | daily |
| exhaustive k=2 ground truth | ~13 min at R=1 | once per test machine, permanent |
| trimming survival | 3 sims per discovery | continuous, 6% overhead |

## Alarms — only one kind can be automated

**Mechanical** (the loop has stopped working): success rate at zero, duplicate rate at
everything, examples-per-update in the thousands, per-answer accuracy flat early, crash counts.
**Thresholds that fire**, not dashboard numbers — nobody watches a dashboard for three days.

**Quality** (the loop works perfectly and produces rubbish): nothing automatic catches this. It
goes through the visualiser and needs a person — which is why point 28's slow, sampled,
parent-and-child publishing matters.
