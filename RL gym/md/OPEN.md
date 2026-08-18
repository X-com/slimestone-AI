# Open problems, known limits, and corrections

Deliberately unsolved. None of these block a basic working model, and guessing at them was doing more
harm than leaving them blank.

---

## Deferred

### Ranking which working changes are useful

The big one. Right now a change either works or it does not; deciding that one working change is
*better* than another needs a measure of usefulness, and that is complex enough to need tuning
against real behaviour rather than a formula picked up front.

Two traps found while trying:

**Counting moved blocks is counting blocks.** Every block in a valid flying machine moves — nothing
may be left behind. So any measure based on "how much moves" is a size measure wearing a quality
costume, and rewards making the machine bigger. That is exactly the trivial extension problem.

**A single scalar punishes the behaviour we want.** `pushed / pistons` looks obvious but:

| machine | pistons | pushed | score |
|---|---|---|---|
| start | 1 | 5 | 5.0 |
| add a block | 1 | 6 | 6.0 |
| at the limit | 1 | 12 | 12.0 |
| add one more | — | — | breaks (push limit) |
| add a piston + block | 2 | 13 | **6.5 — looks worse** |

So a single score would reject "add a piston so it can push more", which is the main example of a
good extension. A shelf/archive binned by `(pistons, volume)` and scored by blocks pushed avoids
this — a new piston lands on a different shelf and counts as progress rather than regression — but
this is not settled.

Also note `blockCount`, `maxPushGroupSize` and `pushLimitFailureCount` describe a machine but do not
judge it. They are diagnostics, not usefulness measures.

### Expect trivial results at first

With "does it still work" as the only objective, a block riding along genuinely works. **Triviality
is the absence of a usefulness objective, not a model failure** — the model optimises exactly what it
was asked. The GA experience will repeat on this axis; what changes is speed and waste, not taste.

Cheap partial detector, nearly free given the log comparison already planned: if the modified machine
has the **same period, the same shift, and every original block still behaves identically at every
tick**, the new block is pure cargo. Flag it. This does not rank good extensions against each other,
but it separates "carried along" from "changed what the machine does".

### Large machines

Everything in `DESIGN.md` assumes the whole machine and its whole cycle fit in the graph. That breaks
at `1797 onepointfive` scale — 27,974 blocks × 1797 ticks is roughly 50 million cell-moments.

Likely answer: show the model a **window around the change** rather than the whole machine, so input
size depends on the window instead of the machine. It can be validated against the uncropped answer
on machines where both fit.

Note that periodicity does not save you here. It compresses well when the cycle is short relative to
the run (`1 wide trencher`: period 20 over 1,126 ticks, 56× repetition) and not at all when it is
not (`1797`: period 1797 over 1797 ticks; `flying_dog`: 468 over 473).

### Edits larger than about 5-10 blocks

The architecture has no ceiling — "place one block or stop" scales to any length, and beam search
cost grows only linearly with depth.

What breaks is **coverage**. A fixed-width beam explores a decent slice at depth 5 and vanishingly
little at depth 50, while the space grows astronomically. Credit assignment degrades too (one reward
signal for 50 decisions), and a predictor trained on small edits drifts out of distribution.

The named path past this is **composition**: large modifications are not 50 independent choices, they
are a few recognisable sub-assemblies. Harvesting recurring successful patterns into macro-actions
turns a 50-block edit into ~5 macro-steps, which is a depth the search handles well. Those macros
should come from the system's own successes rather than being hand-authored.

Not designed yet. This is the main gap between "works on small edits" and "builds real extensions".

### Multi-block search details

Beam search over sequences with only the final structure checked is the plan, but untested. Open
question: if a 3-block edit works while its first 2 blocks alone do not, the predictor is being asked
about something it never saw in training. Either train on whole sequences, or accept it only ranks
the first step well and widen the beam.

### Divergence analysis

Comparing the sim-log before and after a change shows where the two runs first differ, which points
at what the change actually did. Useful as a debugging tool and possibly as a richer training target.

Alignment is only hard *after* the runs fall out of step, and post-divergence detail is not wanted, so
the hard part disappears. `util tools/compare_java_cpp_simlog.py` already diffs two logs tick by tick
and could be repointed from "two engines, same machine" to "same engine, two machines".

One subtlety: divergence is not failure. A successful addition still changes the log immediately —
there are new events for the new block. What matters is the first difference where something got
*worse*.

---

## Known limits

These are properties of the problem, not things to fix.

**Timing spreads.** A change can break the machine somewhere far away by shifting when things happen,
with no local sign of it. The model reasons from the *original* machine's behaviour about a machine
that does not exist yet.

**Some blocks only become pushable once something else is removed.** This cannot be seen in the
original run's log at all — it is a fact about a machine that does not exist. Learned only from
accumulated experiments.

Both mean **the model will be weakest on exactly the surprising cases.** Acceptable only because the
simulator always makes the final call and the model's job is ranking, not deciding.

---

## Corrections log

Recorded so they do not get reintroduced. All were mistakes made and caught during planning.

**Event kinds that do not exist.** `BlockLeftBehind` (`sim_event_log.h:46`) and `ComponentSplit`
(`:52`) are marked *"reserved, not yet emitted"* — they never fire. `PistonExtendBlocked` and
`PistonRetractBlocked` were deliberately removed (`simulator.cpp:1762`): a blocked push is **one**
`PistonMoveExecuted` record with `SEF_SUCCESS` clear and a `failureReason` set. That enum is the real
failure vocabulary — `PushLimitExceeded`, `ImmovableBlockInPath`, `NoSpaceToExtend`,
`BlockCannotBePushed`, `AlreadyInTargetState`, `NotPowered`, `OutOfBounds`. `BlockDestroyed` is real
but only fires for orphaned piston heads and unsupported rails.

**Why the old GA found only trivial extensions.** `population.py:11`'s `canonical_hash` dedupes on
*structure*. A machine plus one carried block hashes differently, so trivial growth was recorded as a
new discovery. The search was rewarded for getting bigger, not different.

**Single-block edits are enumerable; the full space is not.** 66 cells × ~30 types ≈ 2,000
combinations, about 100 seconds of simulation. Filling all 66 cells is 30^66, which is not. Brute
force is a *measurement* available in one small corner, never the strategy.

**A CNN cannot span a push group.** 3×3×3 convolution reaches one cell per layer; a 12-block push
needs 12. And a push group is a connected set, not a box, so a larger reach is still the wrong shape.

**Periods are not short.** Measured: `1797 onepointfive` has a cycle of 1797 ticks and 27,974 blocks;
`flying_dog` 468; `complex_machine` 288. An early assumption that cycles were ~10 ticks was wrong and
invalidated a compression argument built on it.

**Modern RL features do not solve sparsity.** Double/dueling/prioritised/distributional DQN and
friends buy roughly 2-10× sample efficiency — they improve the constant factor, not the order of
magnitude. A success rate of one in a million is an order-of-magnitude problem.

**Model-free RL wastes the simulator.** DQN learns by trial and error without a model of the world.
Here there is a fast, perfect, deterministic simulator, so search guided by a learned evaluator is
the right family.
