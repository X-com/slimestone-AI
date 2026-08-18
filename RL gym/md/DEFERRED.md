# Deliberately unsolved

Problems left open on purpose, because deciding them now would mean guessing before there is any data.
None of them block a basic working model. **Do not treat anything here as decided.**

See `../OPEN.md` for the higher-level deferred items (usefulness ranking, large machines, multi-block
search).

---

# ⚠ MUST EXPAND — mechanical probe tests (point 30)

**A much larger set of tests is needed, and how the existing hand-made tests connect to model
verification is not yet worked out.**

## What a probe is, and why aggregate scores cannot replace one

Every other measurement is an average over many moves. A probe asks **one question whose correct answer
is known exactly**, chosen so that a model which understands and a model which learned a rough
association answer differently.

The case from point 11: a model that understood the push limit and one that learned "adding blocks near
pistons is risky" score almost identically across mixed moves, because the association is usually right.
**They only separate at the exact point where the association breaks.**

## The shape that makes a probe informative: a sharp boundary

Arrange a working machine so one piston pushes exactly 11 blocks, then sweep group sizes 9 through 14
and watch where the model's confidence drops:

- sharply between 12 and 13 → **it knows the rule**
- gradually from 10 onward → it learned "bigger groups are riskier" — an association, not the rule
- not at all, or at 15 → the mechanic was never learned

No aggregate number distinguishes these three.

## Candidate mechanics

- **push limit** — the sharp-boundary case above
- **immovable blocks** — obsidian in the path fails at *every* group size, a flat line where the push
  limit shows a step; a model that only learned "big groups fail" gets small sizes wrong
- **slime dragging** — a block next to a *moving slime* is carried; next to a *moving stone* it is left
  behind and the machine is invalid. Identical placement, opposite outcome, difference is the neighbour
- **glazed terracotta** — `PushOnly`: pushable but never pulled or dragged, so placing it where it must
  be pulled back fails while stone succeeds. Tests block-specific behaviour rather than "solid is solid"
- **quasi-connectivity** — a redstone block *above* a piston activates it, *beside* it does not. Two
  placements one cell apart, opposite results. Non-local and non-obvious, so probably the strongest
  single test of real understanding

## Two constraints that would silently invalidate a probe

**Probes must themselves be working flying machines.** The model's input is built from a simulated
machine's record and it has only trained on machines that cycle. A bare piston pushing a column is
outside anything it has seen, so a wrong answer would prove nothing. A probe is therefore *a real
working machine arranged so one piston sits at eleven*, not a piston and eleven blocks in isolation.

**Probe machines must never be trained on** — including as descendants in the library. Otherwise the
model may have memorised them and the probe measures nothing.

## Connecting the existing hand-made tests — UNRESOLVED

A large set of hand-made tests for various constraints and behaviours already exists or is being built.
How they serve model verification is not yet settled. What is known:

**Their original purpose is different.** Those tests verify the *simulator* — does it behave like real
Minecraft, with ground truth from the Java engine. Model probes verify the *model* — does it predict
what the simulator will do, with ground truth from the simulator. The transfer is not automatic.

**What would make them transfer.** Any small fixture can be given exact ground truth by brute-forcing
its modifications through the simulator, so ground truth is never the obstacle. The obstacles are the
two constraints above: is the fixture a working flying machine, and can the behaviour it tests be
expressed as a *modification* to it rather than as a property of the fixture itself.

**The most valuable thing to add: a mechanic tag per test.** If each fixture records which mechanic it
exercises, model accuracy can be reported **per mechanic** — "understands push limits, does not
understand quasi-connectivity" — instead of one number that hides which parts were learned. This is the
pattern `transformer_gym/mechanic_fixtures.py` used, and it is what makes a large test set diagnostic
rather than merely large.

**Still to work out:** which existing tests are working flying machines; which tested behaviours can be
posed as modifications; what tag vocabulary to use; and how many probes per mechanic are needed before
the measurement is trustworthy rather than noisy.

---

# ⚠ MUST REVISIT — which machines the library keeps and samples from

## The problem

Storage is solved (point 19), but a second problem is not. **Every machine in the library is a
descendant of the base machine.** Grow a million of them and there are a million elaborations of one
design — volume is enormous, variety is not. A model trained on them learns that design family well and
may still fail on anything structurally different.

So training batches should be drawn with a bias toward **variety**, which requires deciding when two
machines count as meaningfully different. That is the hard part, and it is too uncertain to fix now.

## Candidate measures of "meaningfully different"

Proposed, to be weighted and tested later:

- **number of pistons**
- **average blocks each piston pushes**
- **number of observers**
- **number of pistons triggered by redstone blocks** (as opposed to observer-driven)
- **use of rarer blocks that serve a real function** — e.g. glazed terracotta, which is `PushOnly` in
  `block_registry.h`: pushable but never pulled by a sticky piston and never dragged by slime. That
  one-way behaviour is a genuine mechanism, so its presence signals a more sophisticated design rather
  than decoration.

Also available from the footer without extra work: period, net shift, block count, the mix of event
kinds.

## Prerequisite: trim useless blocks before measuring

**None of the measures above are trustworthy until dead blocks are removed.** A machine carrying five
blocks that do nothing would otherwise be counted as larger and more elaborate than an equivalent
machine without them, and the archive would fill with padded variants of the same design — the exact
failure being guarded against.

**How trimming would work.** For each block, remove it and simulate. If the machine still works
identically, the block was doing nothing — drop it. Repeat until nothing more can be removed.

**Cost — and this is worse than one-at-a-time removal suggests.** Trimming is **combinatorial, not
linear**. Knowing which blocks are genuinely doing nothing means testing *subsets*, because removing
block A alone can break the machine and removing B alone can break it while **removing both is fine**.
One-at-a-time removal can never find that pair.

For a k-block set the exact answer needs **2^k − 1** simulations:

| blocks | subsets |
|---|---|
| 1 | 1 |
| 2 | 3 |
| 3 | 7 |
| 4 | 15 |
| 5 | 31 |
| 10 | 1,023 |

Greedy one-at-a-time is k simulations and produces a machine that is **small but not necessarily
minimal**. Whole-machine trimming is subsets over every block, not just recently added ones, so it is
far beyond exact treatment for any real machine.

**Escalation rule:** exact subsets while the set is small (≤4-5 blocks, so 15-31 simulations, affordable
at ~1,000/sec); greedy above an adjustable threshold. That threshold interacts with point 24's size
weights — as larger modifications become common, the cost of exact measurement is what pushes onto the
greedy path.

## Open sub-questions

- Which measures form the bins, and how coarse each one is. Too few axes and genuinely novel designs get
  discarded for being the same size; too many and every machine lands in its own bin, so the variety
  bias does nothing.
- Whether to keep one machine per bin or several.
- Whether "training impact" (oversampling examples the model gets most wrong) should play any role.
  Noted as considered and *not* chosen as the primary axis: it tends to oversample oddities, and it does
  nothing about variety, since the highest-error machines can all still be relatives.
- Whether trimming should run on every archive candidate or only periodically.

---

# ⚠ ALSO REVISIT — which machine to modify next

**Same problem family, equally unsettled. Default chosen only so the loop can run.**

Once the library holds a hundred machines, every round must choose one to work on. **Default: weighted
by past success** — machines that previously yielded working modifications are chosen more often. This
is a **hyperparameter**, not a fixed rule.

Why it needs revisiting: success-weighting **reinforces whatever the early rounds happened to find**. If
the first few lucky discoveries came from one design, effort concentrates there and the library tunnels
into that family — which is precisely the variety failure described above. The default is chosen for
being reasonable to start with, not for being right.

Alternatives to test later:
- **newest machine** — drives deep and reaches larger machines fast, but tunnels into one line of descent
- **uniformly at random** — spreads effort, including onto machines already shown unpromising
- **weighted toward under-explored machines** — favours those with few attempts, spreading coverage
- **weighted by variety contribution** — favours machines in sparsely-occupied bins, directly attacking
  the variety problem, but depends on the binning question above being settled first

Concrete illustration of why there is no obvious answer: machine A produced eight working modifications;
machine B produced none in twenty attempts. Both sit in the library. B might be one move away from
something, or genuinely exhausted, and nothing currently distinguishes those two cases.

This choice is **what determines whether the library spreads out or drills down**, so it should be
revisited together with the binning question rather than separately.

---

# Noted for later: the risk asymmetry in any future deduplication work

If any stronger equivalence check is ever added beyond `canonical_hash`, the two failure directions are
**not** symmetric:

- **missing a duplicate** costs one wasted simulation — trivial
- **wrongly declaring two different machines identical** means a valid machine is skipped and **never
  discovered**, silently and permanently

Any future work here must be biased heavily toward under-merging. Deferred; not to be attempted without
a way to verify the equivalence is real.

Rotation was considered and rejected for exactly this reason — see point 23. Rotating a machine changes
which block is the anchor (`simulator.cpp:2052`) and changes neighbour update order, so rotations
genuinely can behave differently and are treated as distinct machines.
