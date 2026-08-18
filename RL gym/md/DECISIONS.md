# The 33 points — what was decided

What each point asks is in `QUESTIONS.md`. Terms are defined in `README.md`.

---

## Point 1 — Numbers stored per item

Include all three kinds: read off the layout, read off the run record, and looked up from block type
using the simulator's existing `BlockData` table. The looked-up values are **supplied from a preset
table rather than learned**. The list will be expanded once real training shows what is missing;
retraining then is accepted.

**Required safeguard:** save the description of what each number means alongside the trained model and
refuse to load a mismatched one, so a stale saved model fails loudly instead of silently training on
shifted columns.

## Point 2 — Block identity

Block type and facing direction stored **separately**, so what is learned about a piston applies
regardless of which way it points.

Rails stay in the vocabulary but no design effort goes into rail-specific behaviour, and they are
excluded from early training. The unlisted block id fallback (`stone`, `glass` have no
`block_registry.h` entry) is logged as a **separate simulator issue**, not part of this work.

## Point 3 — Position

**No coordinates.** Instead supply the specific geometric facts that matter, as ordinary numbers on
empty cells: whether the cell lies in some piston's push path and how far along; that piston's current
push group size; whether the cell is next to a block that moves; whether it is next to slime that
moves.

Chosen because it generalises better — with no position the model **cannot memorise that a particular
location tends to work**, and these facts carry over unchanged to a machine of any size.

## Point 4 — Relationships between items

**Seven kinds:** same block at next tick; touching; same push group; slime dragging; what caused what;
redstone power reaching a piston; quasi-connectivity.

The observer relationship already exists in `encode.py:180`, computed **geometrically from the
observer's facing** rather than from the run record — so it correctly covers an observer watching a
cell where nothing has happened yet.

Shared transformations with one learned importance value per relationship kind, matching the existing
pattern at `model.py:103`.

## Point 5 — Describing a machine that was never simulated

The description is of an **already simulated machine, with notes attached** saying "assume a block of
this type is here". Notes are added **alongside** the existing description of a cell, never replacing
it — every relationship around that cell was worked out for the machine as it actually was, so
overwriting would leave the description contradicting itself.

Blocks already chosen during an attempt are written in as notes; the next block to choose is read out
as scores over every cell and block type. **One run of the model per block placed, not per option
considered.**

**How many blocks between simulations is a single adjustable number**, not an architectural choice — 1
means every intermediate is verified, higher means longer unverified attempts. It starts at 1 and
rises. This trades **reach against feedback**: verifying every step can only reach machines having a
path of individually valid steps, while allowing broken intermediates opens more paths but delays all
feedback to the end.

Three requirements follow, cheap now and expensive later:
- the model must accept **any** number of notes, so the marking is an ordinary per-cell value rather
  than a single dedicated slot
- nothing may assume the base description is freshly simulated
- nothing may assume intermediate machines are valid

**Record how many notes were attached with every training example**, so accuracy against note count
can be plotted and the usable limit measured rather than guessed.

## Point 6 — How an item combines information from its neighbours

**Choice C**: how much a neighbour counts depends on **what that neighbour is currently doing**, not
only on the kind of connection joining them.

Chosen because update order matters. Pushing several blocks produces different results depending on
which block is triggered first, and that cannot be captured by weighting purely by connection kind.

**This forced a granularity change the earlier design would have made impossible.** Measured on real
logs: a single block at a single tick has 3-4 events on average and up to **13**. In
`simple_observer_engine` only one block-at-a-tick has a single event. Collapsing those into one item
destroys the ordering entirely, so Choice C would have had nothing to weigh.

Inspecting the busiest slot (block at (1,64,1), tick 8, 13 events) shows a **genuine causal chain
inside one tick**, not bookkeeping noise: a block settles at order 67 → notifies the piston at order 69
→ piston queues at order 70 → move executes at order 77 → destroys a block at order 75.

So the graph gains a second kind of item:
- **cell items** — one per cell per tick, holding what block is there; candidate placements attach to
  these. Economy available: only create these at tick 0 and at ticks where the cell changed.
- **event items** — one per event, carrying its order number (`globalSeq` / `activationSubtick`,
  already recorded by the simulator)

Three links: each event to the cell-and-tick it happened at; **each event to the next event in the
same tick**, which is what makes within-tick order visible; and each event to what caused it via
`actorKey`.

Ordering *across* ticks needs nothing new — the time links from point 9 cover it.

## Point 7 — How many rounds of information sharing

An **adjustable setting**. Each round moves information one connection, so the round count is the
furthest the model can relate two things — the same limit that ruled out the grid-based approach.

Push groups are no longer far (all members connect directly, so a 12-block group is one step), but
**effects routinely span several hundred ticks**, which stepping tick-by-tick could never reach. Both
fixes are adopted:

- **time-skip links** — each block links to itself 1, 2, 4, 8, 16, 64, 256 ticks later, so tick 5
  reaches tick 300 in about six steps instead of 295
- **one summary item** that every item connects to, making anywhere reachable from anywhere in two
  steps; coarse, but right for machine-wide effects like the cycle collapsing

## Point 8 — Model size

Adjustable, starting at **128 numbers per item**.

Diagnostic: poor on training machines means too small; good on training machines but poor on an unseen
one means too large for the data. **Both numbers must always be reported** or the two failures are
indistinguishable.

## Point 9 — Time direction

Forward-in-time and backward-in-time are **two separate relationship kinds**, each with its own
learned importance, so which one the model actually relies on can be read off after training rather
than guessed now.

Backward is legitimate here: the model reads a complete recording of a machine that already ran, so it
is not forecasting and cannot cheat by using the future.

## Point 10 — Construction

Learned multipliers are **shared across all rounds** — one rule applied repeatedly, which matches the
domain, since Minecraft's rules are identical at every tick. The decisive advantage given effects
spanning hundreds of ticks: **the round count can be raised later without retraining**, because there
is one rule and you simply apply it more times. Separate per-round multipliers would fix the count at
training time.

Taken as standard defaults, no deliberation: each round adds to the previous result rather than
replacing it (also the remedy for items blurring together over many rounds); numbers kept in a stable
range between rounds; a standard shaping function.

## Point 11 — Which answers the model produces

Three are built:

- **A — will it still work** (one number). Mandatory; nothing can be ranked without it, and it is what
  the stop decision reads.
- **B — which blocks start behaving differently** (one number per block), from comparing the before and
  after record files. Included for signal density: 2,500 simulations teach 2,500 bits if the only
  answer is yes/no, versus ~125,000 numbers with per-block answers. It is also *more specific* — "you
  predicted 8 blocks would move, only 5 did" points at the mistake, where one bit does not.
- **D — what kind of failure** (one of the seven `failureReason` values). Nearly free, and the answer
  changes what to try next: `PushLimitExceeded` means add a piston, `ImmovableBlockInPath` means pick a
  different block.

**Failure location is NOT trained (option C2).** It is obtained by inspecting which parts of the input
the model relied on. Rationale: with answer A alone, a model that understood the push limit and a model
that learned "adding blocks near pistons is bad" are indistinguishable — both say "breaks" and both are
right. Looking at where the model was attending separates them at no cost in capacity. C2 was chosen as
the more expandable option; training it (C1) remains available later if the loop needs the model to
*report* and act on a location.

## Point 12 — Shape of the placement scores

A cell is many items (one per tick), so the per-tick items must be reduced to one score per placement.
**Learned weighting**, because neither simple option is adequate: averaging divides the one tick that
matters by the cycle length (a cell a piston sweeps through at tick 8 has its whole story diluted
tenfold), and taking the largest cannot represent a cell whose outcome depends on *two* ticks together
— pushed at tick 3, then interacting with a second piston at tick 8.

Resolved without discussion: the **stop** score comes from the summary item added in point 7, since "is
this modification finished" is a whole-machine judgement. Moves that change nothing (setting a cell to
the block it already holds) must be excluded or the model learns they always succeed — true and
useless. `BLOCK_PISTON_HEAD`, `BLOCK_PISTON_EXTENSION` and `BLOCK_LIT_REDSTONE_LAMP` are excluded as
not directly placeable.

## Point 12b — Facing in the output

**Every facing is its own option.** For 50 cells: 9 non-directional types plus 3 directional types × 6
facings = **1,350 scores**. Trivial even at 200 cells (5,400).

Facing is not a detail — a piston at (2,64,3) facing west pushes into the machine and may break it,
while the same piston facing east pushes empty air and becomes dead weight. Same cell, same type,
different machine.

The apparent objection — that enumerating six facings undoes point 2's separate type-and-facing input —
is wrong: **the sharing happens in the input and the model body**, and all six facing outputs read from
the same accumulated numbers for that cell. Only the final "west versus east here" step is per-facing,
which is a genuinely different question each time.

Facings that look useless are deliberately **not** masked, since deciding which are pointless is the
design knowledge the model should learn.

## Structure clarified before points 13-16

One run takes the original machine's record plus zero or more notes, and produces all at once:
**A/B/D about the noted machine as it stands**, and **E, the ~1,350 placement scores for a hypothetical
next move**. E can score every option from one run precisely because none of them are notes yet. Commit
to one, it becomes a note, and the next run's A/B/D describe the machine including it.

## Point 13 — Combining the answers' wrongness

All weights are **tunable settings**, deferred.

The problem they must solve: answers contain wildly different counts of numbers — A is 1, B is ~50, D
is 1, E is ~10 labelled. Summed naively, **B takes over 80% of the training effort purely by having
fifty numbers**, while A and E — the two the model exists for — get under 18% between them. The visible
symptom is a model that describes changes well while its ranking never improves, easily misdiagnosed as
needing more data or a bigger model.

Fix in two steps: average *within* each answer to remove the counting accident, then weight
deliberately.

**Mandatory regardless of weights: track each answer's accuracy separately**, since a bad balance is
invisible in a single overall number and shows up as one answer plateauing while another keeps
improving.

## Point 14 — Most changes break the machine

**All three mechanisms are built and kept**, with comparison-based as the default.

At a ~2.4% success rate (12 of 500), answering "breaks" every time scores 97.6% and finds nothing. This
is not only a measurement problem: answering the base rate for everything is a genuinely low-wrongness
resting place that training resists leaving, since early attempts to distinguish moves initially make
wrongness worse.

- **comparison-based** — train that a working move must score above a failing one. 12 successes × 488
  failures = 5,856 pairs, every one balanced by construction. Matches what the search needs, since only
  the *order* is ever used.
- **absolute with rare-class weighting** — needed for answer A, since the stop decision reads an actual
  value and "confident enough" has no meaning for a relative score. **The multiplier must be computed
  from the observed success rate, not fixed**; if the true rate is 0.4% rather than 2.4%, a multiplier
  of 40 is six times too small and the trap remains.
- **balanced batches** — an independent sampling setting.

Keeping both terms on the same output is better than either alone: the absolute term keeps scores
interpretable as probabilities (what point 33 needs) while the comparison term optimises order, **and
it is the fallback when a machine has zero successes** — no pairs can be formed at all, but the
weighted absolute term still learns.

## Point 15 — What is stored per simulation

Labels are extracted immediately and the raw record discarded. **~20 bytes per attempt versus a 30 KB
record — about 1,500× smaller**, since the model's input comes from the *original* machine's record and
the modified one is only needed to derive truth.

Original machines' records are kept permanently (32 machines ≈ 1 MB). Retention of modified records is
an **adjustable setting, default 1 in 100,000, with no permanent value anywhere** — and it must be
changeable **while a run is in progress**, so a new label can be developed by turning the rate up for
an hour and back down, rather than restarting the loop.

A separate small most-recent window (a few hundred records, ~6 MB) exists for debugging, which the
sparse permanent sample cannot serve.

Accepted cost, stated plainly: inventing a new label later means re-simulating.

## Point 16 — Batching

**Concatenate, never pad** — padding to the largest wastes 60-80% when sizes range from ~600 to ~6,200
items, with no compensating benefit.

The grouping rule is a **setting** (default: fill to a fixed total item count, for predictable memory).
It stays freely adjustable only because the model is made indifferent to it, which requires: nothing
assuming a fixed machine count or item count, and **every item carrying a label saying which machine it
belongs to**.

That label is what allows averaging *within* each machine and then across machines — so a machine
contributes equally regardless of its size or its batchmates. Without it, the same machine contributes
100% of a batch's weight when alone and 1/30 when batched with twenty-nine others, which makes training
results depend on the grouping rule and locks it in.

**Hard rule, not a setting: one summary item per machine, never shared across a batch.** A shared
summary lets information leak between machines in two steps, which makes training look better than it
is and then fails silently on a machine evaluated alone.

## Point 17 — Dividing data for testing

Not a problem at this stage. **Train on one machine**, then test generalization on a **separately
handmade** machine. No splitting machinery, rotation, or duplicate grouping. The fixture library can be
expanded when it matters.

(Checked while deciding: only one exact structural duplicate exists in the current fixtures —
`test_flyer_21` and `test_flyer_22` share a `canonical_hash`. `test_flyer_31`-`34` match on size and
timing but are structurally different. Relevant only if a fixed multi-machine split is ever used.)

## Point 18 — Training procedure

Adjustment method, step size and its decay are standard settings. The one real decision is stopping:
**keep training past the first downturn and remember the best point**, since the held-back measure is
noisy and a dip is often a wobble rather than the real turn.

Stopping is measured on **moves held back from the training machine itself** — deliberately the "leaky"
split rejected in point 17, because it answers a different question. It cannot show generalization to
new machines; it can show when the model stops learning and starts memorising this one, which is all a
stopping rule needs.

**The handmade test machine is never used for stopping** — spending it on a decision destroys the only
honest generalization measure there is.

## Point 19 — Retraining as data arrives

**Continuous updates drawing from the whole history**, not only recent results.

The failure being avoided is specific: the model learns "obsidian in a piston's path fails", so the
search stops proposing obsidian, so no new obsidian examples arrive, so the lesson fades, so it starts
proposing obsidian again — and can oscillate indefinitely.

Additionally **retrain from scratch periodically** (e.g. daily) as a check. If the fresh model is
noticeably better than the continuously-updated one, something has accumulated in the continuous path.
This is cheap relative to simulation and is the only way to notice that class of problem, which
otherwise looks like an unexplained plateau.

**Storage is not the constraint it appears to be.** Labels are ~20 bytes (a million examples = 20 MB,
keep forever). Machine *definitions* are ~800 bytes. Machine *records* are ~30 KB — the only large
item, and they are **regenerable**: keep definitions, cache records, re-simulate on a cache miss at
roughly a millisecond each. No rule about which machines "deserve" keeping is needed, which is better
than any selection rule since every selection rule is a guess.

## Point 20 — How often the model does not take its own best choice

**Sample from the model's own scores rather than always taking the maximum**, controlled by one
adjustable number spanning the whole range: very low behaves greedily, middle picks proportionally to
the scores, very high approaches uniform random. Start high (early scores are meaningless anyway) and
lower it as the model becomes worth trusting.

The failure this prevents is self-reinforcing and invisible: the model sees a few early examples where
slime near a piston broke the machine, scores all such placements low, so they never reach the top, so
they are never simulated, so no new evidence about slime ever arrives. **The belief becomes permanent
regardless of whether it is true**, and the model looks confident and consistent throughout.

Uniform-random exploration was rejected as the default: at a ~2.4% success rate a random move is almost
certainly a failure, so it spends simulations with no reason behind them. Sampling from scores
concentrates exploration where the model sees *some* chance. (Uncertainty-directed exploration is more
efficient in principle but needs the model to express uncertainty, which is not planned.)

## Point 21 — How many moves get simulated

Note first that "how many options are judged" is not a decision: one model run scores all ~1,350 moves,
so judging one costs the same as judging all. The only real cost is **how many get simulated**.

Both the simulations-per-round and the beam width are adjustable, defaults around 10-20. Beam width K
costs roughly 2K model runs (cheap) and **K simulations** (the real cost).

The trade-off is not cost but **how much is learned per simulation**. Twenty moves chosen by the same
model before any results return can test essentially one belief twenty times — the model thinks a region
looks promising, picks twenty moves there, and they all fail, teaching roughly one thing. Simulating one
and updating would have sent the other nineteen elsewhere. Sampling from scores (point 20) softens this
by spreading the picks.

**Simulations serve two purposes that want opposite things** — finding good machines wants the
top-ranked candidates; generating training data wants variety including low-rated moves. These are split
by **one adjustable number**, fixed for now, rather than being blended invisibly into the exploration
setting.

## Point 22 — Retraining cadence during the loop

**No cadence — decouple simulation and training entirely.**

They use different hardware and do not compete: simulation is the CPU pool (~1,000 machines/sec),
training is the GPU. Strict alternation leaves one idle at all times (~70 ms per round versus ~50 ms
running concurrently).

So: the simulation side picks moves with whatever the latest model is and writes labels to a shared
store, as fast as the CPU allows. The training side draws from the store and publishes new versions, as
fast as the GPU allows. Neither blocks the other, and the simulations-per-round number from point 21 no
longer silently determines the training rate.

**Cost of this:** the model version that chose a move is no longer pinned. Mitigation is four bytes —
**record the model version alongside each label**, so which version made a choice stays reconstructible
even though an exact replay is not.

**Required diagnostic: examples generated per model update.** Decoupling allows the two sides to run at
wildly different rates. At 1,000 examples per update the model is barely learning from what it generates
— effectively random search with a frozen model — and this looks identical to healthy operation in the
logs.

**Rate mismatch policy:** let simulation run ahead early, when the model's choices are not worth much
anyway and broad data collection is what is wanted; throttle later once its choices beat random.
Adjustable.

## Point 23 — Recognising duplicate machines

Hash each candidate with `genetic_ml/population.py:11`'s existing `canonical_hash` **before
simulating**. It is translation-invariant (correct — a flying machine is the same machine wherever it
is) and includes the trigger position (also correct — identical blocks with a different trigger is a
different machine).

Why it matters more as modifications grow: the same machine can be built in many orders, and the count
is factorial — 2 blocks gives 2 orders, 3 gives 6, 5 gives **120**. Beyond the wasted simulation, the
worse effect is that **training data becomes skewed**, since machines reachable by many routes get
stored many times. Nothing about being easy to stumble into makes a machine more important.

Store hash → **result**, not just a set of seen hashes. Then a duplicate is **a free training example**
rather than a wasted pick — the answer is already known, so it can be recorded at zero simulation cost.
A million hashes is ~8 MB, so the check covers all history.

Guard needed: after N consecutive duplicates, move to a different base machine, or a confident model can
silently stall.

**Rotations are treated as DIFFERENT machines** — no rotation canonicalisation. Not merely cautious: it
is correct. `simulator.cpp:2052`'s `beatsAnchor` orders by y, then z, then x, so rotating changes which
block is the anchor; and Minecraft's neighbour update order is fixed in world terms, so rotating changes
update order — which changes outcomes, per point 6.

## Point 24 — When modifications grow larger

**Adjusted by hand**, structured as **a weight per modification size** rather than a trigger that
switches from one size to the next.

A trigger does not fit: single-block modifications never exhaust, because every working one adds a
machine whose own single-block space then opens up. And triggering on "the model has mastered size 1"
risks a long wait for something largely trivial, since **useful modifications may require several blocks
together** — a piston is only useful with something to push.

Weights (e.g. 80/15/5 across sizes 1/2/3, shifting later) give three things a hard switch does not:
single-block practice is never lost so the model cannot quietly get worse at it; multi-block is
attempted from the start at low weight, so if two-block turns out to be where the interesting results
are you find out early; and it composes with point 5, whose note-count-versus-accuracy measurement is
exactly the evidence for how to shift the weights.

## Point 25 — Driving the simulator

Reuse the long-lived process protocol from `genetic_ml/simulator_process.py` (candidates in as compact
bytes on stdin, verdicts out as JSON lines) and `SimulatorPool`'s crash counting, hang detection and
worker restart — **copied into RL gym**, per the self-containment rule below.

That module carries a hard-won fix worth inheriting: the simulator is dynamically linked against the
msys64 MinGW runtime, and if a different MinGW runtime is earlier on PATH (Git for Windows ships one)
**every candidate dies instantly with no error output** — indistinguishable from "nothing works".

**Decided: change the C++ to write records to standard output instead of one file per candidate.** A
proper fix rather than a workaround — avoids ~1,000 file creations per second, the disk wear, and the
throughput cost.

Smaller than expected, because `sim_event_log.h:15` shows **the record is already buffered entirely in
RAM** (`push()` appends to a `std::vector<SimEvent>`) and nothing is written until `close()`. So only
*where the bytes land* changes; the offset arithmetic in `close()` works unchanged against an in-memory
stream since `tellp()` behaves the same.

Three requirements:
- **Framing** — stdout already carries JSON verdict lines, so use a strict per-candidate order: one JSON
  line, then a length-prefixed block of record bytes. No marker scanning.
- **Set stdout to binary mode on Windows.** `main.cpp` already does this for std*in*, with a comment
  explaining that text-mode translation corrupts binary streams. Without it every `0x0A` inside a record
  becomes `0x0D 0x0A` and the record is silently corrupted. The Python side already reads stdout as
  binary and decodes lines itself, so it needs no change.
- **Error paths must still emit a frame** (length zero). If simulation throws, the JSON error prints but
  `close()` may never run — a missing frame desynchronises the stream permanently.

**Keep the file mode**, because `util tools/compare_java_cpp_simlog.py` and the rest of the verification
tooling read record files — including the harness that would verify this very change.

## Point 26 — Representing a machine in Python

**Reuse the `compact_format.py` representation unchanged** (copied in). Three things already depend on
it: the simulator's stdin protocol, the visualiser's decoder, and the existing stored collections. A new
format means translation at every boundary, forever.

**Two separate indexes alongside it**, because a machine and an attempt are different things:

| | machine index | attempt log |
|---|---|---|
| one entry per | distinct structure | choice the model made |
| keyed by | hash | round |
| holds | blocks, trigger, sim-log reference | parent machine, the move, outcome, model version, note count |
| used for | the library — what can be modified next | training examples |
| rough count | ~10,000 | ~500,000 |

The distinction matters because **the training example is the attempt, not the machine**: the model's
input is *parent plus move*, so reaching the same machine from a different parent is a genuinely
different example of the model's judgement. Reaching machine B from A at round 12 and from C at round
150 is one machine and two training examples.

Metadata stays **out** of the machine format — it is about a machine's history, not the machine, and the
format is parsed by both C++ and TypeScript which want none of it.

Consequence for point 23: **the machine may be a duplicate while the attempt is not.** Skip the
simulation, reuse the stored outcome, still record the attempt.

### Self-containment rule (explicit)

**RL gym owns all its Python.** Anything needed is copied in or written fresh — no imports from
`genetic_ml`, `transformer_gym` or `util tools`. Two exceptions only:

- **the C++ simulator stays where it is** — an external tool, not copied. Forking it would put the fork
  outside the byte-for-byte Java verification that is the entire reason to trust its answers, and the
  fork would be the copy generating all training data.
- **fixtures are supplied from a separate folder**

What gets copied, trimmed on the way since each copy only has to serve one purpose:

| from | take | drop |
|---|---|---|
| `util tools/verify_simulation_data.py` | record decoder: struct layouts, `read_footer`, `read_summary`, section readers | the CLI and reporting, which is most of the file |
| `util tools/simlog_ticks.py` | `iter_ticks` | — |
| `transformer_gym/state.py` | board rebuilding (see point 27 — likely superseded by the C++ flag) | **the graph fields**, a frozen tick-0 snapshot from an abandoned design |
| `transformer_gym`'s `check_state_builder.py` | the verification | — |
| `genetic_ml/compact_format.py` | machine encode/decode | — |
| `genetic_ml/population.py` | `canonical_hash` only | the rest |
| `genetic_ml/simulator_process.py` | process handling | mostly rewritten for the new stdout protocol |
| `genetic_ml/stream_hub.py` | the visualiser feed | — |

**Must not be lost in copying:** the MinGW runtime PATH fix in `simulator_process.py`. Without it every
candidate dies instantly with no error output, indistinguishable from nothing working at all.

## Point 27 — Rebuilding per-tick state

**The C++ emits per-tick board snapshots, behind a toggleable flag.** No Python replay.

Nothing is missing from the log — the starting layout plus every change is enough to derive any tick's
state, which is what `state.py`'s `apply_tick` does. But the argument against doing it in Python is
stronger than speed: **it is a reimplementation of logic the C++ already performs**, two implementations
must be kept in step, and `check_state_builder.py` exists precisely because they can drift. The C++
already holds the exact state at every tick, so emitting it removes the second implementation, the sync
risk and the verification burden together.

At RL gym scale it is also *smaller*: for `simple_observer_engine`, events are 100 × 96 B = 9.6 KB while
a per-tick board is ~60 cells × 10 ticks × 4 B = **2.4 KB**. The tradeoff only inverts on huge machines
(`1797 onepointfive`: 54 MB of events versus ~200 MB of snapshots), which is why it is a **flag
defaulting off** — the `compare_java_cpp_simlog.py` harness runs all 55 fixtures and does not want the
extra bulk.

Both are still needed: events for the event items, their ordering, cause links, push-group membership
and failure reasons; snapshots for the cell items.

## Point 28 — Live viewer feed

**This is the only quality signal that exists.**

Because the usefulness measure is deferred, **there is no automatic way to tell whether discovered
machines are junk.** Apart from crashes, watching the visualiser is the only way to know learning is
working. That makes this a verification tool designed for a person, not a liveness indicator.

**Publish slowly.** At ~1,000 simulations/sec and ~2% success the loop finds ~20 working machines per
second — 7,200 an hour is unbrowsable. If someone is judging output, they need time per machine:
roughly **one every 10-30 seconds**, or a small batch every few minutes. Adjustable.

**Publish periodic samples, not a live trickle.** `StreamHub` keeps a backlog and resends it on connect,
so what accumulates *is* the history. Sampling every few minutes turns that backlog into a browsable
progression — what it found an hour ago against now. A live trickle only ever shows the present, which
cannot answer "is this getting better".

**Parent and child together matters.** Judging junk requires seeing *what changed*; a 12-block machine
alone says almost nothing, while the 11-block machine it came from makes the modification obvious. A
frame holds any number of machines so sending both is free — but **how the viewer presents a two-machine
frame is unverified**, and since this is the only quality check available, viewer-side work to show a
pair clearly is probably justified.

**Keep the backlog resend on connect** — the viewer resets its display on every connection, so without
it the auto-reconnect option shows an empty screen.

## Point 29 — Saving and resuming

One clear priority ordering settles almost everything:

| what | cost to lose | policy |
|---|---|---|
| **the attempt log** | **irreplaceable — every entry cost a simulation** | append-only, flushed ~1/sec |
| model weights | retrainable from the log | saved periodically; **weights only**, no training state |
| machine definitions | rebuildable from the log (parent + move) | kept for convenience |
| sim-log records | regenerable at ~1 ms each | already a cache |
| loop position, counters | trivially restartable | not saved |

Only the attempt log is genuinely irreplaceable. Losing weights costs an hour of GPU time; losing the
log costs every simulation ever run.

**The rule that matters most: never persist encoded model inputs — persist the raw facts.** An attempt
is stored as *machine + move + outcome*, never as the tensors the model consumed. Point 1 already
assumes the feature list will change; if attempts were stored encoded, **a feature change would
invalidate the entire history and every simulation would have to be re-run**. Stored as machines and
moves, a feature change costs a retrain and nothing else.

Flushing every attempt would mean ~1,000 disk syncs per second, itself a bottleneck; flushing ~1/sec
risks losing a few hundred attempts on a crash, which is obviously the better trade. Adjustable.

Resuming: load the log, load the latest weights, rebuild the machine index and duplicate-hash set from
the log, continue. Weights being minutes behind the log is harmless since training is continuous.
**Discard incomplete attempts** — a crash partway through a multi-block modification leaves an attempt
with no outcome, worth nothing.

Training state is deliberately not saved: it triples save size, needs its own version-check treatment
when the model changes shape, and the transient on resume is small — recoverable in any case because
the log is intact.

## Point 30 — Targeted mechanical tests

**Deliberately incomplete — see `DEFERRED.md`.** A much larger set is needed and how the existing
hand-made tests connect to model verification is unresolved.

## Point 31 — What is recorded

Two things with different jobs.

**The attempt log is the complete record** and point 29 already writes it. One small addition completes
it: **store the model's prediction alongside each attempt** (a few bytes). With it, every accuracy
question — overall, per mechanic, per note count, per model version, over any window chosen later — is
answerable retrospectively. Without it you are locked into whichever measurements you happened to
compute live, and a new one means waiting for new data.

**A small live log** of aggregates, one JSON object per line, for glancing at health without running an
analysis. It also carries what is *not* a property of an attempt: simulations/sec, updates/sec, cache
hit rate, queue depths, crash counts. Since simulation and training run on independent clocks (point
22), it records against both rather than assuming a shared round number.

Collected from earlier points, the required numbers: **per-answer accuracy separately** (13); **recall on
working moves, never plain accuracy** (14); **examples per model update** (22); **accuracy against note
count** (5, 24); model version (22).

**Alarms split into two kinds:**

*Mechanical, automatic — the loop has stopped working.* Success rate at zero; duplicate rate at
everything; examples-per-update in the thousands; per-answer accuracy flat early; crash counts, which
`SimulatorPool` already tracks.

*Quality, human, through the visualiser — the loop is working perfectly and producing rubbish.* Nothing
automatic catches this. It is why the periodic publishing in point 28 matters.

## Point 32 — The baseline the model must beat

Without one, a loop discovering machines because the model ranks well and a loop discovering them by
luck **look identical from the inside** — both show the library growing and healthy logs.

The comparison must isolate the model: **the identical loop with the model's scores replaced by
uninformed ones.** Same candidate set, same duplicate skipping, same machine selection — only the
ranking changes. (An untrained model is the more honest version, sharing the architecture's arbitrary
bias rather than being uniform; in practice the two should be close.)

**Run it as a permanent control, not as separate runs.** A baseline measured once goes stale: as the
library grows and machines get larger, the fraction of random moves that happen to work changes, so a
number from the original 6-block machine says nothing about a 40-block descendant. Instead **a small
adjustable share of every round (default ~5%) ignores the model entirely**, tracked separately. Always
current, costs 5% of throughput, no separate runs.

This makes the simulation budget a **three-way split**, all adjustable:

| share | move chosen by | purpose |
|---|---|---|
| top-ranked | highest model scores | find good machines |
| model-sampled | sampled from model scores | training variety (point 21) |
| **uninformed** | ignores the model entirely | the live baseline |

The third must stay genuinely uninformed — the sampled portion still uses model scores, so it cannot
serve as a control.

**Track both measures from the start:** working modifications found, and working modifications whose
added block survives trimming. The first is the honest test of "does the model rank better than
chance". The second is closer to what is wanted but is a stricter bar — early on the model may beat
random clearly on the first while both sit near zero on the second, which would look like the model
doing nothing when it is in fact learning.

**Honest limitation:** this proves the model finds *working* modifications better than random. It does
**not** prove they are worth having — the model could beat random at discovering blocks that ride along
doing nothing. That is the deferred usefulness problem arriving again, and the trimmed measure is only a
partial proxy for it.

**Trimming cost is not one simulation** except in the single-block case — see the escalation rule in
`DEFERRED.md`. At single-block modifications it genuinely is one extra simulation, which is why both
measures are affordable from day one.

## Point 33 — Whether an answer is trustworthy (calibration)

Being *calibrated* is separate from being accurate: a model can order moves perfectly while its numbers
are meaningless as probabilities — every working move above every failing one, but all scored between
0.45 and 0.55.

Ordering alone is enough for "which move next". **Two decisions read the value, not the order:**

- **the stop decision** — stop placing blocks when confidence exceeds a threshold. If the model's 0.8
  really means 30%, it submits broken machines constantly; if it means 99%, it never stops.
- **whether a candidate set is worth simulating at all** — abandoning a machine when nothing scores
  above a floor.

**Neither failure looks like a calibration problem.** The first looks like the model judging machines
badly; the second looks like the machine-selection rule being wrong.

**Measurement needs no new data** — point 31 stores every prediction beside its outcome. Group
predictions into bands and compare against observed success rates; if the model says 0.7 and reality is
20%, it is overconfident, which is the usual direction and exactly what breaks a stop threshold.

**Fix by adjusting the numbers afterwards**, not by fighting training: fit a small correction mapping
raw output onto observed rates. Seconds to fit, refittable continuously, touches neither model nor
training. Raising point 14's absolute-term weight also improves calibration but trades away ranking
quality, which matters more.

**Wrinkle: the log is not a fair sample.** The loop mostly simulates moves the model rated highly, so
predictions are dense at high confidence and sparse elsewhere. **The uninformed control group from point
32 is an unbiased sample of the model's predictions** — calibration measured on it is honest, measured
on everything it is distorted by the loop's own choices. A second, unanticipated use for the control
group.

**Stop threshold comes from the calibration table**, expressed as what is actually wanted ("stop when
the real chance of working is at least 90%") rather than a raw output value, so it stays correct as the
model changes. **Fall back to a fixed number early in a run**, while the table is based on too few
examples to trust.
