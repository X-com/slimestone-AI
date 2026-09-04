# RL gym

Home of a new model that modifies working flying machines and gets better at it over time.

This folder holds the design decisions and their reasons. The code lives in `../rlgym/` and its tests
in `../test_unit/`. See **Status** below for what is built.

## The goal

A model that proposes modifications to a working flying machine, has them verified by the C++
simulator, and improves from the results.

**The model places the blocks itself.** There is no separate random generator feeding it candidates.

## The loop

At each step the model scores **every** legal placement and picks one — about **1,250 options** with a
one-cell candidate shell, ~9,750 with two cells. Cheap enough to score in a single pass, so "the model
chooses" is literal. Multi-block is the same step repeated: place a block, re-score everything, place
the next.

Then simulate the result, learn from what happened, and rank better next round.

Since `ALPHAZERO.md`, the loop is search-guided: the model's scores are a *prior*, tree search improves
on them, and the model is trained toward what the search found. Randomness enters as **Dirichlet noise
at the search root** and **temperature on the visit counts**, so the model keeps seeing things outside
its current beliefs. Both start high and decay. Nothing switches over; it is one loop throughout.

## What to expect first

The realistic first win is a **filter, not a designer**. Learning to stop proposing obviously-doomed
placements is far easier than learning to propose useful ones — the doomed cases are the majority and
have strong local signals. That filter is what makes everything downstream affordable.

Trivial-but-working extensions will still appear. That is not a model failure: "does it still work" is
the only objective it has, and a block riding along genuinely works. See `OPEN.md`.

## Files

| file | contents |
|---|---|
| `ALPHAZERO.md` | **how the model searches and improves — architecture, MCTS, training, build order** |
| `TRAINING.md` | **how to run it — the pieces, the three stages, and which number to look at** |
| `VERIFICATION.md` | how we tell a broken implementation from a model that is not learning |
| `BENCHMARKS.md` | the label corpus, the baseline ladder, and measured timings |
| `QUESTIONS.md` | what each of the 33 design points asks, and why it matters |
| `DECISIONS.md` | what was decided for each point — **the main reference for inputs and outputs** |
| `DEFERRED.md` | problems deliberately left unsolved, with notes for whoever picks them up |
| `SLOWDOWNS.md` | every identified performance risk, since training speed matters |
| `DESIGN.md` | the earlier high-level design, written before the 33 points |
| `OPEN.md` | the earlier high-level deferred items, plus a log of corrections made while planning |

**Precedence, newest wins: `ALPHAZERO.md` > `DECISIONS.md` > `DESIGN.md` / `OPEN.md`.**

`DESIGN.md` and `OPEN.md` came first and are the shorter overview. Where they disagree with
`DECISIONS.md`, `DECISIONS.md` wins — the clearest example is the relationship kinds: `DESIGN.md` lists
five, and the settled answer is seven (power and quasi-connectivity were added in point 4).

`ALPHAZERO.md` came last. It overturns `DESIGN.md`'s rejection of AlphaZero in favour of beam search,
because beam search yields a ranked list rather than a distribution and so **cannot produce a training
target for the policy**. It also resolves four `DECISIONS.md` points: 11 (the value head conflated two
different questions), 13 (the weighting crisis largely evaporates), 14 (its machinery is unnecessary on
the policy side) and 19 ("keep everything" holds for three of four label types, not all four).

## Status

All 33 design points are settled or explicitly deferred, and `ALPHAZERO.md` settles the search and
training loop on top of them.

**No C++ change is on the critical path** — see the point 27 reversal below. The one remaining
candidate, records to stdout (point 25), is a throughput fix that matters only once the loop runs
continuously at ~1,000/sec for days.

**Phase A is built and Milestone 1 has run.** `rlgym/game.py`, `rlgym/sim.py` and `rlgym/labeller.py`,
with 98 tests in `test_unit/` (78 fast, 20 simulator-driven).

Milestone 1 on `simple_observer_engine` at R=1, all 988 legal single-block modifications:

| | |
|---|---|
| working | **314 = 31.78%** |
| of which cargo (same period and shift as the base) | **313** |
| **non-cargo working** | **1 = 0.10%** |

Both branches of the plan's decision table fired at once. 99.7% of the positive signal is a block
riding along, so **a graded reward is now the highest-value open item**, not a refinement. The single
non-cargo discovery reverses the machine's flight direction. Full result in `ALPHAZERO.md`.

**Phases B and C are built, and the training loop runs end to end** — network, supervised
training, PUCT with simulator leaves, the machine library and attempt log, and the outer loop with
its three-way budget split. 215 tests. `TRAINING.md` is the operating manual: what each file does,
how to run each stage, and which number to look at.

**MILESTONE 2 passed.** Held-out policy AUC **0.800** against baseline 1's **0.659** — the only
baseline that transfers. Train (0.825) and held-out (0.800) track within 0.025, which is the
"it generalised" row of the diagnosis table rather than the "memorising" one.

**MILESTONE 4 has a number.** Pooled over 10 Stage 1 rounds at k=2:

| | simulator calls | non-cargo found | **non-cargo / 1,000** |
|---|---|---|---|
| **model** | 350 | 10 | **28.6** |
| **uninformed control** | 1,084 | 2 | **1.8** |

**15.9x the control per simulator call.** Full numbers, and the caveat about the model
re-treading cached ground, in `BENCHMARKS.md`.

**The label corpus exists.** 33 machines, **207,935 exhaustively labelled actions** - every action of
every small fixture, labelled exactly. That replaces point 17's "train on one machine" with a real
train/test split. Corpus-wide: 43.60% working, **0.348% non-cargo**, so Milestone 1's cargo ratio holds
at scale. See `BENCHMARKS.md` for the baseline ladder the model must beat.

**Point 27 is reversed.** It called for per-tick snapshots from the C++ simulator because rebuilding
state in Python would reimplement physics. That is true of `transformer_gym/state.py`, and false of
`BlockStateChanged` — the log's replication backstop, which records every world write with the old and
new state, so replay is a dictionary assignment with no logic to keep in step. It is also self-checking:
every write states what it overwrites, so a missing write cannot hide. **Zero mismatches across all 46
fixtures.** No C++ change is on the critical path, and the byte-for-byte Java-verified simulator stays
untouched.

**Graph construction is ~90x slower than simulation** (157 ms mean per machine against 0.26 ms per
candidate) — `SLOWDOWNS.md` #3 confirmed with numbers rather than predicted. Already mitigated by a
property that falls out of the design rather than an optimisation: **a placement changes features, never
structure**, so a machine needs exactly one graph build and every candidate on it is a note patch.
Verified directly — the same machine built with and without a note produces byte-identical edge arrays.

Next: `net.py` and `train.py`, then search and the loop.

---

## Words used in these documents

Defined once so no term is used without a meaning already given.

**Model** — the thing being trained. It takes a description of a flying machine and produces numbers
as answers.

**Input** — everything the model is given to look at.

**Output** — the numbers the model produces as its answer.

**Graph** — a way of describing the machine as a set of separate items with stated connections
between them, rather than as a picture or a grid.

**Item** — one thing in that set. Here an item is either *one block at one tick* or *one event*.

**Connection** — a stated relationship between two items, for example "these two blocks are touching
at this tick".

**Feature** — one number stored on an item, describing something about it.

**Feature list** — the full ordered set of numbers stored on every item. Every item has the same
length of list, in the same order.

**Lookup table of learned numbers** — for something with a fixed set of possible values, such as
block type, the model keeps a small list of numbers per value and learns them during training. This
lets the model represent "obsidian" as learned numbers rather than as an arbitrary id.

**One run of the model** — feeding an input in once and reading the output once. This is the unit of
computation cost.

**Training** — repeatedly showing the model examples where the correct answer is known, and adjusting
its internal numbers so its answers get closer to correct.

**Learned numbers** — the internal values adjusted during training. Also called weights.

**Saved copy of the model** — a file holding all the learned numbers, so training can be stopped and
resumed.

**Answer type** — one distinct question the model answers. A model can produce several answers from
the same run.

**Wrongness score** — a single number measuring how far an answer was from correct. Training works by
making this number smaller.

**Step of information travel** — information starting at one item and moving to a directly connected
item. Moving from A to C through B takes two steps.

**Simulator** — the existing C++ program that runs a flying machine and reports exactly what happens.
It is always correct, and is the source of every correct answer used in training.

**Move** — one change to a machine: setting one cell to one block type. Adding, removing (setting a
cell to air) and replacing are all this same operation.

**Attempt** — one choice the model made and its outcome. Distinct from a machine: the same machine
reached from two different parents is one machine and two attempts.

**Episode** — one attempt at building a modification, from the first block placed to the moment the
result is submitted to the simulator.

**Note** — a marking on the input saying "assume a block of this type is here", used to describe a
machine that has not been simulated.
