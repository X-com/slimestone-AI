# Design decisions

Each decision with the reason it was chosen. Claims that reference code cite a file and line so they
can be checked rather than trusted.

---

## A move

**`set cell C to block type T`.**

Add, remove and replace are all the same operation — removing is setting a cell to air. One list of
possible moves covers all three, and one model output covers them without special cases.

## Validity

**`validCycle` from the simulator footer. That is the only definition of failure.**

A machine is valid if its configuration after one period equals its start translated by `cycleShift`.
Blocks may be destroyed or replaced mid-cycle — only the cycle endpoints matter.

This means validity is a property of the whole cycle, so **nothing can be judged from the machine at
standstill.**

## What the model sees

A **space-time graph** of the unmodified machine.

- one node per block per tick
- plus one node per empty cell per tick, so "add a block here" has somewhere to attach

Five link types:

1. **time** — a block now, to the same block next tick
2. **touching** — two blocks adjacent at that tick (this changes as the machine moves)
3. **push group** — every block moving together in one push, linked to each other
4. **sticky** — a slime block and what it drags
5. **cause** — from the log's `actorKey` (what caused it) and `blockKey` (what it happened to),
   which also carries observer chains

Sizes are small at the scale we start with: `simple_observer_engine` is 6 blocks × 10 ticks = 60
block nodes, and its entire log is 100 events.

## Backbone: a graph network

**Not a CNN.** Two reasons, the second worse than the first:

1. A 3×3×3 convolution extends its reach by one cell per layer, so a 4-layer CNN sees 4 cells. A
   piston pushes up to 12. It cannot see the far end of the group.
2. More fundamentally, **a push group is not a spatial neighbourhood.** Slime drags blocks sideways,
   so a group is a connected set that can sprawl into any shape. A convolution's reach is always a
   box — the wrong *shape* at any size, including cells that aren't in the group while missing ones
   that are.

The push-group links make a 12-block group **one step across**, regardless of size or shape.

`transformer_gym/encode.py` already builds typed relation edges of exactly this kind (`sticks_to`,
`same_push_group`, `observes`, `would_power_direct`, `would_power_qc`) and already creates
"interface-air" nodes for empty cells beside solid blocks. That encoding work is reusable; what was
wrong in the old design was the task it pointed at, not the representation.

## What it predicts (phase one)

**The richer target**: what changes, where the first failure appears, and whether `validCycle` holds
— not just a yes/no.

Every simulation produces a full log either way, so reducing it to one bit throws away what was
already paid for.

**Predictor, not policy.** "This breaks the machine on its own" is useful and true. "Never do this"
is harmful, because good multi-block edits need first steps that fail alone. Training the model to
predict outcomes gives the knowledge without the reflex, so phase one does not poison the multi-block
phase.

## Growing from single to multi-block

The output is always **"place one block, or stop"**.

- a single-block edit = one step
- a five-block edit = five steps

Same network, same weights, no shape change. You expand by raising the step limit: 1 → 2 → 3 → 5.
Everything learned carries over; the genuinely new thing is coordination — that block A alone fails
but A+B works.

This only works if the action is sequential from the start. An output layer of "pick one edit from a
fixed list of 2,000" would be locked in and need retraining from scratch.

## Search

**Beam search** guided by the predictor: score the options, keep the best hundred, extend each by one
block, repeat.

Not AlphaZero's machinery. Branching here is comparable to Go, but depth is roughly 40× smaller
(3-5 blocks versus ~200 moves), and depth is what makes tree search expensive. MCTS, self-play and
rollouts exist to cope with depth this problem does not have. The concept worth taking is *search
guided by a learned network, trained on what the search finds* — that is the self-improvement loop,
and it costs a beam search rather than hardware you do not have.

## Pruning proposals

Supply **physics facts** the simulator already knows: push limit 12, piston reach, and that a block
must end up attached or it is left behind. These are the rules of the game, not design knowledge, and
they remove most impossible proposals before any of them cost a simulation.

Do **not** supply design patterns such as "an observer facing a piston makes a clock". That is what
the model should discover. Motifs should be harvested from what works, not authored.

The distinction matters: the first category does nearly all the pruning while teaching nothing about
design.

## Scale to start

Small machines. Measured across all 55 fixtures, 32 have ≤60 blocks and a cycle ≤32 ticks.

| fixture | blocks | cycle |
|---|---|---|
| simple_machine2 | 5 | 10 |
| simple_observer_engine | 6 | 10 |
| simple_caterpillar | 7 | 12 |
| simple_upwards_engine | 8 | 12 |
| test_flyer_31 | 17 | 10 |

For contrast, the machines we are *not* starting with: `1797 onepointfive` has 27,974 blocks and a
cycle of 1797 ticks; `complex_machine` 288; `flying_dog` 468.

**A cycle length of 2 always means no cycle was found** — every `_doesnt_loop` fixture shows it, and
so do `1 wide trencher` and `1-wide with rails`.

## Data generation

`main.cpp:59`'s `processStream` is a `while (true)` loop reading candidates from the stream until
clean EOF. For each it opens a **per-candidate** sim-log via `tracePathForCandidate` →
`<stem>-<id>.simlog`, simulates, prints the JSON verdict to stdout, and finalises the log.
`--simulation-data` is honoured in stdin-stream mode as well as file mode.

So **one process handles many candidates and returns both the verdict and the full log for each.**
Batching with sim-logs was designed in from the start (`main.cpp:32` explains the per-candidate
naming exists so batched runs don't scramble into one shared stream).

Measured throughput, from `gpu simulator/README.md`: 12,800 candidates in ~14.5s (~880/sec), 1,584 in
~1.4s (~1,130/sec).

**The GPU port is not the lever.** Its own README reports the kernel ~15% *slower* than the
12-process CPU pool on a GTX 960, memory-limited to ~50-60 concurrent workers rather than
compute-limited. Redstone is a branchy sequential state machine, so the only parallelism is across
candidates. Use the CPU pool.

`simulator_pool.py` never passes `--simulation-data`, but driving the exe directly with a batch on
stdin avoids needing to change its protocol.

Likely throttle is **file-creation churn** at ~1,000 files/sec on Windows, not compute. Fixes if it
bites: a RAM disk, or processing in chunks. Neither touches the C++.

## Verifying that it is learning

Four levels, increasing strictness.

### 1. Better than trivial

**Never report accuracy.** If 95% of edits break the machine, a model that always answers "breaks"
scores 95% and has learned nothing. Report **recall on working edits** — the rare class.

### 2. Not memorised

Train on some fixtures, then predict **every** edit on a fixture never seen and compare against
exhaustively simulated truth. At single-block scale that ground truth is affordable — about 2,000
simulations.

Anything short of an unseen machine only demonstrates recall of the training set.

### 3. Understands the mechanics

Targeted probes, not aggregate scores.

**The push-limit probe.** Build a piston with a column of 11 blocks. Adding a 12th should work; a
13th should fail. **Does the prediction flip at exactly the right count?** If it flips at 10 or 15,
or not at all, it learned a fuzzy correlation rather than the rule — and no accuracy number would
have revealed that.

Same shape for the others: obsidian in a piston's path always fails; a block against moving slime is
dragged, against moving stone it is not.

Because the target is the richer one, there is a stricter version free: **does it name the right
failure in the right place?** A model that says "this breaks" while pointing at the wrong block got
the right answer for the wrong reason — visible here, invisible with yes/no.

`transformer_gym/mechanic_fixtures.py` is the existing precedent for this pattern.

### 4. Actually helps

Working edits found per simulation, model ranking versus random ranking, same budget. If it does not
beat random ordering, the accuracy does not matter.

Supporting measures: a **learning curve** over data fraction (rising means more simulation is worth
buying; flat means something else is wrong), and **calibration** — the search *ranks* by these
scores, so miscalibrated scores rank badly even at good accuracy.

**Two headline numbers:** survival rate versus random placement (did it stop wasting simulations),
and fraction of survivors that aren't cargo (did it find anything that matters).

## Live viewing

Already built — nothing to design.

- `genetic_ml/stream_hub.py`'s `StreamHub`: background-thread WebSocket server that keeps a backlog
  and resends it on every connect (the visualizer resets its view per connection, which is what makes
  auto-reconnect work)
- `genetic_ml/compact_format.encode_candidate`: candidate → bytes, same format as the `.data` files
- the visualizer's Live Training page decodes frames with `parseCompactData`

The contract is loose: `ga_loop.py:115` documents it as *"anything with a `publish(frame: bytes)`
method"*, and `tests/test_ga_loop.py:40` already fakes it with a stub. `main_ga.py:131` is a working
reference to copy.

```python
hub = StreamHub(backlog=backlog, host=STREAM_HOST, port=STREAM_PORT)
hub.start()
hub.publish(b"".join(encode_candidate(c) for c in new_working))
```

Two notes: publish only **newly discovered** working machines, since at ~1,000 sims/sec a
modification loop would otherwise flood the socket with near-duplicates. And consider sending parent
and child in the same frame — the interesting artifact is *what changed*, not a machine alone.
