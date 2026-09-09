# Running the loop

What each file does, how to run each stage, and which number to look at. `ALPHAZERO.md` is the
specification; this is the operating manual for what was built from it.

## The pieces

| file | job |
|---|---|
| `config.py` | **every knob, one dataclass.** No deferred decision is settled in code |
| `dataset.py` | graph cache, the note patch, and the Stage 0 supervised targets |
| `net.py` | the network — trunk plus `p` / `v` / `B` / `D` |
| `train.py` | losses, the training loop, evaluation against the ladder, checkpoints |
| `search.py` | PUCT with simulator leaves |
| `store.py` | machine library and attempt log — Part 6's two destinations |
| `loop.py` | the outer loop — machine choice, budget split, trimming, retraining |
| `function.py` | **redundant-block detection — what replaced the cargo metric** |
| `recall.py` | MILESTONE 3 — recall@B against exhaustive k=2 ground truth |
| `metrics.py` | one JSONL row per evaluation, carrying the whole config |
| `bench.py` | timing, never fails the build |
| `stream.py` | discoveries to the flyer-web-visualizer, over a WebSocket |
| `serve.py` | training plus the stream and the heartbeat — what the launcher runs |

## The shortest way in

**`train.bat`, here in `RL gym`. Double-click it.**

It runs four stages and **skips any whose output already exists**, so a fresh clone does the slow
work once and every run after it starts training in seconds:

    [1/4] labels    exhaustive k=1 ground truth        ~20 min, once, ever
    [2/4] stage 0   supervised on those labels         ~65 min, once, ever
    [3/4] viewer    flyer-web-visualizer, own window   skipped if npm is missing
    [4/4] stage 1   the loop, streaming to the viewer  until you stop it

It refuses to start, with the fix printed, if Python, numpy/torch/websockets, or the C++
simulator is missing — each of those failing later looks like a broken model rather than a
missing tool. A missing viewer is deliberately **not** fatal: training is the point and the run
is fully recorded to disk either way.

## Watching a run

**The viewer is `flyer-web-visualizer`, not a page this project ships.** Open the URL its window
prints, go to **Live Training**, and press Connect (`ws://localhost:8765`). Machines appear in 3D
as they are found — orbit, zoom, click one for its details, page back through history.

`rlgym/stream.py` is the producer. Two frame kinds on one socket:

| frame | carries |
|---|---|
| binary | the machines, as concatenated compact records |
| text | one JSON object: per-machine metadata, plus the round's stats |

The binary half needed **no new format**: `parseCompactData` in the viewer reads a 20-byte
`<iiiiI>` header plus 16-byte `<iiiI>` blocks, which is byte-for-byte what
`game.py:encode_candidate` already emits. The text half exists because that format carries
geometry and nothing else, so period, shift, generation, round found and the redundant-block
count have nowhere else to travel. They are joined on the `id` in each record's own header.

**One batch per round, not one per machine.** The viewer's top viewport is captioned "Latest
batch"; a batch per discovery would make it flicker on every find while saying no more than the
console already does.

**The backlog is capped at 2,000 records.** The viewer resets its history on every connect, so
the server must resend it or auto-reconnect repopulates an empty page — and an uncapped backlog
is a crash, not a slowdown. See below.

The console still prints the heartbeat, because a round takes minutes and prints only at its end:

    loaded data/runs/stage0/best.pt (step 400)

      run         data/runs/live
      stream      ws://localhost:8765
      viewer      open the visualizer's Live Training page and press Connect
      plan        10 rounds x 8 episodes, k=2, 120 simulator calls each
      stop        Ctrl-C

      ...   10s   attempts     34   library    5   replay     46   viewers 1
    round   1  118.9s   functional/1k: model 106.38  control 500.00 ...

## Why the dashboard used to lag, measured

Not the 3D. Only `PAGE = 25` machines are ever in a scene and `scene.ts` already packs one
`InstancedMesh` per block type, so ten thousand discoveries render exactly as fast as
twenty-five. Both real causes were in the data layer, and both were measured in V8 rather than
guessed at:

| | before | after |
|---|---|---|
| appending a 200,000-machine backlog | **RangeError: Maximum call stack size exceeded** | 3.3 ms |
| 2,000 scene rebuilds of 25 machines | 1,283 ms | **39 ms** |

The first is `machines.push(...batch)`, which spreads the batch into *arguments* — fine at
100,000, fatal at 200,000, and `trimHistory()` runs after the append so the client's own 2,500
cap never protected it. Fixed by capping the backlog server-side and appending with a loop.

The second is Svelte 5's `$state`, which deep-proxies everything written into it — about 40,000
block objects at the history cap, every one of them read through a proxy trap by `setMachines`'s
three `blocks.map()` passes. History needs no reactivity at all: only the length is ever
observed, so it became a plain array beside one `$state` counter. **33x** on that read pattern.

## The three stages

```
py -m rlgym.labeller --all --out data/labels        # once, ~20 min. The corpus.
py -m rlgym.baselines --budget 100                  # the ladder rungs 0-3. The bar.
py -m rlgym.train --config configs/stage0.json      # Stage 0. Supervised, no search.
py -m rlgym.loop  --config configs/stage1.json --checkpoint data/runs/stage0/best.pt
py -m rlgym.serve --config configs/stage1.json --checkpoint data/runs/stage0/best.pt   # + viewer
py -m rlgym.labeller simple_machine2 --k2           # once, ~25 min. MILESTONE 3's denominator.
py -m rlgym.recall --budget 1000                    # MILESTONE 3. Recall against that.
py bench.py
py -m pytest test_unit                              # everything
```

**Stage 0 is not reinforcement learning.** With exhaustive labels and no search it is ordinary
supervised learning: `pi` is the exact distribution over working moves and `z` the exact outcome.
No exploration, no bootstrapping, no credit assignment. RL begins at Stage 1, when `pi` starts
coming from search instead of brute force.

## What Stage 0 trains on, and what it does not

| head | target | source |
|---|---|---|
| `p` | exact distribution over working moves, **dense over every action** | brute force |
| `v` | the exact outcome of a noted terminal | brute force |
| `B`, `D` | — | **not supervised here** |

`v` is supervised **only on terminals**. Part 5 shows why labelling `v` at a non-terminal would
be labelling the wrong question: for the same state `s1`, "does this work right now" is 0 while
"what reward will this episode end with" is 0.58. No cheap label for the second exists before
search, so Stage 0 does not invent one.

`B` and `D` are wired through the loss and left unsupervised. Their labels need a record per
candidate — 207,935 sim-logs for a signal AlphaZero does not have at all. The heads exist so that
turning them on later is a config change rather than a shape change that makes every earlier
checkpoint unloadable.

## The two measurements that made the code what it is

**A placement changes features, never structure.** So a machine is built once and every candidate
on it is `apply_notes`, which is microseconds. Without this, Stage 0's 207,935 examples would be
about nine hours of graph rebuilding. `test_dataset.py` asserts the patch is byte-identical to a
rebuild, because if it ever drifts, training silently uses a different input from the one every
structural test checks.

**Memory, not time, is the wall.** Attention gathers `q`, `k` and `v` per *edge*: one round of a
430,000-edge batch holds about 660 MB, and eight rounds exhausted memory — the process segfaults
rather than raising, so it looks like a crash and not like an out-of-memory. Two defences, both
measured rather than guessed:

| defence | effect |
|---|---|
| `checkpoint_trunk` — recompute each round in the backward pass | 8x less memory, ~33% more compute |
| `max_items_per_step` — bound the batch by **items**, not by graph count | step cost stops swinging 23x with which machines were drawn |

Machines span 519 to 15,647 items, so a fixed graph count is either wasteful or fatal. Half the
budget is reserved for value states: without the reservation the largest machine eats the whole
step with its root alone and contributes no value example ever, while every counter still looks
healthy.

## The unit of budget, and why it may not survive

`ALPHAZERO.md` Part 4 counts budget in **simulator calls** rather than MCTS iterations, on the
grounds that the network runs on separate hardware at about 1 ms while the simulator is the
bottleneck. `bench.py` measures the opposite here:

| | measured |
|---|---|
| simulator call, verdict only, persistent process | **0.55 ms** |
| network forward, smallest machine in the corpus | **~80 ms** |

**torch is CPU-only on this machine, so the network is the bottleneck, not the simulator** — an
MCTS iteration costs more than the simulator call it exists to save. The argument for counting
simulator calls does not depend on which is faster (it is the only currency the uninformed control
can also spend), so the unit stands. But the *justification* Part 4 gives for it does not hold
here, and Part 4 says explicitly that this breaks "if GPU inference ever becomes the bottleneck".
It already is, in the CPU direction. `bench.py` prints this ratio every run so the moment it
changes is visible.

The practical consequence is that search budget, not simulator throughput, sets how long a round
takes — and it scales with machine size, because a forward pass scales with items and edges.

## Reading a training run

```
step    25  loss 9.098  (p 8.552 v 0.545)   train auc 0.657   held auc 0.665
```

**Both must be reported every run** or the two failure modes are indistinguishable
(`ALPHAZERO.md` Part 3, point 8):

| train | held | diagnosis |
|---|---|---|
| well below baseline 2 | — | underfitting: too small, too few rounds, undertrained |
| matches baseline 2 | below baseline 1 | memorising — it learned the cells, not the rules |
| matches baseline 2 | above baseline 1 | **it generalised** |

The value head is checked separately, by calibration rather than by AUC: a model that ranks
perfectly but predicts 0.02 everywhere passes every AUC check and fails the Stage 0 gate. The
`calibration` field of each `eval` row buckets `v` and prints predicted against actual.

## Reading a loop run

```
round 3   33.0s   functional/1k: model 272.73  control 16.67   stripped 13   library 12   replay 250
```

**The headline is functional discoveries per 1,000 simulator calls, model versus control.**

A discovery counts when, after every redundant block has been stripped, what remains is **not the
machine we started from**. Every block in it serves a function: removing any one would move a
piston to another tick, reorder two pistons within a tick, or stop the machine working.

This replaced "non-cargo per 1,000", which asked whether the flight changed — a question that
cannot separate a useless block from one placed for looks, as a floor, or for any other purpose
in the game. See `rlgym/function.py`.

`stripped` counts redundant blocks removed before admission. If it climbs while functional
discoveries do not, the loop is proposing waste, and the trimming is the only thing standing
between the library and a lineage of dead weight.

Everything else in the row is diagnostic, and a single round is noise — it is logged per round so
it can be pooled, not so it can be read alone.

**Library size is not progress.** `canonical_hash` dedupes on structure, so a machine plus one
carried block hashes differently and trivial growth registers as discovery. That is exactly how
the GA convinced itself it was working. Size rising means the loop is running, nothing more. The
`library` field also reports generation spread and descendants-per-root, which is where a
narrowing search actually shows.

## The one measurement with a denominator

Every other number here is relative — the ladder against frequency tables, the loop against a
control. **Recall@B is the only one that knows what there was to find.** Of the N working 2-block
modifications that exist on a machine, how many did B simulator calls actually find?

That denominator is `py -m rlgym.labeller <machine> --k2`: about 25 minutes of CPU, once per
machine, forever. It is a **test set and never a strategy** — the same enumeration is 3.8 days at
k=3 and roughly twelve thousand years at k=5.

Two things the measurement does deliberately:

- **Counts hashes, not routes.** A discovery is a machine, not a path to it. Different action
  pairs reach the same machine, and counting routes would let a search claim one discovery twice.
- **Gives each side its own transposition cache.** Sharing one would let whichever ran second
  inherit the other's answers for free and look dramatically better for no reason but ordering.

## The knobs that exist to be measured

Each is a deferred decision the plan says to settle by observing the net effect, not by argument.
Defaults reproduce today's behaviour, so turning one on is a config edit and the metrics row
records which setting produced which number.

| open problem | knob | what settles it |
|---|---|---|
| binary reward is dominated by no-op blocks | `train.reward_cargo` = 1.0 | functional discoveries/1k over rounds, with and without |
| how long an edit should be | `search.allow_stop` = False, `search.k` | `stopped` and `mean_depth` in the round row |
| what a working machine is worth | `loop.functional_reward` = 0.0 | functional discoveries/1k at equal budget |
| policy-target ageing | `train.policy_age_decay` = 0.0 | held-out policy loss both ways |
| root oversampling | `train.root_weight` = 1.0 | held-out loss by state depth |
| top-M search restriction | `search.search_top_m` = 0 | recall@B at equal budget, both ways |
| library variety | `loop.share_*` | descendants per root, block histogram drift |
| graph cost | `tick_cap` = 32 | `bench.py` |
| capacity | `net.d_model` / `n_rounds` / `n_heads` | train and held-out reported separately, always |

`configs/cargo_probe.json` and `configs/stop_probe.json` are the two already written out; run
either against `configs/stage1.json` and compare the `round` rows.

## The stop action, and why it is off

`k` is an exact edit length, not a ceiling: the stop slot exists in the policy head but
`legal_mask` masks it. `search.allow_stop` unmasks it, which turns `k` into a ceiling the model
may end short of - and that is the whole of what "let the model decide how many blocks to place"
needs, mechanically.

**Mechanically. The reward is the problem.** A candidate's reward is `validCycle`, and an episode
that stops at depth 0 hands back the base machine, which works. So a perfect 1.0 is available for
zero risk and *doing nothing is the optimal policy*. Turning `allow_stop` on by itself measures
that degenerate policy and nothing else.

`loop.functional_reward` is what prices it. At weight `w` a working candidate scores

    (1 - w) + w * (added blocks that survived trimming / added blocks)

so a machine whose every addition was redundant - including the one that added nothing - scores
`1 - w`. Redundancy is the one objective usefulness signal that exists here
(`rlgym/function.py`), and this is where it stops being only a filter.

Grading is not free: it needs `trim`, about 6 simulator calls, on each **working** candidate.
Those calls are reported as `grading_calls` and are deliberately **not** added to `calls`, or a
graded run would look worse at finding things when all that changed was what it paid to know.

Two numbers say what happened, both in the round row:

| | reading |
|---|---|
| `stopped` near `episodes`, `mean_depth` near 0 | the degenerate policy - it learned that doing nothing is safe |
| `stopped` low, `mean_depth` near `k` | the ceiling is binding; raise `k` or the stop action is not being learned |
| `stopped` moderate, `mean_depth` in between | it is choosing, which is the result worth having |

**The expected outcome is the first one**, and running it is how that stops being a prediction.
The deeper obstacle is `md/OPEN.md`'s "Ranking which working changes are useful": surviving
trimming says a block *does something*, not that what it does is *wanted*. Coverage and credit
assignment also degrade with depth - one binary signal for eight decisions - which is the
composition problem OPEN.md leaves open.

A config refuses an unknown key rather than ignoring it. A typo would otherwise be a silent no-op
that looks exactly like the knob having no effect — the one conclusion this whole measurement
plan must not reach falsely.

## Two things deliberately not done

**No intermediate simulation.** The only reason to simulate `s1` would be to label "does it work
right now", and that has no consumer: MCTS does not need it, the k=1 labeller already covers every
one-block modification, and detecting a broken intermediate early is explicitly banned.

**No leaf evaluated by simulating the partial machine.** It is cheap and exact and answers the
wrong question — a piston placed alone is dead weight and fails, while `piston + slime` is exactly
the extension being hunted. It would systematically prune away the modifications we want while
looking authoritative, because the number really is exact. `test_search.py` asserts every state
handed to the oracle is terminal.
