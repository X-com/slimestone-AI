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
| `loop.py` | the outer loop — machine choice, budget split, cargo filter, retraining |
| `recall.py` | MILESTONE 3 — recall@B against exhaustive k=2 ground truth |
| `metrics.py` | one JSONL row per evaluation, carrying the whole config |
| `bench.py` | timing, never fails the build |

## The three stages

```
py -m rlgym.labeller --all --out data/labels        # once, ~20 min. The corpus.
py -m rlgym.baselines --budget 100                  # the ladder rungs 0-3. The bar.
py -m rlgym.train --config configs/stage0.json      # Stage 0. Supervised, no search.
py -m rlgym.loop  --config configs/stage1.json --checkpoint data/runs/stage0/best.pt
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
round   3    47.2s   non-cargo/1k: model  12.50  control   0.00   library   4   replay   1008
```

**The headline is non-cargo discoveries per 1,000 simulator calls, model versus control.**
Everything else in the row is diagnostic. At 0.348% non-cargo, a few hundred calls contain about
one by chance, so a single round is noise — it is logged per round so it can be pooled, not so it
can be read alone.

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
| binary reward is 313:1 cargo | `train.reward_cargo` = 1.0 | non-cargo/1k over rounds, with and without |
| policy-target ageing | `train.policy_age_decay` = 0.0 | held-out policy loss both ways |
| root oversampling | `train.root_weight` = 1.0 | held-out loss by state depth |
| top-M search restriction | `search.search_top_m` = 0 | recall@B at equal budget, both ways |
| library variety | `loop.share_*` | descendants per root, block histogram drift |
| graph cost | `tick_cap` = 32 | `bench.py` |
| capacity | `net.d_model` / `n_rounds` / `n_heads` | train and held-out reported separately, always |

`configs/cargo_probe.json` is the first of these already written out; run it against
`configs/stage1.json` and compare the `round` rows.

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
