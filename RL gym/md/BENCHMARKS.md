# Benchmarks

What "is it working" means, in numbers. Every model result is reported against these.

## The corpus

Exhaustive k=1 labelling over every fixture of 200 blocks or fewer, at R=1:

| | |
|---|---|
| machines with a valid cycle | **33** |
| exhaustively labelled actions | **207,935** |
| working | **90,657 (43.60%)** |
| **non-cargo working** | **723 (0.348%)** |

Every action of every machine is labelled exactly. This replaces `DECISIONS.md` point 17's "train
on one machine, test on a handmade one" with a real train/test split by machine, and it cost about
twenty minutes of CPU once.

**Milestone 1's finding holds at corpus scale**: the raw working rate is high and the non-cargo
rate is not. 125 working modifications are cargo for every one that changes what the machine does.
But 723 non-cargo examples is a usable signal, where one machine alone gave exactly 1.

## The ladder

| # | baseline | captures | transfers to an unseen machine |
|---|---|---|---|
| 0 | uniform random over legal actions | nothing | — |
| 1 | **per-block-type frequency** | block identity | **yes** |
| 2 | per-cell frequency | position | **no** — cells are machine-specific |
| 3 | per-(cell, type) frequency | both, memorised | no |
| 4 | the model | should generalise | the whole question |

Measured, 26 training machines and 7 held out, budget = top 100 actions:

| baseline | TRAIN precision | TRAIN AUC | HELD precision | HELD AUC |
|---|---|---|---|---|
| 0 uniform | 48.9% | 0.500 | 38.0% | 0.490 |
| **1 per-entry** | 61.5% | 0.661 | **51.1%** | **0.648** |
| 2 per-cell | 92.9% | 0.923 | 90.1% | 0.925 |
| 3 per-cell-entry | 100% | 1.000 | 100% | 1.000 |

### What each rung means

**Baseline 0 sits at 0.490 AUC** — chance, as it must. Note the trap it guards: actions are
enumerated sorted by cell, and cells are strongly predictive, so a ranker that scores everything
equally would inherit that order and look far better than chance. Ties are broken randomly for
exactly this reason, and `test_baselines.py` asserts it.

**Baseline 1 is the bar.** 0.648 AUC from block type alone, and it transfers because block types
are not machine-specific. Fitted rates run from `redstone_lamp` at 57.7% down to `air` at 13.9%,
with `glazed_terracotta` at 26.2% — its `PushOnly` behaviour showing up as a measurable penalty
without anyone encoding it.

**Baseline 2 is the real target, not just a diagnostic.** 0.925 AUC says **position explains
almost everything**. It is fitted per machine, so it does not transfer — but it defines the job:
the model must *derive* from a machine's structure what baseline 2 memorises from its answers. A
model that reaches 0.9 AUC on a machine it has never seen has done something baseline 2 cannot.

**Baseline 3 is the ceiling**, and worth nothing anywhere else.

### MILESTONE 2 — the model, measured

600 steps, 26 minutes of wall time per 100 steps on CPU, 261,564 parameters, best checkpoint at
step 400:

| | model | baseline 1 | baseline 2 |
|---|---|---|---|
| TRAIN AUC | 0.825 | 0.664 | 0.923 |
| **HELD-OUT AUC** | **0.800** | **0.659** | 0.925 (does not transfer) |

**The model beats the only baseline that transfers, by a wide margin.** Read against the
diagnosis table below: train 0.825 and held 0.800 track each other within 0.025, which is the
"it generalised" row and not the "memorising" one. It is not yet at baseline 2's 0.925, so there
is capacity or training left on the table rather than an overfitting problem.

Note what the ladder could *not* see, and what the value gate caught — next section.

### The value head, and why calibration is a separate gate

At the moment the policy reached 0.773 held-out AUC, `v` on a held-out machine was:

| | |
|---|---|
| min over 60 actions | 0.4802 |
| max over 60 actions | 0.4804 |
| mean on working / failing | 0.4803 / 0.4803 |

**A constant.** It had learned the machine's base rate and nothing else, while every number in
the ladder looked healthy. Cause, measured rather than guessed: a note moves the noted cell's own
vector by **~6.0** and the summary item's vector by **~0.004**, and `v` read only the summary.
Part 3 predicted the mechanism — *"the summary item is a bottleneck, so fine detail cannot route
that way"* — without connecting it to this head.

Two fixes, and the first one was not enough, which is the useful part:

| | `v` spread over 80 actions | value AUC |
|---|---|---|
| summary only | 0.0002 | 0.647 |
| + learned attention pool over all items | 0.0006 | — |
| **+ max pool** | **0.62** | **0.835** |

A softmax over ~1,500 items with O(1) scores is near-uniform: to concentrate, the query has to
grow until scores span log(n) ~ 7, and it does not get there in a few hundred steps. A max pool
concentrates by construction and the noted cell *is* the outlier. Point 12 rejected max for
pooling a cell's **ticks**, because a cell can need two ticks together; that argument does not
apply to *"is there something unusual here"*, which is what `v` needs.

After the full run, `v` has real spread (0.32-0.38 per machine) but **uneven transfer** — value
AUC 0.816, 0.744 and 0.465 on three held-out machines. It ranks on some and not others. Stage 1
gives it no authority by design, so this is the number to watch rather than a blocker; Stage 2 is
where it becomes load-bearing.

### Reading the result

| model vs 2 on training | model vs 1 on held-out | diagnosis |
|---|---|---|
| well below | — | underfitting: too small, too few rounds, or undertrained |
| matches | below | memorising: it learned the cells, not the rules |
| matches | above | **it generalised** |

## MILESTONE 3 — recall against exhaustive k=2 ground truth

The only measurement in the project with a **denominator**. `simple_machine2`, R=1, k=2:

| | |
|---|---|
| unordered action pairs enumerated | **359,712** |
| distinct machines among them | **359,712** — every pair gives a different machine |
| working | **5,457 = 1.52%** |
| non-cargo | **17 = 0.005%** |
| cost | 484 seconds, once, forever |

At a budget of 1,000 simulator calls:

| policy | calls | working found | **recall** | working/call |
|---|---|---|---|---|
| uninformed control | 1,000 | 6 | 0.11% | 0.006 |
| **model** | 1,000 | **47** | **0.86%** | **0.047** |

**7.83x the control at equal budget.**

### Checking the control is not simply handicapped

A ratio is only as good as its denominator, so the control was checked against a *true* uniform
draw — 1,000 pairs sampled directly from the enumeration and simulated:

| | working in 1,000 |
|---|---|
| true uniform draw | **8** |
| search-based control | **6** |
| expected from the 1.52% base rate | 15.2 |

The two agree within noise, so the control behaves like uniform sampling and the 7.83x is real
rather than an artefact of a crippled baseline. Both sit below the base-rate expectation by about
2 standard deviations, which is unremarkable at n=6.

That same check independently confirmed the ground truth: of the 1,000 uniformly drawn pairs,
**8 were working by simulation and 8 by ground-truth hash lookup** — the two agree exactly, on a
sample the enumeration had no way to anticipate.

Raising the budget widens the gap rather than closing it, which is the shape a real prior should
produce — the model keeps finding new working machines where uniform sampling saturates:

| budget | control recall | model recall | ratio |
|---|---|---|---|
| 1,000 | 0.11% (6) | 0.86% (47) | **7.83x** |
| 6,000 | 0.38% (21) | **3.88% (212)** | **10.10x** |

### What this measurement cannot yet answer

**Non-cargo recall was 0.00% for both sides at 1,000 calls and still 0.00% at 6,000.** There are
17 non-cargo machines in 359,712 — even at the model's 10x enrichment, 6,000 calls expect 0.28 of
them. Roughly 21,000 calls would be needed to expect a single one.

**The cause is the machine, not the budget.** `simple_machine2` was chosen for Milestone 3 because
it is the smallest — and it turns out to be the worst possible choice for this particular column:

| machine | k=1 legal actions | non-cargo | rate |
|---|---|---|---|
| simple_machine3 | 2,660 | 34 | **1.278%** |
| simple_no_sticky_loop | 6,232 | 47 | 0.754% |
| simple_machine1 | 3,648 | 20 | 0.548% |
| simple_caterpillar | 1,444 | 3 | 0.208% |
| **simple_machine2** | 874 | **0** | **0.000%** |

**Non-cargo density spans at least 250x across the corpus, and the test machine has none at all
at k=1.** That is also why the Stage 1 loop reported 28.6 non-cargo per 1,000 calls while this
machine's entire k=2 space allows at most ~0.5 per 1,000: they are the same metric measured on
different populations.

**Consequence for the headline number: it is not comparable across machines and must never be
reported as a single figure from a varying population.** Either fix the machine set or pool over
a fixed one. Measuring non-cargo recall properly means running the k=2 ground truth on
`simple_machine3` — about ten times the enumeration, so roughly 80 minutes once.

## The headline number

**Non-cargo discoveries per 1,000 simulator calls, model versus the uninformed control.**

Everything else is diagnostic. Note that at 0.348% non-cargo, a budget of 100 actions contains
about 0.35 of them by chance — so this number is noisy per round and must be pooled.

### MILESTONE 4 — measured

Stage 1: 10 rounds, 8 episodes each, k=2, PUCT with ground-truth leaves, starting from the
Stage 0 checkpoint. Pooled across all rounds:

| source | simulator calls | working | non-cargo | working/1k | **non-cargo/1k** |
|---|---|---|---|---|---|
| top | 262 | 52 | 6 | 198.5 | 22.9 |
| sampled | 88 | 27 | 4 | 306.8 | 45.5 |
| **model (top + sampled)** | **350** | **79** | **10** | **225.7** | **28.6** |
| **uninformed control** | **1,084** | **16** | **2** | **14.8** | **1.8** |

**The model finds 15.9x more non-cargo modifications per simulator call than the control**, and
15.3x more working ones. The control's 1.8/1,000 is the right order for k=2 given the k=1 corpus
rate of 3.48/1,000, so the baseline is behaving as the corpus predicts rather than being broken.

### The number under the number

The control spends its full 120-call budget every round. The model spends 5 to 105:

```
new simulator calls per round      round  1: uninformed 120   top  25   sampled 15
                                   round  6: uninformed 120   top   5   sampled  3
                                   round 10: uninformed 120   top  30   sampled 21
```

It is still running its full ~120 MCTS iterations. The difference is that most of the leaves it
reaches are **already in the transposition cache**, so they cost a dictionary lookup instead of a
simulator call — which is exactly the benefit point 23 predicted, and the reason the budget is
counted in calls rather than iterations. The 10 discoveries are all from genuinely new candidates;
cached repeats never enter `attempts`.

But the same number is the early signature of the self-selection problem `ALPHAZERO.md` Part 6
warns about: **the model keeps re-treading ground it has already covered.** The library shows it
too — 17 machines from 5 roots, with 8 descendants concentrated on one. That is `DEFERRED.md`'s
binning problem arriving on schedule, not a surprise, and the uninformed 5% is the only thing
sampling outside it.

## Timing

| operation | measured |
|---|---|
| simulation, single candidate (`elapsedNs`) | **0.26 ms** |
| simulation, verdict via a persistent process | **0.55 ms** |
| simulation throughput, 12 workers | **572/sec** |
| graph build, smallest (`simple_machine2`, 793 items) | **19 ms** |
| graph build, worst (`24 onepointfive`, 22,797 items) | **~500 ms** |
| **note patch, per candidate** | **0.03 ms** |
| **network forward, smallest machine** | **~80 ms** |
| network forward + backward, smallest machine | **~275 ms** |

**Graph construction is roughly 90x slower than simulation** — `SLOWDOWNS.md` #3, confirmed with
numbers rather than predicted.

**And the network is 147x slower than a simulator call**, which contradicts the premise
`ALPHAZERO.md` Part 4 gives for counting budget in simulator calls: that the network runs on
separate hardware at ~1 ms. torch is CPU-only here, so **an MCTS iteration costs more than the
simulator call it exists to save**. The unit still stands — it is the only currency the
uninformed control can also spend — but Part 4 names this exact condition as what breaks its
justification, and it is now the case. `bench.py` prints the ratio every run.

Item counts rose after the graph was corrected to cover every cell a legal placement can write:
23% of extended-piston actions previously put their head outside the encoded region, so those
placements were noted as a body with no head. `simple_observer_engine` went 519 -> 849 items.

It is also already mitigated, by a property that falls out of the design rather than an
optimisation: **a placement changes features, never structure**, so a machine needs exactly one
graph build and every candidate on it is a note patch costing microseconds. Verified directly —
the same machine built with and without a note produces byte-identical edge arrays. So the cost
is 31 builds totalling 4.9 seconds, cached, across all 207,935 training examples.

The escalation path in `SLOWDOWNS.md` (numeric arrays, then C++ emitting the structure) stays
available if per-machine cost ever matters again, but it does not today.

## The corpus is 31 machines, and the held-out set is 6

Declared and usable are not the same number, and until this was reported the difference was
silent:

| | declared | usable |
|---|---|---|
| corpus | 33 | **31** |
| held out (`HELD_OUT`) | 7 | **6** |

`test_flyer_7` and `test_flyer_9` have 44-tick cycles and are refused by the 32-tick cap;
`test_flyer_9` is one of the held-out machines. So every held-out number reported here is an
average over 6 machines. `train.py` now prints the skips in its header and records them in the
metrics row, because a refusal that is loud in a log line and invisible in a result is the wrong
way round.

**The cap is a cost boundary, not an arbitrary limit.** Raising it to 48 does recover both
machines — at 35,691 and 41,935 items, roughly **twice the largest machine currently trainable**
(22,797). They would then dominate the item budget of every step they appeared in.

`HELD_OUT` is deliberately *not* being edited to match. Changing the split now would silently
invalidate every number already measured against it, which is a worse failure than a six-machine
test set that says it is six.

## Data scarcity worth knowing before training

**Quasi-connectivity has almost no training data.** Across all 33 machines the record contains
**60 would-power edges, of which 22 are quasi-connectivity**. `DEFERRED.md` calls QC "probably the
strongest single test of real understanding" — and there are 22 examples of it in the entire
corpus. A model failing the QC probe may be short of data rather than short of capacity, and the
probe cannot distinguish those.

Thirteen of 33 machines have any power edge at all; `simple_observer_engine` has none, being
purely observer-driven.

## Reproducing

```
py -m rlgym.labeller --all --out data/labels                      # ~20 min, once
py -m rlgym.baselines --budget 100                                # the ladder
py -m rlgym.train --config configs/stage0.json                    # MILESTONE 2, ~64 min
py -m rlgym.loop  --config configs/stage1.json \
                  --checkpoint data/runs/stage0/best.pt           # MILESTONE 4, ~20 min
py bench.py
py -m pytest test_unit                                            # 215 tests (130 without the simulator)
```

`md/TRAINING.md` is the operating manual for these — what each knob does and which number to
look at.
