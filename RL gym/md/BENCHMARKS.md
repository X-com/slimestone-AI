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

### Reading the result

| model vs 2 on training | model vs 1 on held-out | diagnosis |
|---|---|---|
| well below | — | underfitting: too small, too few rounds, or undertrained |
| matches | below | memorising: it learned the cells, not the rules |
| matches | above | **it generalised** |

## The headline number

**Non-cargo discoveries per 1,000 simulator calls on a held-out machine, versus baseline 1.**

Everything else is diagnostic. Note that at 0.348% non-cargo, a budget of 100 actions contains
about 0.35 of them by chance — so this number is noisy per machine and must be pooled.

## Timing

| operation | measured |
|---|---|
| simulation, single candidate (`elapsedNs`) | **0.26 ms** |
| simulation throughput, 12 workers | **572/sec** |
| **graph build, mean over 31 machines** | **157 ms** |
| graph build, worst (`24 onepointfive`, 15,647 items) | **412 ms** |
| graph build, smallest (`simple_machine2`, 496 items) | **11 ms** |

**Graph construction is roughly 90x slower than simulation** — `SLOWDOWNS.md` #3, confirmed with
numbers rather than predicted.

It is also already mitigated, by a property that falls out of the design rather than an
optimisation: **a placement changes features, never structure**, so a machine needs exactly one
graph build and every candidate on it is a note patch costing microseconds. Verified directly —
the same machine built with and without a note produces byte-identical edge arrays. So the cost
is 31 builds totalling 4.9 seconds, cached, across all 207,935 training examples.

The escalation path in `SLOWDOWNS.md` (numeric arrays, then C++ emitting the structure) stays
available if per-machine cost ever matters again, but it does not today.

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
py -m rlgym.labeller --all --out data/labels    # ~20 min, once
py -m rlgym.baselines --budget 100
py -m pytest test_unit
```
