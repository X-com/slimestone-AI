# AlphaZero design

How the model searches and improves. `DECISIONS.md` settled *what the model looks at* and *what it
answers* across 33 points; it did not settle the search or the self-improvement loop. This document
does, and it is a vehicle for learning how AlphaZero works as much as a specification.

**Where this supersedes earlier documents:**

| document | what changes |
|---|---|
| `DESIGN.md` | its beam-search section is overturned — see below |
| `DECISIONS.md` point 11 | the value head conflated two different questions; resolved in Part 5 |
| `DECISIONS.md` point 13 | the weighting crisis largely evaporates under this framing |
| `DECISIONS.md` point 14 | its machinery is unnecessary on the policy side, still required on the value side |
| `DECISIONS.md` point 19 | "keep everything" is right for three of four label types, not all four |
| `DEFERRED.md` | the trimming-cadence sub-question is closed; the variety problem is restated more sharply |

**Why `DESIGN.md`'s rejection of AlphaZero was wrong.** It argued *"depth is 40x smaller than Go, and
MCTS exists to cope with depth"*. True about depth, wrong about the conclusion, for a reason it missed:
**beam search cannot produce a training target for the policy.** It yields a ranked list, not a
distribution over actions. AlphaZero's self-improvement engine is *train the network to imitate the
search's visit counts*. Without visit counts there is no loop, only a ranker that never improves.

**Goal:** a model that, given a working flying machine, proposes modifications that keep it working and
make it more capable, improving on its own from a single base machine.

**Working assumption:** this is experimental and nothing will work first try. **Retraining is expected
and cheap** (~252k parameters, minutes per run). Future-proofing therefore only earns its place when it
is free — a masked slot or an always-true flag, never added complexity.

---

# Part 1 — What AlphaZero is

Five parts, nothing else.

| # | part | what it does |
|---|---|---|
| 1 | **the game** | states, legal actions, transitions, terminal test, terminal reward |
| 2 | **one network, two heads** | `f(state) -> (p, v)`. `p` = prior over actions, `v` = predicted outcome |
| 3 | **MCTS** | `p` decides where to look, `v` scores leaves, PUCT balances them |
| 4 | **policy improvement** | MCTS visit counts `pi` are a **better** policy than `p`. Train `p` toward `pi` |
| 5 | **the data loop** | play with MCTS, store `(state, pi, z)`, train, repeat |

Part 4 is the engine, and it is the part people miss. AlphaZero does not learn from winning — it learns
from **search being better than the network**, closes that gap, then searches again from a stronger base.

**This is the single-player deterministic setting** (AlphaTensor / AlphaDev), not two-player. So: no
opponent; no chance nodes; no sign flipping during backup; and the terminal reward is *exact* rather
than a noisy sample of one game, which makes value learning far easier here than in Go.

---

# Part 2 — The game definition

| | |
|---|---|
| **state** | `(base machine's simulated record, set of placements)`. Base fixed for the episode |
| **actions** | `(cell, palette entry)` over candidate cells, plus a reserved masked `stop` |
| **transition** | write the placement into the set. Deterministic, free, **no simulator call** |
| **terminal** | `len(placements) == k`, a fixed constant |
| **reward** | build -> `canonical_hash` -> simulate if new -> `validCycle` as a **float** |

## One episode, traced

k=2, base = `simple_observer_engine` (6 blocks, 10-tick cycle):

```
s0 = (engine, {})
     ~1,250 legal actions
     a0 = sticky piston at cell (1,1,1) facing west
s1 = (engine, {(1,1,1): sticky_piston/west})
     ~1,211 legal actions
     a1 = slime at cell (0,1,2)
s2 = terminal — k reached
     -> build, hash, simulate -> validCycle = true -> reward 1.0
```

MCTS calls nothing but those five things. Swapping the reward from binary to graded later touches the
reward step and nothing else, which is why keeping it a float matters.

## Candidate cells

A cell is a candidate if its **face-step distance to the nearest machine block is <= R**, measured on the
tick-0 layout. Placements edit the starting configuration, so the swept volume does not affect legality
— it is information for the model, not a rule.

**R is a hyperparameter. Start at R=1; move to R=2 once the pipeline is verified.**

**Measured** on `simple_observer_engine`, not estimated — the earlier estimate of ~250 cells at R=2 was
wrong by nearly 4x, because face-step distance grows far more slowly than the bounding-box shell it
assumed:

| R | cells (6-block machine) | action slots | legal actions | exhaustive k=1 | exhaustive k=2 |
|---|---|---|---|---|---|
| 1 | **26** | 1,249 | **988** | 1.7 s | **~8 min** |
| 2 | **68** | 3,265 | 2,584 | 4.5 s | ~1 h |

Legal actions are `cells x 39 live entries - cells`, because **every cell has exactly one no-op**:
placing what is already there on an occupied cell, or placing air on an empty one. 26 x 39 - 26 = 988.

R=2 is the eventual target — a block two cells in front of a piston is sometimes exactly what is needed,
and so is deliberately keeping things from touching. R=1 first only because it keeps the exhaustive k=2
ground-truth set — the only unambiguous measure of whether search works — regenerable in minutes while
code is still changing.

## The palette — 39 live entries per cell, 48 slots

Counted against all 55 fixtures, not assumed:

| # | entry | facings | cells written |
|---|---|---|---|
| 1-2 | piston / sticky piston, **retracted** | 6 each | 1 |
| 3-4 | piston / sticky piston, **extended** | 6 each | **2** (base + head) |
| 5 | observer | 6 | 1 |
| 6-13 | slime, stone, glass, redstone block, glazed terracotta, redstone lamp, fence gate, trapdoor | — | 1 |
| 14 | air (removal) | — | 1 |

Block usage measured across the corpus, which is what the palette is derived from:

| id | block | uses |
|---|---|---|
| 20 | glass | 12,570 |
| 29 | sticky piston | 6,867 (3,621 extended) |
| 165 | slime | 6,422 |
| 33 | piston | 4,799 (732 extended) |
| 152 | redstone block | 4,563 |
| 34 | piston head | 4,353 (never placed alone) |
| 1 | stone | 1,405 |
| 218 | observer | 1,314 |
| 24, 41, 57, 155, 159, 168, 169, 173, 95, 113, 85, 46 | decorative solid cubes | ~1,100 |
| 235/238 | glazed terracotta | 47 |
| 107/96/167 | fence gate / trapdoors | 13 |
| 123 | redstone lamp | 4 |

- **Stone stands in for every plain solid cube** — a dozen ids, all functionally identical.
- **Fence gate and trapdoor are fixed at meta 0.** One entry each; facings not modelled.
- **Extended pistons are compound actions** writing two cells atomically: base with meta bit 3 set, head
  at `base + facing` with meta `facing | (sticky ? 8 : 0)`. Verified in the corpus — every extended
  piston has exactly one matching head (`(sticky, meta 11)` appears 2,597 times, `(head, meta 11)` 2,597
  times). A lone head is destroyed as an orphan (`simulator.cpp:718`); replacing an extended piston's
  head kills the base (`:2197`). The head is never independently placeable.
- **Immovable blocks are excluded from the palette** — never placed during training. Obsidian is
  *correctly* immovable (`piston.cpp:18` mirrors `BlockPistonBase.canPush:384`, which hardcodes it), so
  `DEFERRED.md`'s immovable-block probe is valid as written. Note: **no fixture contains obsidian**, so
  the Java/C++ comparison harness has zero coverage of that path.
- Also sharper than point 2's note: stone is fine (id 1 *is* in `normalCubes`); **glass (20) is the id
  actually missing from `block_registry.cpp`**, so it gets `isFullBlock = false` and nothing can be
  supported on top of it. Affects only rails, which are excluded from training — but glass is the most
  common block in the corpus, so it is worth knowing.

## Legality

Placements are legal on **any** candidate cell, occupied or empty — air is removal, a different block is
replacement (`DESIGN.md`: "add, remove and replace are all the same operation"). Masked each turn:
no-ops; any cell already written this episode (**both** cells for extended pistons); reserved slots;
`stop`.

Forbidding a second write to the same cell makes the placements a genuine **set**, so order never
matters and point 23's `k!` duplicate explosion is prevented structurally rather than caught afterwards.

## Terminology — two things previously conflated

| term | meaning |
|---|---|
| **placement** (the action) | "cell C now holds block B". Any candidate cell |
| **note** (the encoding) | how that placement is *shown to the model* |

Point 5's "notes never overwrite" is a rule about the **feature vector only** — a cell node's push-group
and movement facts came from a simulation where the original block was there, so they cannot be
rewritten. **It is not a restriction on which cells are legal targets.**

## Consequence: class imbalance is worse than assumed

Roughly 20 of ~1,250 actions are ever correct at R=1, ~20 of 9,750 at R=2 — **0.2 to 1.6%**. Point 14's
mechanisms were designed against an assumed 2.4%. Its note that *"the multiplier must be computed from
the observed success rate, not fixed"* is therefore essential, not prudent.

## Still open

**Reward shape.** Binary today; will become graded 0-1 by means not yet decided. Locked in now so the
change is free: reward is a **float everywhere**, and the value head uses **BCE against a scalar
target**, not a 2-class softmax, since BCE accepts soft targets unchanged. Until then a block that rides
along scores identically to a real extension — the trivial-extension problem arriving through the reward.

**`stop`.** Not in play; `k` is fixed. The slot is **reserved and masked** from day one. Its degenerate
optimum must be closed before it is ever unmasked: stopping at zero placements leaves the machine
unmodified and therefore certainly valid, so it needs either a floor on placements or zero reward for an
unmodified machine.

---

# Part 3 — The network

One function. State in, four answers out, **all from a single pass**:

| output | meaning | AlphaZero name |
|---|---|---|
| `p` | how promising is each placement (~1,250 at R=1) | **policy** |
| `v` | expected final reward | **value** |
| `B` | which blocks start behaving differently | auxiliary |
| `D` | which of the 7 `failureReason` values | auxiliary |

`B` and `D` exist because every simulation produces a full record anyway (point 11) — reducing it to one
bit wastes what was already paid for. AlphaZero has no analogue; they are a standard trick to densify a
sparse signal.

## The invariant

> **No model dimension may depend on machine size, cell count, or tick count.**

The policy head is `Linear(128 -> 48)` applied **per cell** — the same parameters whether there are 32
cells or 3,000. Cell count is a data dimension, not a weight dimension. Changing R, the machine, or the
cycle length changes **no weights**. Anything violating this is rejected on those grounds.

## Why a graph

| | why not |
|---|---|
| 3D grid -> CNN | a 3x3x3 convolution reaches **one cell per layer**; a piston pushes 12. And a push group is a *connected set* — slime drags sideways — so it is not box-shaped at any size |
| flat list -> plain transformer | no structure, and cost grows as items squared. 421 items = 177k pairs; 30,000 items = 900M |
| **graph -> relational attention** | structure is given rather than rediscovered, and cost grows with *edges* |

## Items — `simple_observer_engine`, R=1, 10-tick cycle

| kind | count | what it is |
|---|---|---|
| **cell items** | 32 cells x 10 ticks = **320** | what is at this cell at this tick |
| **event items** | **100** | one per record event, carrying its order number |
| **summary item** | **1** | one per machine, connected to everything |
| | **421** | |

Event items exist because point 6 measured 3-4 events per cell-tick, up to **13**. Collapsing them
destroys ordering, and ordering decides outcomes when several blocks move at once.

Three item types, each with **its own input projection** plus a type embedding — event items carry no
block type, so one shared input layer with zero-filled columns would be wrong:

| type | features |
|---|---|
| cell | block type, facing, is-trigger, is-air, moving, direction, push group size, powered, observed, in-flight, `BlockData` lookups, note fields |
| event | event kind, order within tick (`globalSeq`), success flag, `failureReason`, tick |
| summary | period, net shift, block count, `view_is_complete` |

## From features to 128 numbers

**Categorical -> learned lookup table.** `block_type_emb` has one row of 128 numbers per block id. They
start random; training discovers that piston and sticky piston belong near each other. This is why point
2 stored type and facing separately — one table of 256 rows and one of 7, rather than 60+ combined rows
each learned from scratch.

**Numeric -> one small linear layer** over the ~15 scalars.

**All terms are added, not concatenated** — which is why point 1's "the feature list will be expanded
later" stays cheap: a new group is one more term in the sum, and `d_model` never changes.

## Relations

Seven physics relations (point 4) plus time forward/backward, bucketed time-skip (1, 2, 4, 8, 16, 64,
256), event ordering, event cause, and summary. **64 slots reserved**, ~35 used.

| relation | example |
|---|---|
| touching | slime (0,0,1) to sticky piston (1,0,1) |
| same push group | every block moving together, **all linked to each other** |
| slime dragging | slime (1,0,2) to what it carries |
| observes | observer (1,1,2) to the single cell it faces |
| power reaches | redstone source to piston |
| quasi-connectivity | power to the cell **above** a piston |
| caused-by | from the record's `actorKey` |

**"Same push group" is the load-bearing one**: every member links to every other, so a 12-block group is
**one step across** regardless of size or shape.

**But one step is not counting.** Attention computes a weighted average, and the average of 12 things
looks like the average of 11 — which is exactly the push limit. That is why point 1 puts **push group
size on the item as a number**. And it exposes the real task: the recorded size describes the machine
*without* the placement, so the model must reason *"recorded 11 -> my note is adjacent -> really 12 ->
still within the limit -> it works"*. **That inference is the whole problem**, and it is why `v` cannot
be a lookup.

## The trunk

```
x = x + attention( norm(x) )      <- mix information BETWEEN items
x = x + ffn( norm(x) )            <- process information WITHIN an item
```
repeated T times with the **same weights** (point 10).

**Attention** — each item emits a query ("what am I looking for"), key ("what am I") and value ("what
I would tell you"). Per edge:

```
score = query(destination) . key(source) + relation_bias[edge type]
```

softmaxed across each item's incoming edges. **This is exactly point 6's Choice C**: `query.key` makes
the weight depend on what the neighbour currently *is* — a moving slime and a stationary stone produce
different keys — while `relation_bias` covers the *kind* of connection.

**Why the residual add** — keeps item identity across 8 rounds, and prevents oversmoothing (repeated
averaging makes a slime and a piston converge to the same vector).

**Why the FFN** — attention can only blend; it cannot compute. The FFN is `Linear(128 -> 512) -> GELU ->
Linear(512 -> 128)` per item, and it is where *"11 + 1 = 12, which is at the limit"* happens.
Alternating the two is the design: attention moves facts, the FFN combines them.

**Why pre-norm** — normalising going in and adding the raw result is what makes iterated stacks
trainable without warmup tricks.

**4 heads x 32 dims** — one head can follow push-group links while another follows time and a third
follows power, simultaneously. With one head the item must pick a single blend for everything.

**T = 8** by the longest specific chain:

| chain | steps |
|---|---|
| observer -> watched cell -> piston | 2 |
| redstone -> quasi-connectivity cell -> piston -> head -> pushed block | 4 |
| the above, then across the push group | 5 |
| tick-3 cause -> tick-10 endpoint, **with** skip links | ~2 |
| the same with only gap-1 links | 7 |

The summary item makes everything 2 steps apart, but it is a bottleneck — 421 items squeezed through one
128-number vector — so fine detail cannot route that way.

## Weight tying — revisited

Point 10 tied the trunk partly so T could rise without retraining. **That argument is void now that
retraining is expected.** The decision survives for a different and stronger reason:

| | trunk parameters |
|---|---|
| **tied** (1 block, applied 8x) | **197,000** |
| untied (8 blocks) | 1,580,000 |

Stage 0's entire training set is ~1,250 examples — one machine's complete single-block space. 252k
parameters against that is already heavily over-parameterised; untying makes it 1.6M. **Tying acts as
regularisation, and at this data size that dominates.** Revisit if the library reaches tens of thousands
of machines. A middle option exists (untie round 1, tie the rest) and is deliberately not built.

## How a placement appears in the input

A placement edits the **initial configuration**. What the cell holds at tick 5 afterwards is unknown —
that is the thing being predicted. So:

| feature | where | why |
|---|---|---|
| `note_block_type`, `note_facing` | **tick-0 item only** | the only tick where the placement is a fact |
| `cell_is_noted` | **every tick item of that cell** | "everything here describes a machine that no longer exists" |

Without the second flag, a cell that held slime still reads *"slime, moving east, push group of 5"* at
tick 5 — a description of a machine that will not exist. Both cells are noted for extended pistons.

## Masking

Illegal actions get their logit set to **minus infinity before the softmax**, never zeroed after.
Zeroing afterwards leaves the remaining probabilities mis-normalised.

## Batching without padding (point 16)

Machine A has 421 items, machine B has 1,200. Padding A wastes 65%. **Concatenate into one flat list** —
which works because attention is edge-restricted and no edge crosses between machines, so they genuinely
cannot see each other and no masking is needed to keep them apart.

Two requirements: **every item carries a `machine_id`** so losses average within a machine before
averaging across machines (otherwise a machine's weight depends on its batchmates); and **one summary
item per machine, never shared** (a shared one leaks information between machines in two steps, making
training look better than it is and then failing silently on a machine evaluated alone).

## The complete forward pass

```
1.  build graph            421 items, ~3,000 edges
2.  encode                 per-type projection + embeddings, summed  ->  [421, 128]
3.  trunk x 8              x = x + attn(norm(x));  x = x + ffn(norm(x))
4.  final norm
5.  v   <- sigmoid(Linear(128->2)  on summary), output 0            1
6.  D   <-         Linear(128->7)  on summary                       7
7.  pool  each cell's tick-items -> 1 vector, by learned attention -> [32, 128]
8.  B   <-         Linear(128->1)  per cell                         32
9.  p   <-         Linear(128->48) per cell + stop logit from summary
                   -> mask illegals to -inf -> softmax
```

**Why learned pooling** (point 12): a cell empty at ticks 0-3 then swept by a piston at tick 4 has its
one informative tick divided by 10 under averaging; taking the maximum cannot represent a cell needing
*two* ticks together — pushed at tick 3, hit by a second piston at tick 8. A single learned query vector
scores each tick item and softmaxes, so it can put all weight on tick 4 or split between 3 and 8.

## Parameters

| part | count |
|---|---|
| block type table (256 x 128) | 32,768 |
| facing, note, `BlockData`, event-kind, type tables | ~10,000 |
| three per-type input projections | ~5,000 |
| **trunk — one block, reused 8x** | **197,000** |
| relation biases (64 x 4) | 256 |
| pooling query | 128 |
| heads (`p` 48, `v` 2, `B` 1, `D` 7) | ~7,200 |
| **total** | **~252,000** |

AlphaZero's chess network is ~45M — this is **180x smaller**, and a full retrain is minutes.

**Least confident numbers: T=8, d_model=128, 4 heads.** Conventional rather than derived. Point 8's
diagnostic is the only settlement — poor on the training machine means too small, good on training but
poor on held-out means too large. **Both must be reported every run** or the two failures are
indistinguishable.

`transformer_gym/model.py` is **not** reused. It was built for a different task.

---

# Part 4 — MCTS

## Terminology, fixed

| term | meaning | cost |
|---|---|---|
| **MCTS iteration** | one pass down the tree and back up. AlphaZero calls this a "simulation" | one network call, sometimes |
| **simulator call** | one invocation of the C++ engine on a finished candidate | ~1 ms |

1:1 in Stage 1; they diverge in Stage 2.

## Why search when `p` already scores every move

`p` is a *guess* about how episodes starting with a move tend to end. Search *tries* it.

> The network rates piston-at-(1,1,1)-west 0.30 and slime-at-(0,1,2) 0.25 — nearly tied. Search finds
> working follow-ups under the piston and none under the slime. The piston's estimate rises to 0.58,
> slime's falls to 0.00. Final visits: 140 vs 20.

**The gap between the network's guess and what search found is the entire learning signal.**

## One iteration

```
1. SELECT    from the root, take the highest-PUCT child until reaching an unexpanded node
2. EXPAND    run the network -> p (children's priors) and v
3. EVALUATE  produce a number for this position
4. BACKUP    walk back up; for every edge taken: N += 1, W += z
```

Step 3 is where Stage 1 differs from AlphaZero: **descend to terminal and call the simulator** rather
than reading `v`. Exact rather than estimated. This is AlphaZero's tree with AlphaGo's rollouts — except
the rollout is a perfect simulator, affordable only because depth is 2 and the oracle costs 1 ms.

**Banned: evaluating a leaf by simulating the partial machine.** Cheap and exact, but it answers the
wrong question. A piston placed alone is dead weight and fails; `piston + slime` is the extension being
hunted. Mid-edit validity **systematically prunes away exactly the modifications we want**, while
looking authoritative because the number is exact.

## PUCT

Each edge stores `P` (the network's prior, fixed at expansion), `N` (visits), `W` (total reward),
`Q = W/N`. Selection maximises:

```
PUCT(a)  =  Q(a)  +  c_puct . P(a) . sqrt(sum N) / (1 + N(a))
            -----     ---------------------------------------
            what I     how much I still want to look here
            found
```

At the start every `N` and `Q` is 0, so ranking is purely `P` — **the first iterations follow the
network's opinion exactly.** As visits accumulate `Q` dominates and evidence displaces opinion.
`c_puct` ~ 1.25 sets how long opinion holds out.

**Root after 20 iterations** (`sqrt(20) = 4.472`, `c_puct = 1.25`):

| action | `P` | `N` | `W` | `Q` | bonus | **PUCT** |
|---|---|---|---|---|---|---|
| piston (1,1,1) west | 0.30 | 12 | 7.0 | **0.583** | 0.129 | **0.712** |
| slime (0,1,2) | 0.25 | 5 | 0.0 | 0.000 | 0.233 | 0.233 |
| stone (1,1,1) | 0.10 | 2 | 0.0 | 0.000 | 0.186 | 0.186 |
| observer (0,1,2) north | 0.05 | 1 | 0.0 | 0.000 | 0.140 | 0.140 |
| ...1,244 others | ~0.001 | 0 | 0.0 | 0.000 | 0.006 | 0.006 |

The piston wins **not because its prior is highest** — 0.30 vs 0.25 is nearly a tie — but because 7 of
its 12 attempts worked. Slime has the *larger* exploration bonus and still loses. If the piston starts
failing, `Q` falls and slime overtakes automatically; nothing needs scheduling.

## Policy improvement — the engine

After 200 iterations, visit counts normalise to `pi`:

| action | network's `p` | visits | `pi` |
|---|---|---|---|
| piston (1,1,1) west | 0.30 | 140 | **0.70** |
| slime (0,1,2) | 0.25 | 20 | 0.10 |
| stone (1,1,1) | 0.10 | 15 | 0.075 |
| others | 0.55 total | 25 | 0.125 total |

`pi` is better than `p` because it is backed by 200 real outcomes. **Train `p` toward `pi`.** The next
search then starts from a better prior and finds more within the same budget. **No human ever says which
move was good** — improvement comes from search out-performing the network.

## Two knobs against getting stuck

**Dirichlet noise at the root**: `P = 0.75.P + 0.25.Dirichlet(alpha)`. Without it an action rated near
zero is never selected, so no evidence about it ever arrives and **the belief becomes permanent
regardless of truth** — point 20's failure exactly. `alpha ~ 10/actions` (AlphaZero: 0.3 at 35 moves,
0.03 at 250), so **alpha ~ 0.008** at 1,248 actions.

**Temperature** on visit counts when choosing the move to play: `tau=1` proportional, `tau->0` argmax.
This *refines* point 20 — the exploration knob moves from raw scores onto visit counts, strictly better
information.

## The budget problem specific to us

| | actions | iterations | **visits per action** |
|---|---|---|---|
| AlphaZero chess | 35 | 800 | **23** |
| here, R=1 | 1,248 | 200 | **0.16** |

~1,230 actions get zero visits, so `pi` is zero across 98% of the space. Therefore:

1. **The prior does nearly all the work**; search only reorders the top ~20 that `P` surfaced.
2. **`pi` is a sparse target** — nothing is learned about the 1,228 untried actions.
3. **Stage 0's exhaustive labelling is not a warm start but the only *dense* policy signal this project
   will ever have.** Brute force gives the true value of every action; no amount of search can.

**Top-M restriction is a hyperparameter, default unrestricted.** M=50 would give 4 visits per action
instead of 0.16, but anything outside the top 50 becomes unreachable, so a bad early prior locks in.
Decided by *measuring* recall@B both ways against the exhaustive k=2 set — a measurement AlphaZero never
had access to — rather than by intuition.

## Budget unit: simulator calls, not iterations

They coincide in Stage 1. At k=4 in Stage 2, 200 iterations might be only 40 simulator calls. The unit
decides what the headline result says:

> Random finds 3 working modifications in 200 simulator calls. The model finds 5.
> If "200" meant iterations, the model spent only 40 simulator calls — 8x better per simulation.
> If "200" meant simulator calls, both spent 200 — 1.7x better.

Both are defensible and they differ fivefold. **Count simulator calls.** It is the only currency the
control group can also spend (it has no iterations), and it is the actual bottleneck (~1,000/sec on CPU,
while GPU passes are ~1 ms on separate hardware per point 22). Spending 1,000 iterations to save 160
simulator calls is a genuine win, not an accounting trick.

**Breaks if GPU inference ever becomes the bottleneck**; point 22's examples-per-update diagnostic
watches for that.

## Transpositions and two simplifications

**No DAG merging.** Keep an ordinary tree plus point 23's global `canonical_hash -> result` cache: a
duplicate terminal costs a dictionary lookup instead of a simulator call, capturing the whole practical
benefit without multi-parent backup. Point 26 already records the attempt even when the machine is not
re-simulated, so it stays a free training example.

**No sign flipping in backup.** Two-player MCTS negates the value at each level; single-player does not.
A common source of sign bugs that simply does not exist here.

**Subtree reuse:** after playing move 1 the chosen child becomes the new root, keeping its statistics —
roughly halving the effective cost of an episode.

---

# Part 5 — Training

## One training example

A k=2 episode yields **two** examples, both labelled with the same final `z`:

| state | placements | `pi` | `z` |
|---|---|---|---|
| `s0` | `{}` | from the root search | final outcome |
| `s1` | `{piston at (1,1,1) west}` | from the search after subtree reuse | **the same** |
| `s2` | terminal | none — no actions left | — |

Each also carries `B` and `D` from comparing the before and after records.

## The loss

AlphaZero: `l = (z - v)^2 - pi^T log p + c||theta||^2`. Ours:

```
l = w_v . BCE(v, z) + w_p . CE(p, pi) + w_B . BCE(B, b) + w_D . CE(D, d) + weight decay
```

**BCE for value, not MSE.** AlphaZero's `z` is -1/0/+1 so squared error is natural; ours is a
probability in [0,1], where BCE is the proper scoring rule — and it takes soft targets unchanged when the
reward becomes graded, keeping that change free.

## Two DECISIONS problems the framing dissolves

**Point 13's weighting crisis largely evaporates.** It feared `B`'s ~50 numbers would take 80% of
training. But the policy is **one cross-entropy over a 1,248-way softmax**, not 1,248 independent
predictions; apply point 13's own within-answer averaging and all four terms are one number each.
Weights remain a tuning question, not a structural failure.

**Point 14's imbalance machinery is unnecessary on the policy side.** `pi` is a distribution summing to
1, so cross-entropy against it is automatically balanced, and the softmax is *forced* to rank working
moves above failing ones because that is where `pi` puts its mass. **Point 14's objective is achieved by
construction rather than by machinery.**

Still fully required for the **value** head: `z` is 0 about 98% of the time, so plain BCE collapses `v`
to ~0.02 everywhere. Point 14's insistence that the multiplier be computed from the observed rate stands.

## Two different values — a conflation in DECISIONS, resolved

Point 11's answer A is *"does the machine work right now"*. AlphaZero's `v` is *"what reward will this
episode end with"*. Identical at terminals, wildly different elsewhere:

> `s1` = engine + one sticky piston, one placement left.
> **A(s1) = 0** — the piston is dead weight, nothing pushes it, the cycle breaks.
> **v(s1) = 0.58** — one placement remains and slime at (0,1,2) makes it work.

Same state, same head, two answers differing by 50x.

| wrong pairing | failure |
|---|---|
| head means A, MCTS backs up with it | reads 0, abandons the branch — and a piston alone *always* fails, so **every multi-block extension needing a piston is systematically killed**. The banned leaf evaluator arriving through another door |
| head means B, the stop decision reads it | submits a certainly-broken machine believing it is a coin flip — **and point 33's calibration table shows nothing wrong**, because `v` is well calibrated for the question it actually answers. A silent failure with a clean dashboard |

**Settled: one live head meaning expected final reward** (MCTS needs it; the stop decision is deferred).
`Linear(128 -> 2)`, output 1 reserved and unlabelled — because adding a head later would otherwise be a
shape change that makes every checkpoint unloadable at exactly the moment you want to compare against
them.

**No intermediate simulation.** The only reason to simulate `s1` would be to label meaning A, and it has
no consumer:

| possible reason | holds? |
|---|---|
| MCTS needs it | **no** — reward exists only at terminals |
| it might be a discovery | **no** — the k=1 labeller covers every 1-block modification; `canonical_hash` reports a duplicate |
| it gives `B`/`D` labels | only useful if that head exists |
| detect a broken intermediate early | **explicitly banned** — see Part 4 |

**And A becomes free later anyway**: once `stop` is unmasked, episodes end where the model chose, that
terminal is simulated regardless, and the result *is* A's label. There was never a reason to pay early.

## Replay — a distinction point 19 did not make

| label | ages? | why |
|---|---|---|
| `z`, `B`, `D` | **never** | simulator facts. A modification that worked in round 5 works in round 5,000 |
| `pi` | **yes** | an opinion from an older, weaker search |

**This is exactly why AlphaZero has a window** — its only labels are the aging kind. Point 19's blanket
"keep everything" is right for three of four. **Age-decay on the policy term is a hyperparameter,
default flat**, so it can be measured rather than assumed.

## Root oversampling

At k=2 each episode yields two examples and one is always `s0`, the unmodified base. With a single base
machine **half of all training data is the same input**. Chess gives one root example per 80-move game;
we give one per two. Targets differ (search noise, outcomes), so it is repeated *measurement* of one
state — good for estimating it precisely, useless for generalising.

Both responses adopted: **a root down-weight hyperparameter**, and **growing the library early**, which
is the real fix and ties into `DEFERRED.md`'s variety problem.

## The loop, optimiser, stopping

Point 22's decoupling — CPU simulation writing examples, GPU training reading them — **is AlphaZero's
actual architecture** (self-play workers, separate training worker, periodic weight broadcast). We match
it rather than deviate.

AdamW, lr 3e-4, cosine decay, weight decay 1e-4. **Keep training past the first downturn and remember
the best point** (point 18) — the held-out measure is noisy and a dip is usually a wobble. Stopping is
measured on moves held back from the training machine; the handmade generalisation machine is never
spent on a stopping decision.

## Stage 0 is not reinforcement learning

With exhaustive labels and no search it is **ordinary supervised learning** — `pi` is the exact
distribution over working moves, `z` the exact outcome. No exploration, no bootstrapping, no credit
assignment. **RL begins at Stage 1**, when `pi` starts coming from search instead of brute force.

**Edge case:** if no move works on a machine, `pi` is undefined. Skip the policy term for that example
and keep the value, `B` and `D` terms.

---

# Part 6 — The outer loop

## Three timescales

| loop | one iteration is | cost | produces |
|---|---|---|---|
| **episode** | k placements on one base machine | ~200 simulator calls | 2 policy examples, ~200 value examples, ~200 attempt records |
| **round** | pick a base machine, run N episodes on it | minutes | library candidates |
| **lifetime** | library grows, k rises, model improves | days | the thing we actually want |

## An asymmetry to expect, not to misdiagnose

One k=2 episode produces **2 policy examples** and **~200 value examples** (every terminal the search
simulated). **The value head sees 100x more data than the policy head**, so `v` will look like it is
learning well while `p` appears to plateau. Point 13's per-answer tracking will show exactly that shape;
it is the data rates, **not a broken policy head**.

It also reinforces Part 4's conclusion: exhaustive brute force is the only *dense* policy signal, and
search will never replace it.

## Two destinations for a working modification

A search runs 200 simulator calls and finds ~4 working modifications — not just the one the episode
ended on. **All of them are recorded and all non-cargo ones enter the library**; nothing paid for is
discarded.

| destination | what goes there | why |
|---|---|---|
| **attempt log** | **every** simulated candidate, working or not | a true label. Training on cargo is fine — a block riding along is physics the model should know |
| **machine library** | only **non-cargo** working modifications | building on cargo compounds |

**Why cargo must never become a parent.** `OPEN.md`'s free detector — same period, same shift, every
original block behaving identically at every tick — is a comparison of two records already held. If
cargo enters the library it becomes a base machine and its children inherit it:

| generation | useful blocks | cargo blocks |
|---|---|---|
| 1 | 1 | 2 |
| 10 | 10 | **20** |

Three times the cell items to encode, three times the simulation cost, and the model's input becomes
mostly noise. **Cargo accumulates monotonically because nothing removes it.** Point 26's
machine-index / attempt-log split already provides exactly these two destinations.

Accepted risk of admitting all four discoveries: they are near-siblings, so library *count* grows faster
than library *variety*. That is `DEFERRED.md`'s binning problem arriving early, not a new issue — and
the faster growth directly attacks the root-oversampling problem, where more distinct base machines is
the real fix.

## The loop with no external correction — the real structural weakness

```
model chooses what to try  ->  what works enters the library
        ^                                    |
        |                                    v
   model trains on  <---------------  the library IS the training data
```

AlphaZero's curriculum is self-balancing: the opponent is always exactly your strength. **We have no
opponent**, so **the model's biases shape its own future training data with nothing outside the loop to
correct them.**

Concretely: glass is inert, pushable and safe, so the model learns glass is low-risk, proposes it often,
and it often works. The library fills with glass. The model now trains mostly on glass, stops proposing
observers, never gets more observer data, never improves at observers, never proposes them.

This is sharper than `DEFERRED.md`'s variety framing — that is about design diversity, this is about the
**training distribution being self-selected** — and it is more dangerous because it looks healthy from
every angle: high success rate, growing library, falling losses.

**The defence already exists**, and this gives point 32's uninformed control group a third job:

| point | use |
|---|---|
| 32 | measurement: does the model beat random |
| 33 | an unbiased sample for the calibration table |
| **here** | **the only unbiased training data the system produces** |

Three uses for one 5% expenditure — a strong argument for keeping it permanent, and possibly raising it.

## The three-way budget split

| share | move chosen by | purpose | default |
|---|---|---|---|
| top-ranked | highest search value | find good machines | 60% |
| model-sampled | sampled from `pi` at temperature `tau` | training variety | 35% |
| **uninformed** | ignores the model entirely | baseline + calibration + unbiased data | 5% |

All adjustable; the exact shifts are expected to come from training rather than being set now. The third
must stay *genuinely* uninformed — the sampled portion still uses model scores and cannot double as the
control.

## Escalating k

A fixed constant now, point 24's size weights later — the same plan at different times, not a conflict:

| when | k |
|---|---|
| Stage 0 | fixed at 1 |
| Stage 1 | fixed at 2 |
| Stage 2+ | weights across sizes, e.g. 80/15/5 over k=1/2/3 |

Point 24's argument for weights over a hard switch holds: single-block modifications never exhaust,
because every working one adds a machine whose own single-block space then opens up; and useful
modifications may need several blocks together — a piston is only useful with something to push.

## What "getting better" looks like — and the trap

| measure | trustworthy? |
|---|---|
| recall@B versus the uninformed control | **yes** — exact ground truth exists |
| `p`'s recall on a held-out machine | **yes** — exhaustively labelled |
| calibration table drift | **yes** — measured on the control group |
| the visualiser | **the only quality signal**, and it needs a human |
| **library size** | **no — the trap** |

`OPEN.md` records why: `canonical_hash` dedupes on structure, so a machine plus one carried block hashes
differently and **trivial growth registers as discovery**. That is exactly how the GA convinced itself
it was working. Library size rising means the loop is running, nothing more.

---

# Part 7 — Verification

## AlphaZero's answer does not transfer

AlphaZero verifies itself by **playing its previous version** — Elo needs no external ground truth and
never saturates. We have no opponent. What we have splits sharply:

| | measurable? |
|---|---|
| is a modification **correct** (machine still works) | **exactly** — the simulator is ground truth |
| is a modification **good** (worth having) | **not at all** — the deferred usefulness problem |

That gap is the honest summary of this project's verification story.

## Three success numbers, never one

The dangerous failure is **success rate rising while everything found is cargo** — the GA's history.

| # | count | cost | relation |
|---|---|---|---|
| 1 | working | free | |
| 2 | working **& non-cargo** | free — compares two records already held | superset of #3 |
| 3 | working **& not removable** | k or 2^k-1 simulator calls | strictest |

**cargo is a subset of removable**: a cargo block changed nothing, so removing it restores a working
machine; but a block can change the period *and* be removable. So **#2 is a cheap over-estimate of #3**,
and the gap between them is itself informative.

Expect #1 to rise clearly while #2 and #3 sit near zero early. Point 32 already flagged this — it is
the model learning "don't break things" before "build things", **not the model doing nothing**.

## Trimming cadence — closes a DEFERRED sub-question

| scope | cost at k=2 | when |
|---|---|---|
| **new blocks only** | 2^2-1 = **3 sims, exact** | **every library candidate** |
| whole machine | 2^n-1 — hopeless | periodic maintenance pass only |

3 calls per discovery, ~12 per episode against 200 — **6% overhead** for the strictest success measure
available. `DEFERRED.md`'s combinatorial warning applies only to whole-machine trimming, which nothing
requires per-discovery.

## Probes — two mechanisms DEFERRED does not specify

**The probe set is frozen once created.** Adding a probe makes every historical number incomparable —
you can no longer tell whether month 6 is better than month 1 or just measured differently. New probes
start a new named set with its own history.

**Probes are built outside the library's lineage.** `DEFERRED.md` requires probes never be trained on,
*including as descendants*. A probe derived from our base machine would very likely be rediscovered and
quietly trained on. Building them from unrelated hand-made machines makes this structural rather than a
blocklist that must work perfectly forever; a hash blocklist stays as a safety net that should almost
never fire.

## What runs when

These differ in cost by five orders of magnitude, so they cannot share a cadence:

| check | cost | cadence |
|---|---|---|
| mechanic probes | ~20 forward passes | every checkpoint |
| calibration table | reads the attempt log | every checkpoint |
| held-out `p` recall | 1 forward pass against stored labels | every checkpoint |
| recall@B vs control | 2 x B simulator calls | daily |
| exhaustive k=2 ground truth | 13 min | once per test machine, permanent |
| trimming survival | 3 sims per discovery | continuous, 6% overhead |

## Alarms — only one kind can be automated

**Mechanical** (the loop has stopped working): success rate at zero, duplicate rate at everything,
examples-per-update in the thousands, per-answer accuracy flat early, crash counts. **Thresholds that
fire**, not dashboard numbers — nobody watches a dashboard for three days.

**Quality** (the loop works perfectly and produces rubbish): nothing automatic catches this. It goes
through the visualiser and needs a person — which is why point 28's slow, sampled, parent-and-child
publishing matters.

## Past-self comparison — the Elo analogue

The random control **saturates**. Measured on a held-out machine at B=200:

| | random | model | ratio |
|---|---|---|---|
| day 1 | 3 | 4 | 1.3x |
| day 7 | 3 | 31 | 10.3x |
| day 30 | 3 | 44 | 14.7x |
| day 60 | 3 | 47 | 15.7x |

Random never moves, and the model compresses against the machine's ceiling — 14.7 to 15.7 over a month
is indistinguishable from noise, since three-out-of-200 is a small denominator with large relative
variance. Running **v3000 against v900 on the same machine, budget and seeds** answers "did the last
month do anything" directly, and does not compress because **the comparison target moves with you**.
This is why Elo works: you always play someone near your own strength.

**Decided: keep every periodic checkpoint.** ~1 MB each, ~1.4 GB over two months of hourly saves. This
is the only irreversible part — the harness can be written whenever the random baseline goes flat, but
only if the checkpoints exist. It also allows bisecting over checkpoints when a training change makes
things worse.

Caveat: repeated measurement on a held-out machine slowly erodes its independence. Another argument for
`DEFERRED.md`'s MUST EXPAND on building more of them.

---

# Part 8 — Build order

Per point 26's self-containment rule everything lives in `RL gym/` and imports nothing from
`genetic_ml`, `transformer_gym` or `util tools`. The C++ simulator stays where it is.

## The arithmetic that sets build order

Exhaustive search over unordered k-block modifications at ~1,000 sims/sec, R=1:

| k | modifications | exhaustive time |
|---|---|---|
| 1 | ~1,250 | **1.3 seconds** |
| 2 | ~780,000 | 13 minutes |
| 3 | ~3.2 x 10^8 | 3.8 days |
| 5 | ~4.0 x 10^14 | ~12,700 years |

1. **At k=1 the model is useless for discovery** — brute force gives the complete exact answer in
   seconds. What k=1 gives instead is a **perfect supervised curriculum**: the true optimal policy
   target, free, and *dense* over every action.
2. **k=2 is affordable once**, as an exact test set rather than a strategy.
3. **k>=3 is where the model is the only option.**

## The three stages

### Stage 0 — network only, no search
Brute-force every single-block modification. Train `f` supervised: `pi` <- exact distribution over
working moves; `z` <- exact `validCycle`; `B`, `D` <- from the record. A **perfect** policy-improvement
operator. Verifies encoder, heads, loss balance and data path before any search code exists.
*Gate:* `v` calibrated (point 33), `p` beating a uniform prior on a held-out machine.

### Stage 1 — add MCTS, ground-truth leaves
k=2, PUCT, every leaf a real simulation. `v` orders the prior only, and **is measured without being
given authority**: log its prediction at every node, compare against the exact terminal result the
search later finds. Point 31 applied inside the tree.
*Gate:* **recall@B** against brute-forced 2-block ground truth, versus random ordering (point 32).

### Stage 2 — value bootstrapping, k >= 3
Leaves can no longer be simulated, so `v` evaluates them. Full AlphaZero. Point 33's calibration becomes
load-bearing: in stages 0-1 only `v`'s *ranking* mattered; now its *value* is backed up the tree.

## The verdict JSON already carries the reward

`Result` (`simulator.h:19`), emitted as one JSON line per candidate on stdout, contains:

| field | gives us |
|---|---|
| **`validCycle`** | **the reward, directly** |
| `period`, `shift`, `finalShift` | **the cheap half of the cargo detector, free** |
| `elapsedNs` | per-candidate timing — measures the "~1 ms" assumption directly |
| `working`, `cycles`, `settled`, `ticks` | diagnostics |

So the first milestone needs **no record decoding, no sim-log, and no C++ change**.

## Milestone 1 — RESULT

Ran. `simple_observer_engine`, R=1, all 988 legal single-block modifications simulated.

```
WORKING                314   =  31.78% of legal actions
  of which maybe cargo 313   (same period AND same shift as base)
NON-CARGO WORKING      1     =  0.10%
```

**Both branches of the decision table fired at once**, which is the finding. The raw rate says the task
is easy; the non-cargo rate says it is nearly impossible. The gap between them **is** the
trivial-extension problem, arriving through the binary reward exactly as `OPEN.md` predicted:

> *"With 'does it still work' as the only objective, a block riding along genuinely works. Triviality is
> the absence of a usefulness objective, not a model failure."*

### Three findings that change what the model has to learn

**1. Success is driven by the cell, not the block.** Eleven of the 26 candidate cells accept 66-92% of
anything placed in them; the other fifteen accept 0-3%. Position dominates identity — which is the
strongest possible argument for point 3's decision to supply targeted geometric facts rather than
coordinates, since *where* is nearly the whole answer.

**2. The physics shows up in the palette breakdown**, unprompted, which is strong evidence the encoding
is right:

| entry | works | of | why |
|---|---|---|---|
| stone, glass, lamp, fence gate, trapdoor, observers, retracted pistons | 11 | 26 | inert or movable — they ride along |
| `redstone_block` | 9 | 26 | it powers things, so it changes behaviour |
| `glazed_terracotta` | 4 | 26 | **`PushOnly`** — pushable, never pulled or dragged |
| extended pistons | 2-6 | 26 | **immovable** while extended |
| `air` | **0** | 6 | **every block in this machine is load-bearing** |

`air` at 0-for-6 is the important one: there is nothing dead in the base machine to trim, which is what
makes it a sound starting point.

**3. The single non-cargo discovery is genuinely non-trivial.** An extended sticky piston at (-1,0,2)
facing east **reverses the machine's direction** — shift flips from (1,0,0) to (-1,0,0) at the same
period. One such modification exists in the entire single-block space.

### What this means for the plan

- **A binary reward is not merely imperfect here, it is 313:1 misleading.** 99.7% of the positive signal
  is cargo. Whatever replaces it, the graded reward is now the highest-value open item rather than a
  refinement.
- **The cheap cargo detector is enough to see this**, using only `period` and `finalShift` from the
  verdict. The full detector needs records and arrives in Phase B; it can only move the count down.
- **Stage 0 has 314 positive examples, not 1.** The supervised bootstrap is viable — the model can learn
  "do not break things" from this data. What it cannot learn from it is "build something useful", and
  the plan already said that was the realistic first win.

### Free diagnostics collected alongside

| | |
|---|---|
| `elapsedNs` per candidate | **0.26 ms** on the base machine, ~4x faster than the 1 ms assumption |
| throughput | **572 sims/sec** at 12 workers (after fixing a chunking bug — see `VERIFICATION.md`) |
| `canonical_hash` duplicates | **0** across all 988 — no single-block action reaches another one's machine |
| `working` vs `validCycle` disagreements | **0** — they never differ on this machine |
| simulator refusals | **21**, all *"structural-verify trigger must point at an observer or a piston"* |

The 21 refusals are a clean end-to-end confirmation of the compound extended-piston action: nine
overwrite the trigger cell directly, and **twelve land an extended piston's *head* on it** — from
(-1,1,1) east, (0,0,1) up, (0,1,0) south, (0,1,2) north, (0,2,1) down and (1,1,1) west, every one of
which puts its head at exactly (0,1,1). The C++ side independently notices. Both sides agree about where
a head goes.

## Phases

```
PHASE A — no model, no C++ changes                                            [DONE]
   game.py -> sim.py -> labeller.py       + test_unit/ (98 tests)
   +------> MILESTONE 1: 31.78% raw, 0.10% non-cargo

PHASE B — needs the C++ snapshot flag
   C++ per-tick snapshots (point 27) --+
   record decoder (copied, point 26) --+--> graph.py -> net.py -> train.py
   +------> MILESTONE 2 / Stage 0 gate: v calibrated, p beats a uniform prior on a held-out machine

PHASE C
   search.py (PUCT) -> store.py (machine index + attempt log)
   C++ records to stdout (point 25) — only when throughput demands it
   +------> MILESTONE 3 / Stage 1 gate: recall@B beats the uninformed control on the k=2 test machine
```

| file | contents |
|---|---|
| `rlgym/game.py` | Part 2 — palette, action enumeration, legality masks, encoding, `canonical_hash` |
| `rlgym/sim.py` | simulator process pool — **must carry over the MinGW PATH fix** or every candidate dies silently with no error output |
| `rlgym/labeller.py` | exhaustive k=1 / k=2 — Stage 0 data and Stage 1 test sets |
| `rlgym/graph.py` | record + snapshots + placements -> graph (Part 3) |
| `rlgym/net.py` | trunk + `p` / `v` / `B` / `D` heads (Part 3) |
| `rlgym/train.py` | losses and weighting (Part 5) |
| `rlgym/search.py` | swappable: stage 0 none, stage 1 PUCT + sim leaves, stage 2 PUCT + value leaves |
| `rlgym/store.py` | machine index + attempt log (points 26, 29) — **Phase C**, not before |

`store.py` waits because Milestone 1 is a one-off run whose output is a JSON dump; a machine index and
append-only attempt log are what the *loop* needs, and there is no loop until search exists.

## Which C++ change blocks what

| change | blocks | why |
|---|---|---|
| **per-tick snapshots** (point 27) | **Phase B** | cell items are "what is at this cell at this tick"; the Python replay alternative was rejected |
| records to stdout (point 25) | **nothing until Stage 1** | a throughput fix. Phase A writes no sim-logs; Stage 0 writes ~1,250 files once per machine — seconds of churn. It matters when the loop runs at 1,000/sec for days |
| framing + binary mode | ships with the stdout change | — |

**Only one of the three C++ changes is on the critical path**, and it is not the one that started as the
headline.

---

# Reserved-but-frozen mechanisms

**Reserve the mechanism, freeze the value.** All of these are free — a masked slot or an always-true
flag, never added complexity. The later change is a config edit, not a shape change, so saved models
stay loadable under point 1's version check.

| # | mechanism | frozen at |
|---|---|---|
| 1 | `stop` slot in the policy head | masked |
| 2 | reward as float + BCE value head | 0.0 / 1.0 |
| 3 | per-cell head, cell count is data | R = 1 |
| 4 | weight-tied trunk | T = 8 |
| 5 | bucketed time-gap relation types | always gap 1 |
| 6 | `view_is_complete` on the summary item | always true |
| 7 | palette slots 39 -> 48 | 9 masked |
| 8 | relation slots ~35 -> 64 | 29 unused |
| 9 | tick cap, refuse loudly above it | 32 ticks |
| 10 | `ticks_to_encode(record)` as one function | `range(cycle_length)` |
| 11 | second value-head output | reserved, unlabelled |

**Cell items at every tick for now.** The economy (only tick 0 and ticks where the cell changed) is
deferred, and switching later is free *because* the time-skip relation types already exist at gaps 1, 2,
4, 8, 16, 64, 256 — a changed-only edge with gap 7 buckets into a type that is already trained. Zero
shape change. Caveat: an item's meaning shifts from "the cell at tick 5" to "the cell from tick 5 until
it next changes", which is a data-distribution change wanting a retrain — but a retrain, not a load
failure. This is `SLOWDOWNS.md` #3, the known hotspot to **measure before optimising**.

**The tick cap is the time half of `OPEN.md`'s windowing solution.** Capping forces the code to handle
"you are seeing part of this machine's life" from day one, so the spatial half later slots into the same
place instead of being retrofitted into code that assumed completeness. `view_is_complete` is what stops
the model learning that the last encoded tick is the end of the cycle.

**Where this does not help:** the feature list itself (point 1 accepts a retrain, which the version check
makes loud) and `d_model` (a real commitment).
