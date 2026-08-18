# The 33 points — what each one asks

What must be decided, why it matters, and what goes wrong later if it is decided badly. The answers
are in `DECISIONS.md`. See `README.md` for term definitions.

---

# Group 1 — What the model is given to look at

## Point 1 — Which numbers are stored on each item

**What must be decided:** the exact list of numbers stored on every item, where one item is a single
block at a single tick.

Candidates fall into three kinds. *Read directly off the layout*: block type; whether the cell is
empty; facing; whether this is the trigger block. *Read off the record of the machine running*: push
group size at this tick; whether it is moving; direction; whether it is powered; whether an observer
watches it; whether it is mid-flight. *Worked out from block type by table lookup*: whether it can be
pushed, whether it provides power, whether it is a full solid cube — all already in
`block_registry.h`'s `BlockData`.

**Why it matters:** this list is the model's entire view. Anything not in it must be worked out by
following connections, or cannot be known at all.

**What goes wrong:** changing the list later makes every saved model unusable. Worse, if the *length*
happens to stay the same, a saved model loads successfully and produces nonsense, because column
three now means something different than during training.

## Point 2 — How block type is represented

**What must be decided:** whether "a piston facing north" is one kind of thing, or two pieces of
information — "it is a piston" and "it faces north".

19 named block types exist. Pistons, sticky pistons and observers face 6 directions; rails have 10
shapes; trapdoors vary by facing, half and open state. Counting variants gives 60 or more.

**Why it matters:** if every variant is separate, what is learned about a piston facing north teaches
nothing about one facing up. With limited data that wastes most of it.

**Additional problem:** the list is not closed. `stone` and `glass` appear in machines but have no
`block_registry.h` entry, so they fall through to defaults. Assuming a fixed list means an unexpected
block either crashes or is silently treated as air.

## Point 3 — Whether an item knows where it is in space

**What must be decided:** whether items carry position numbers, and measured from what.

Three possibilities: position in the Minecraft world; no position at all; or an offset from a chosen
reference point.

**Why it matters:** distance and direction not given as numbers must be worked out by following
connections one step at a time, and counting steps is something this kind of model does poorly.

**What goes wrong:** world position changes every cycle because the machine flies. And a reference at
the bounding box edge moves whenever the machine changes size, and becomes meaningless if a window of
a large machine is used later.

## Point 4 — How the different kinds of connection are handled

**What must be decided:** whether each kind of connection gets its own learned numbers, or all share
one set with a type label.

Kinds identified: same block at the next tick; two blocks touching; two blocks in the same push
group; a slime block and what it drags; what caused what, from the simulator's record.

**Separate question inside this point:** which kinds to include at all. `transformer_gym/encode.py`
already computes two more — whether redstone power reaches a piston, and quasi-connectivity, the rule
where a piston activates from power reaching the block above it.

**What goes wrong:** leaving out quasi-connectivity forces the model to rediscover a non-local,
non-obvious rule from examples, when it is already computed elsewhere.

## Point 5 — Whether the proposed change is part of the input or part of the output

**What must be decided:** when the model judges "set this cell to this block type", whether that
change is fed in as part of what it looks at, or read out as part of its answer.

*Fed in*: mark the cell, run the model, read one score. The model can follow that block's
consequences. Costs one run per move considered — roughly 2,000 runs to judge every option.

*Read out*: run once on the unchanged machine, read a score for every cell-and-type pair. Roughly
2,000× cheaper, but the model never sees the proposed block and must judge from the unchanged state.

**Why it matters:** this decides whether judging every move costs one run or two thousand, which
decides whether the approach is affordable.

**What goes wrong:** the expensive version makes the intended loop too slow. The cheap version, chosen
carelessly, makes multi-block modification impossible, because at the second placement the model must
know what it placed first.

---

# Group 2 — The shape of the model itself

## Point 6 — How information travels between connected items

**What must be decided:** the rule by which an item combines information from its connected items.
Options include treating every connection equally, weighting by connection kind, or weighting by what
the connected item is actually doing.

**Why it matters:** this is the core computation. Everything the model concludes comes from repeated
rounds of items sharing information.

## Point 7 — How many rounds of information sharing

**What must be decided:** how many times information passes along connections before the answer is
read. Each round moves information one step further.

**Why it matters:** this sets the maximum distance across which the model can relate two things. It
is why a grid-based image-style model was rejected — four rounds relate things four cells apart, and
a piston pushes twelve blocks.

**What goes wrong:** too few rounds and the model physically cannot see the far end of a push group,
no matter how much data it gets.

## Point 8 — How large the model is

**What must be decided:** how many numbers each item carries internally, and the resulting total
count of learned numbers.

**Why it matters:** too small and it cannot represent what it needs to; too large and it memorises the
training machines instead of learning rules.

## Point 9 — Whether time is treated differently from space

**What must be decided:** whether "the same block at the next tick" is handled by the same rule as
physical connections, or given special treatment.

**Why it matters:** time has an order — earlier causes later — while touching is symmetric. Treating
them identically discards that.

## Point 10 — Standard construction details

**What must be decided:** whether each round adds to the previous result rather than replacing it,
how numbers are kept in a stable range, which shaping function is applied — and whether all rounds
share the same learned numbers or each has its own.

**Why it matters:** most of these barely change what the model can express but strongly affect whether
training succeeds at all. The shared-versus-separate question is the exception and is a real decision.

---

# Group 3 — What the model produces and how it is scored

## Point 11 — Which separate answers the model produces

**What must be decided:** the full list of distinct answers. Candidates: whether the machine still
works; which blocks behave differently; where the first failure happens; what kind of failure.

**Why it matters:** each answer is separate work and separate learning signal. Predicting more from
the same simulation extracts more value from each simulation.

## Point 12 — The shape of the score for every possible placement

**What must be decided:** the arrangement of numbers giving a score for each cell-and-block-type
combination, how "place nothing, submit now" is represented, and how the many per-tick items for one
cell reduce to a single score.

**Why it matters:** this is what the search reads to choose a move, and it must allow illegal options
to be excluded cleanly.

## Point 13 — How each answer's wrongness is measured, and how they are combined

**What must be decided:** the measurement per answer type, and how much each contributes to the total.

**What goes wrong:** one answer with more numbers in it dominates training purely by size, and the
others are effectively ignored.

## Point 14 — Handling the fact that most changes break the machine

**What must be decided:** how training compensates for the large majority of examples having the same
answer.

**Why it matters:** without compensation the model learns to always answer "breaks", which is correct
most of the time and completely useless.

---

# Group 4 — Training data and the training process

## Point 15 — What is stored on disk per simulated attempt

**What must be decided:** exactly what is saved after each simulation, so no machine is simulated
twice.

**Why it matters:** simulation is the main cost. Re-simulating because something was not saved is
pure waste.

## Point 16 — How several machines are processed together

**What must be decided:** how machines of different sizes are combined into one group for training,
given each produces a different number of items and connections.

## Point 17 — How data is divided for honest testing

**What must be decided:** the rule dividing data into what is learned from and what is tested on.

**What goes wrong:** if moves from the same machine appear in both, the test measures memory rather
than understanding, and the model looks far better than it is.

## Point 18 — The training procedure settings

**What must be decided:** the method for adjusting learned numbers, how large each adjustment is, how
that changes over time, and when to stop.

## Point 19 — How the model is retrained as new data arrives

**What must be decided:** whether the model is trained fresh from all data at intervals, or updated
continuously as results come in.

**What goes wrong:** continuous updating on recent results only can cause the model to forget earlier
lessons.

---

# Group 5 — The loop that chooses moves and improves

## Point 20 — How often the model does not take its own best choice

**What must be decided:** what fraction of moves are chosen other than the top-scoring one, and how
that changes as the model improves.

**Why it matters:** a model that only tries what it already rates highly only receives confirmation,
and never discovers where it is wrong.

## Point 21 — How many options are judged, and how many partial builds are kept

**What must be decided:** how many candidate moves are simulated per round, and for multi-block
modifications, how many partly-built modifications are carried forward at each step.

## Point 22 — How often the model is retrained during the loop

**What must be decided:** whether retraining happens after a fixed number of simulations, after a
fixed time, or continuously.

## Point 23 — How duplicate machines are recognised

**What must be decided:** how to detect that two different sequences of placements produced the same
final machine.

**Why it matters:** placing A then B gives the same machine as B then A. Without detection, work is
duplicated and stored data becomes skewed toward machines reachable many ways.

## Point 24 — When modifications are allowed to grow larger

**What must be decided:** the rule that increases the permitted number of blocks per modification, and
what evidence triggers each increase.

---

# Group 6 — Connections to code that already exists

## Point 25 — Driving the simulator

**What must be decided:** how machines are sent to the simulator and results collected. The simulator
already accepts many machines in one run through its input stream and writes one record per machine
(`main.cpp:59`), so this is about batch sizes, file handling and cleanup.

**Known concern:** at ~1,000 machines/sec, creating 1,000 files/sec on Windows is the likely slow
point, not the simulation.

## Point 26 — How a machine is represented in Python

**What must be decided:** whether to reuse the existing candidate representation and binary encoding
in `genetic_ml/compact_format.py`, or define something new.

**Why it matters:** reusing it means the live viewer and existing stored collections work without
translation.

## Point 27 — Rebuilding the machine's state at each tick

**What must be decided:** whether to reuse `transformer_gym/state.py`'s tick-by-tick state rebuilder,
already checked by `util tools/check_state_builder.py`, or obtain per-tick state another way.

## Point 28 — Sending discovered machines to the live viewer

**What must be decided:** how the loop publishes to the existing viewer, and what counts as new enough
to be worth publishing.

**Known concern:** at 1,000 simulations/sec, publishing everything floods the viewer with machines
differing by one block.

## Point 29 — Saving and resuming

**What must be decided:** what is written when a run is interrupted, and what is needed to resume
without losing collected simulations.

---

# Group 7 — Checking that the model is actually learning

## Point 30 — The targeted mechanical tests

**What must be decided:** the specific small machines built by hand to test single mechanics.

The clearest example: a piston with eleven blocks. Adding a twelfth should work and a thirteenth
should fail. The test is whether the model's answer changes at exactly the right count. If it changes
at ten, or fifteen, or never, it has learned a rough association rather than the rule — and no overall
score would reveal that.

## Point 31 — Which numbers are recorded and where

**What must be decided:** what is logged each round and in what format, so progress over a long run
can be examined.

## Point 32 — The comparison the model must beat

**What must be decided:** the exact definition of choosing moves without the model, so "the model
helps" is a measured claim rather than an impression.

## Point 33 — Deciding whether an answer is trustworthy

**What must be decided:** how to check that when the model reports high confidence it is actually
right that often.

**Why it matters:** the search orders moves by these scores. If confidence does not track real success
rate, the ordering is wrong even when yes-or-no answers are frequently correct.
