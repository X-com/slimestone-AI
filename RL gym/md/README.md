# RL gym

Home of a new model that modifies working flying machines and gets better at it over time.

Nothing is built yet. This folder holds the design decisions and their reasons, so implementation
starts from settled ground rather than re-deriving the same arguments.

## The goal

A model that proposes modifications to a working flying machine, has them verified by the C++
simulator, and improves from the results.

**The model places the blocks itself.** There is no separate random generator feeding it candidates.

## The loop

At each step the model scores **every** legal placement and picks one. At single-block scale that is
about 1,350 options — cheap enough to score in a single pass, so "the model chooses" is literal.
Multi-block is the same step repeated: place a block, re-score everything, place the next.

Then simulate the result, learn from what happened, and rank better next round.

Randomness enters only as **exploration** — sometimes not taking the top choice, so the model keeps
seeing things outside its current beliefs. That fraction starts high (an untrained model's ranking is
meaningless anyway) and decays as it improves. Nothing switches over; it is one loop throughout, with
the exploration dial turning down.

## What to expect first

The realistic first win is a **filter, not a designer**. Learning to stop proposing obviously-doomed
placements is far easier than learning to propose useful ones — the doomed cases are the majority and
have strong local signals. That filter is what makes everything downstream affordable.

Trivial-but-working extensions will still appear. That is not a model failure: "does it still work" is
the only objective it has, and a block riding along genuinely works. See `OPEN.md`.

## Files

| file | contents |
|---|---|
| `QUESTIONS.md` | what each of the 33 design points asks, and why it matters |
| `DECISIONS.md` | what was decided for each point, with reasons — **the main reference** |
| `DEFERRED.md` | problems deliberately left unsolved, with notes for whoever picks them up |
| `SLOWDOWNS.md` | every identified performance risk, since training speed matters |
| `DESIGN.md` | the earlier high-level design, written before the 33 points |
| `OPEN.md` | the earlier high-level deferred items, plus a log of corrections made while planning |

`DESIGN.md` and `OPEN.md` came first and are the shorter overview. **Where they disagree with
`DECISIONS.md`, `DECISIONS.md` wins** — it was written later and in more detail. The clearest example
is the relationship kinds: `DESIGN.md` lists five, and the settled answer is seven (power and
quasi-connectivity were added in point 4).

## Status

All 33 design points are settled or explicitly deferred. Three changes to the C++ simulator fall out of
them — records written to standard output instead of one file per candidate, per-tick board snapshots
behind a flag, and the framing and binary-mode details both require.

Implementation has not started.

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
