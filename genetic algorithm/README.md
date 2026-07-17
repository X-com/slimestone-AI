# Genetic ML Simulator Bridge

This project runs flying-machine JSON candidates through the C++ stream simulator in
`../cpp simulator`.

The current milestone is the simulator bridge, not model training:

```text
candidate JSON line -> long-lived C++ worker -> result JSON line
```

## Getting Started

New to this project? Here's the shortest path to a working run.

1. **Open the right folder.** In VS Code, open `genetic algorithm` itself (not the
   whole repo) as the workspace folder - the `.vscode/` config in this folder
   (pytest discovery, debug configs) is scoped to it.
2. **Pick a Python interpreter.** Any Python 3.11+ works; there are no third-party
   dependencies to install. Use VS Code's interpreter picker (bottom-right of the
   status bar, or `Ctrl+Shift+P` -> "Python: Select Interpreter") if one isn't
   already selected.
3. **Check the simulator .exe exists.** This project drives a prebuilt C++ binary
   at `../cpp simulator/build/cpp_simulator_stream.exe`. If that file is missing,
   run `../cpp simulator/build-cpp.bat` first (it needs the msys64/ucrt64 MinGW
   toolchain).
4. **Run the simulator bridge.** Open `main.py` and run it (`F5`, the Run/Debug
   button, or Code Runner). This feeds every machine under `data/working/*.json`
   through the simulator and writes `data/outputs/cpp_results.jsonl`. Watch the
   terminal for a line like `Wrote <N> results ... ok=<N> working=<M>` (with
   `M > 0`) - that confirms the simulator is reachable and producing results.
5. **Run the genetic algorithm.** Open `main_ga.py` and run it the same way. It
   seeds its starting population directly from every file in `data/working/`, so
   there must be at least one machine there already (this repo ships a handful).
   Progress prints one line per generation; when it finishes, check
   `data/outputs/ga_archive.jsonl` for the full discovery log, or `data/working/`
   itself for the newly discovered machines as individually reusable files.
6. **(Optional) Run the tests.** Either use VS Code's Test Explorer (enabled by
   `.vscode/settings.json`), or run `python -m pytest` from this folder in a
   terminal. No simulator or built `.exe` is needed for the tests - they only
   exercise the pure-Python mutation/population/archive logic.

If step 4 or 5 reports every candidate as failing/crashing/`working=false` across
the board, see the Troubleshooting section near the bottom of this file before
assuming something is wrong with the candidates themselves.

## Run From VS Code

Open this folder (`genetic algorithm`) in VS Code, edit the variables at the top of
`main.py` or `main_ga.py`, then run the file - via the Python extension's Run/Debug
button, `.vscode/launch.json`'s "Run simulator bridge (main.py)" /
"Run genetic algorithm (main_ga.py)" configs (F5), or the Code Runner extension.
All three work the same way since both entry points resolve every path from
`Path(__file__)`, so they don't depend on the terminal's working directory.

The most important setting is:

```python
SIMULATOR_EXE = PROJECT_ROOT.parent / "cpp simulator" / "build" / "cpp_simulator_stream.exe"
```

The C++ process is kept alive and reused for many candidates. `WORKER_COUNT` controls
how many simulator processes are started in parallel.

`.vscode/settings.json` enables pytest discovery (`tests/`) in the Test Explorer.
Any Python 3.11+ interpreter works - there are no third-party dependencies.

## Candidate Format

Each candidate is a JSON object with at least:

```json
{
  "id": 1,
  "trigger": {"x": 0, "y": 0, "z": 0},
  "blocks": [{"x": 0, "y": 0, "z": 0, "state": 152}]
}
```

Extra fields such as `name` and `path` are preserved in Python-side dataset output.

## Folder layout

```text
data/
  working/    one JSON file per verified-working flying machine (id/trigger/blocks).
              main.py reads its input from here; main_ga.py seeds from here and
              writes every newly discovered working machine back into it.
  crash/      candidates that crashed the simulator, as bare candidate JSON
              (crash_0001.json, crash_0002.json, ...).
  hangs/      candidates that hung the simulator, as bare candidate JSON
              (hung_0001.json, hung_0002.json, ...).
  outputs/
    cpp_results.jsonl   full result dataset from the most recent main.py run
                         (candidate + result + simulator metadata).
    ga_archive.jsonl    full discovery log from main_ga.py (every distinct working
                         machine ever found, with generation/origin/result stats -
                         this is the audit trail; data/working/ is the reusable pool).
```

`data/working/` is meant to only ever contain verified-working machines - drop a
hand-authored or externally-sourced candidate JSON in there and it becomes part of
the seed pool the next time `main.py` or `main_ga.py` runs.

## Genetic Algorithm

`main_ga.py` runs a mutation-only genetic search for new working flying machines.
Fitness is the simulator's boolean `validCycle` field only - there is no other signal. (`validCycle`
is the ground-truth check: does the machine settle and end up an exact translated copy of its
starting layout - not the older `working`/`cycles` hash-based cycle detector, which can report a
repeat was detected without the final layout actually lining up.)

```text
data/working/*.json -> smallest N as seed population
  -> per generation: mutate each lineage into several children
  -> simulate children through SimulatorPool
  -> working children are admitted into the population (evicting the largest
     lineage once the pool is full), appended to data/outputs/ga_archive.jsonl,
     and (if genuinely new) saved as their own file in data/working/
  -> a small fraction of non-working children are kept for a few more generations
     of mutation anyway, since some working designs are only reachable through a
     temporarily-broken intermediate
```

Every newly discovered working machine is written to `data/working/` as
`<DISCOVERED_NAME_PREFIX>_0001.json`, `..._0002.json`, etc. - `DISCOVERED_NAME_PREFIX`
(default `"discovered"`) is a variable at the top of `main_ga.py`, so it's just the
default name a discovery gets until you rename the file to something more
descriptive. The postfix always continues from whatever's already on disk, so
re-running never overwrites an earlier discovery.

### Mutation palette

Mutation only ever places/edits blocks from a fixed, small palette
(`genetic_ml/blocks.py`) matching the block types the C++ simulator gives special
physics to: piston, sticky piston, redstone block, slime, observer, redstone lamp,
obsidian, fence gate, detector rail, plus air (used to remove blocks). Anything
outside this palette is treated as an inert solid by the simulator, so mutating
toward it would almost never change behavior - keeping mutations inside the palette
keeps the search productive. Piston head/extension and lit redstone lamp are
excluded because those states are simulator-managed outputs, not valid placements.

The block at `trigger` is never removed, moved, or retyped by mutation (only its
facing may change), since removing or retyping it would sever how the machine gets
powered on.

Every distinct discovery (by a position/rotation-independent structural hash) is
recorded exactly once in `data/outputs/ga_archive.jsonl` regardless of whether it
survives in the live population - that's the full audit trail - and, if genuinely
new, saved as its own file in `data/working/` so it's immediately reusable as a
seed.

### Crash and hang logging

Mutation deliberately wanders into structures the (still work-in-progress) C++
simulator doesn't handle correctly yet - some crash the process outright, others
hang it in an unbounded update cascade. `SimulatorPool` recovers from both and
keeps the run going (this is separate from the PATH/DLL environment quirk below,
which would otherwise fail *every* candidate regardless of content), but each
offending candidate is also written out to `data/crash/` or `data/hangs/` for later
use as a simulator regression/repro case. Unlike `data/working/`, these files are
*only* the candidate JSON itself - nothing else - since the point is to be able to
feed one straight back into the simulator as-is.

### Troubleshooting: every candidate reports working=False or crashes

`cpp_simulator_stream.exe` is dynamically linked against the msys64/ucrt64 MinGW
runtime (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, same toolchain `build-cpp.bat`
builds with) and does not bundle those DLLs next to the executable. If whatever
shell launches Python has a *different* MinGW runtime earlier on `PATH` - Git for
Windows bundles its own at `...\Git\mingw64\bin`, for example - Windows loads the
wrong DLL and the process exits immediately on every single candidate with no
stderr output, which looks like every machine failing rather than an environment
problem.

`SimulatorProcess.start()` (`genetic_ml/simulator_process.py`) works around this by
prepending `C:\msys64\ucrt64\bin` to the subprocess's `PATH` itself whenever that
directory exists, so it doesn't matter which shell VS Code, Code Runner, or a
terminal launches Python from. If the toolchain lives somewhere else on your
machine, or you still see every candidate failing, update `_MINGW_RUNTIME_DIR` in
that file to match.

