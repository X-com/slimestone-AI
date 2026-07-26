"""Streams .simlog shard files to the flyer-web-visualizer's /generator page over a plain
websocket, so the generator package (now proven correct by tests/test_generators.py) can also be
watched, block-by-block, with its actual piston-push animation - the visual verification the
generator's own pytest suite can't give you.

Two modes, chosen by the GENERATOR_INDEX/COUNT/SEED fields below (edit the file, not a CLI flag -
this is meant to be a "set it and run it" script, not a many-flag CLI):
  GENERATOR_INDEX = <N>   Generates COUNT candidates from ONE generator, selected by number N (see
                          GENERATORS below - running the script prints the numbered list at
                          startup), via corpus.iter_corpus (the same generate/simulate/dedup/
                          dead-filter loop build_corpus/generate.py already use) in a background
                          thread. What happens to each candidate is controlled by SAVE_TO_DISK
                          below: written to --dir and kept (True), or generated, streamed, and
                          discarded without ever being left on disk (False, the default - see
                          SAVE_TO_DISK's own comment).
  GENERATOR_INDEX = None  Just watches --dir - the same directory transformer_gym.dataset.
                          SimlogDirDataset trains from, written by corpus.build_corpus/
                          generate.py - and streams whatever .simlog files are already there, then
                          keeps polling for new ones. No generation happens in this mode; it's for
                          verifying a corpus someone already built for training.

Either way, every record sent is decoded from the literal .simlog bytes via the same
transformer_gym.simlog_reader helpers transformer_gym.encode.encode() uses to build training
targets - no re-simulation, so what you see animated is provably what the transformer trained on.

Run (from anywhere - paths below are resolved relative to this file, not the CWD):
    py "util tools/stream_to_visualizer.py"

ponytail: plain ws:// only (this session's decision - simplest to stand up right now). Add wss://
by mirroring flyer-web-visualizer/scripts/dev_tls.py's build_ssl_context() the same way
stream_flyers.py does, if/when the training-stream page's TLS setup is actually needed here too.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path

# This file lives in `util tools/`, a sibling of `transformer gym/` (which is where the
# `generator` package actually lives) - not a descendant of it, so the path inserted here must
# name `transformer gym/` explicitly rather than just walking up parent directories.
_TRANSFORMER_GYM = Path(__file__).resolve().parent.parent / "transformer gym"
sys.path.insert(0, str(_TRANSFORMER_GYM))  # so `import generator` resolves
import generator  # noqa: F401,E402

from corpus import GENERATOR_WEIGHTS, iter_corpus  # noqa: E402
from genetic_ml.blocks import BLOCK_OBSERVER, BLOCK_PISTON, BLOCK_STICKY_PISTON, make_state  # noqa: E402
from transformer_gym.simlog_reader import (  # noqa: E402
    KIND_NAMES,
    iter_block_events,
    read_block_index,
    read_footer,
    read_initial_state,
)
from wave_function import gen_wfc  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

_BLOCK_PUSHED_KIND = next(k for k, name in KIND_NAMES.items() if name == "BlockPushed")
_PISTON_MOVE_KIND = next(k for k, name in KIND_NAMES.items() if name == "PistonMoveExecuted")
_OBSERVER_FIRED_KIND = next(k for k, name in KIND_NAMES.items() if name == "ObserverFired")
_PISTON_IDS = {BLOCK_PISTON, BLOCK_STICKY_PISTON}
# sim_event_log.h's SEF_EXTEND/SEF_SUCCESS flag bits on a PistonMoveExecuted event (not
# re-exported by transformer_gym.simlog_reader - only verify_simulation_data.py defines them).
_SEF_EXTEND = 1 << 0
_SEF_SUCCESS = 1 << 1

# Numbered so GENERATOR_INDEX below can select one by a plain int, per the user's request.
# Order/names match generator/tests/test_generators.py's GENERATORS tuple (perturb/forward/puzzle/
# pushgroup/splice/random from corpus.GENERATOR_WEIGHTS, plus wave_function - excluded from that
# mix by an earlier session's "standalone" decision, but selectable here for preview).
GENERATORS: tuple[tuple[str, object], ...] = tuple((n, g) for n, g, _ in GENERATOR_WEIGHTS) + (
    ("wave_function", gen_wfc.generate),
)

# ---- Edit these to choose what to generate, then just run the script ---------------------------
# None = generate nothing, only watch --dir (e.g. to preview a corpus already built for training).
# Otherwise an index into GENERATORS above (printed at startup) - e.g. 6 = wave_function.

GENERATOR_INDEX = 6
# GENERATOR_INDEX: int | None = None
COUNT: int = 200
SEED: int = 0

# False (default): generated candidates are simulated, streamed, and discarded - never written
# anywhere under the project. This is a preview tool, not a corpus-building one; leaving it
# writing hundreds of .simlog files into generator/generated/ every run just clutters the repo
# with throwaway data nobody trains on. iter_corpus() still needs to write each candidate ONCE to
# a real path internally (encode()'s too-big check reads from a path, not bytes) - that happens in
# a system temp directory and is deleted immediately after this script reads it into memory, so
# nothing persists in the project either way.
# True: keeps the old behavior - candidates are written to --dir and kept, exactly like
# corpus.build_corpus/generate.py's real training-corpus runs. Turn this on if you actually want
# to build/extend a corpus while previewing it, not just look at it.
SAVE_TO_DISK: bool = False
# --------------------------------------------------------------------------------------------


def _print_generators() -> None:
    print("GENERATORS (set GENERATOR_INDEX at the top of this file to one of these):")
    for i, (name, _) in enumerate(GENERATORS):
        print(f"  {i}  {name}")


def build_animation_record_from_bytes(data: bytes) -> dict:
    """Projects raw .simlog bytes - the literal bytes transformer_gym.encode.encode() also reads
    for training - into the JSON wire shape /generator animates. Every keyframe below is a real
    SimEvent's own executedTick/toX,toY,toZ (or flags, for piston extension), read via the same
    transformer_gym.simlog_reader helpers encode.py uses to build training targets: no
    re-simulation, no synthesized motion. Split from build_animation_record() below so
    SAVE_TO_DISK=False's in-memory path never needs a persisted file just to decode one."""
    footer = read_footer(data)
    initial = read_initial_state(data, footer)
    index = read_block_index(data, footer)
    key_to_idx = {s.stableKey: i for i, s in enumerate(initial)}
    entry_by_key = {e.originalKey: e for e in index}

    blocks = [
        {"x": s.x, "y": s.y, "z": s.z, "state": make_state(s.blockTypeId, s.facing if 0 <= s.facing <= 5 else 0)}
        for s in initial
    ]
    trigger_block = next((s for s in initial if s.isTrigger), None)
    trigger = (
        {"x": trigger_block.x, "y": trigger_block.y, "z": trigger_block.z}
        if trigger_block is not None
        else {"x": 0, "y": 0, "z": 0}
    )

    # events: the literal, itemized log the /generator page's tick/subtick stepper walks through -
    # one entry per real SimEvent that has a visible effect, in true recorded order (tick, order),
    # not a wall-clock projection. Built from the same per-entry iter_block_events() calls the
    # moves/extensions loops below already need, so nothing is scanned twice.
    events = []

    moves = []
    termination_tick = 0
    for entry in index:
        i = key_to_idx.get(entry.originalKey)
        if i is None:
            continue
        steps = []
        for ev in iter_block_events(data, entry):
            if ev.kind != _BLOCK_PUSHED_KIND:
                continue
            steps.append({"tick": ev.executedTick, "order": ev.executedSubtick, "x": ev.toX, "y": ev.toY, "z": ev.toZ})
            termination_tick = max(termination_tick, ev.executedTick)
            events.append({"tick": ev.executedTick, "order": ev.executedSubtick, "kind": "blockPushed", "blockIndex": i})
        if steps:
            moves.append({"blockIndex": i, "steps": steps})

    # Piston head extension: never modeled before this - the head isn't a separate block in the
    # initial state (it only comes into existence on extension), so it can't be tracked as a
    # "moved" block the way BlockPushed entries are. Instead every piston/sticky_piston gets its
    # own extend/retract timeline, starting from its real t=0 state (stateFlags bit 0, the same
    # bit encode.py's `flags[i, 0]` reads) and updated by its own PistonMoveExecuted events
    # (kind=1; SEF_EXTEND/SEF_SUCCESS flag bits). Failed (blocked) attempts don't move anything
    # visually, but they DO get their own "pistonBlocked" event below - that's the whole point of
    # the stepper (see a piston try and fail to push, not just silently do nothing).
    extensions = []
    for i, s in enumerate(initial):
        if s.blockTypeId not in _PISTON_IDS:
            continue
        ext_steps = [{"tick": 0, "order": 0, "extended": bool(s.stateFlags & 1)}]
        entry = entry_by_key.get(s.stableKey)
        if entry is not None:
            for ev in iter_block_events(data, entry):
                if ev.kind != _PISTON_MOVE_KIND:
                    continue
                if ev.flags & _SEF_SUCCESS:
                    ext_steps.append({"tick": ev.executedTick, "order": ev.executedSubtick, "extended": bool(ev.flags & _SEF_EXTEND)})
                    kind = "pistonExtend" if (ev.flags & _SEF_EXTEND) else "pistonRetract"
                else:
                    kind = "pistonBlocked"
                termination_tick = max(termination_tick, ev.executedTick)
                events.append({"tick": ev.executedTick, "order": ev.executedSubtick, "kind": kind, "blockIndex": i})
        extensions.append({"blockIndex": i, "steps": ext_steps})

    # Observer fires: never read before this - ObserverFired (kind 3) carries blockKey/actorKey =
    # the observer's own key, so it maps to a blockIndex exactly like everything else above.
    for i, s in enumerate(initial):
        if s.blockTypeId != BLOCK_OBSERVER:
            continue
        entry = entry_by_key.get(s.stableKey)
        if entry is None:
            continue
        for ev in iter_block_events(data, entry):
            if ev.kind != _OBSERVER_FIRED_KIND:
                continue
            termination_tick = max(termination_tick, ev.executedTick)
            events.append({"tick": ev.executedTick, "order": ev.executedSubtick, "kind": "observerFired", "blockIndex": i})

    events.sort(key=lambda e: (e["tick"], e["order"]))

    return {
        "trigger": trigger, "blocks": blocks, "moves": moves, "extensions": extensions,
        "events": events, "terminationTick": termination_tick,
    }


def build_animation_record(shard_path: Path) -> dict:
    """Reads one .simlog shard file straight off disk and decodes it - the --dir watch-only path
    (and SAVE_TO_DISK=True generation) both go through here, since their shards are real files."""
    return build_animation_record_from_bytes(shard_path.read_bytes())


# SAVE_TO_DISK=False's generated records land here instead of a shard directory - handle_client
# drains this the same way it drains a directory glob (see _send_memory_records), so a client
# can't tell which mode is active except that no files ever appear on disk. Appended to from the
# background generation thread, read from the event loop thread; a plain list append is atomic
# under the GIL, but the lock makes the snapshot-then-iterate in _send_memory_records safe too.
_memory_records: list[tuple[str, dict]] = []
_memory_lock = threading.Lock()


async def _send_existing(ws, shard_dir: Path, seen: set[str], batch_size: int) -> None:
    pending: list[str] = []
    for path in sorted(shard_dir.glob("*.simlog")):
        if path.name in seen:
            continue
        seen.add(path.name)
        record = build_animation_record(path)
        record["name"] = path.stem
        pending.append(json.dumps(record))
        if len(pending) >= batch_size:
            await ws.send("\n".join(pending))
            pending = []
    if pending:
        await ws.send("\n".join(pending))


async def _send_memory_records(ws, seen: set[str], batch_size: int) -> None:
    with _memory_lock:
        snapshot = list(_memory_records)
    pending: list[str] = []
    for name, record in snapshot:
        if name in seen:
            continue
        seen.add(name)
        pending.append(json.dumps(record))
        if len(pending) >= batch_size:
            await ws.send("\n".join(pending))
            pending = []
    if pending:
        await ws.send("\n".join(pending))


async def handle_client(ws, shard_dir: Path, poll_interval: float, batch_size: int) -> None:
    peer = getattr(ws, "remote_address", "?")
    print(f"[+] client {peer} connected", flush=True)
    seen: set[str] = set()
    try:
        while True:
            await _send_existing(ws, shard_dir, seen, batch_size)
            await _send_memory_records(ws, seen, batch_size)
            await asyncio.sleep(poll_interval)
    except ConnectionClosed:
        pass
    finally:
        print(f"[-] client {peer} disconnected", flush=True)


def _run_generation(shard_dir: Path, generator_index: int, count: int, seed: int, save_to_disk: bool) -> None:
    """Blocking (real subprocess calls per candidate) - always run via run_in_executor, never
    awaited directly on the server's event loop. Prints as it goes (flush=True - stdout is fully
    buffered, not line-buffered, when it isn't a real terminal, e.g. redirected to a file/pipe, so
    unflushed prints can sit invisible for the whole run otherwise)."""
    name, generate = GENERATORS[generator_index]
    written = 0
    if save_to_disk:
        print(f"[generate] {count} candidates from '{name}' (seed={seed}) -> {shard_dir}", flush=True)
        for shard_name, shard_path in iter_corpus(shard_dir, count=count, seed=seed, generators=((name, generate, 1.0),)):
            written += 1
            print(f"    wrote {shard_path.name} ({shard_name}) [{written}/{count}]", flush=True)
    else:
        print(f"[generate] {count} candidates from '{name}' (seed={seed}) -> memory only, SAVE_TO_DISK=False", flush=True)
        # iter_corpus still writes each accepted candidate once (encode()'s too-big check reads a
        # real path, not bytes) - into a system temp dir, deleted immediately per-file and gone
        # entirely once this block exits, so nothing ever persists under the project.
        with tempfile.TemporaryDirectory(prefix="stream_to_visualizer_") as tmp:
            for shard_name, shard_path in iter_corpus(Path(tmp), count=count, seed=seed, generators=((name, generate, 1.0),)):
                data = shard_path.read_bytes()
                shard_path.unlink()
                record = build_animation_record_from_bytes(data)
                record["name"] = shard_path.stem
                with _memory_lock:
                    _memory_records.append((record["name"], record))
                written += 1
                print(f"    generated {shard_path.stem} ({shard_name}) [{written}/{count}] (not saved)", flush=True)
    print(f"[generate] done: {written}/{count} written", flush=True)


async def main() -> None:
    _print_generators()
    print()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dir", type=Path, default=_TRANSFORMER_GYM / "generator" / "generated",
        help="corpus shard directory to watch/stream (and, if SAVE_TO_DISK=True, to write into)",
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--poll-interval", type=float, default=1.0, help="seconds between directory scans")
    ap.add_argument("--batch", type=int, default=20, help="records per websocket frame")
    args = ap.parse_args()

    if GENERATOR_INDEX is not None and not (0 <= GENERATOR_INDEX < len(GENERATORS)):
        raise SystemExit(
            f"GENERATOR_INDEX={GENERATOR_INDEX} is out of range (0-{len(GENERATORS) - 1}) - "
            "edit it at the top of this file"
        )

    args.dir.mkdir(parents=True, exist_ok=True)

    if GENERATOR_INDEX is not None:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _run_generation, args.dir, GENERATOR_INDEX, COUNT, SEED, SAVE_TO_DISK)

        def _report_if_failed(f: asyncio.Future) -> None:
            exc = f.exception()
            if exc is not None:
                print("[generate] FAILED:", flush=True)
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        future.add_done_callback(_report_if_failed)

    async with serve(
        lambda ws: handle_client(ws, args.dir, args.poll_interval, args.batch),
        args.host,
        args.port,
        max_size=None,
    ):
        print(f"streaming {args.dir} on ws://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
