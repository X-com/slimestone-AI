"""Training, streamed to the flyer-web-visualizer - one process, one double-click.

The loop prints a round row every few minutes and writes everything else to `metrics.jsonl` and
`library/`. That is right for a run you come back to and wrong for a run you are watching: a
round takes minutes, so the console looks frozen, and the discoveries themselves - the actual
product - are only visible by opening a JSON file.

This adds two things and changes nothing about the training:

    the stream    every discovered machine to the visualizer's Live Training page, in 3D
    a heartbeat   a console line every few seconds, so a live run is distinguishable from a
                  hung one WITHOUT waiting a whole round to find out

**The viewer is `flyer-web-visualizer`, not a page written here.** An earlier version of this
file served its own HTML table, which was a worse answer to a question already answered well:
that project renders machines in 3D with orbit and zoom, split latest-batch and history
viewports, paging, click-to-select and auto-reconnect. Two dashboards is two things to maintain
and one of them is always behind. See `rlgym/stream.py` for the protocol.

**The training thread owns the data.** The loop runs in the main thread so Ctrl-C stops it the
way it always did; the hub serves on a daemon thread and only ever reads what the loop hands it.

Usage:
    py -m rlgym.serve --config configs/stage1.json --checkpoint data/runs/stage0/best.pt
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from rlgym.animation import animate
from rlgym.config import Config
from rlgym.loop import Loop
from rlgym.stream import DEFAULT_BACKLOG, DEFAULT_PORT, StreamHub
from rlgym.train import load_checkpoint


def heartbeat(loop: Loop, started: float, finished: threading.Event, every: float) -> None:
    """One line every `every` seconds, so a live run looks different from a hung one.

    A round is minutes long and prints only at its end, so without this the console is silent
    for long enough to look like a crash - which is exactly when someone kills a healthy run.
    Printed only when something actually moved, so a genuinely stuck loop goes quiet and that
    silence means something.
    """
    last: tuple[int, int] = (-1, -1)
    while not finished.wait(every):
        now = (len(loop.store.seen), len(loop.store.entries))
        if now == last:
            continue
        last = now
        watchers = loop.hub.clients if loop.hub is not None else 0
        print(
            f"  ...{round(time.perf_counter() - started):>5}s   "
            f"attempts {now[0]:>6}   library {now[1]:>4}   "
            f"replay {len(loop.replay.examples):>6}"
            + (f"   viewers {watchers}" if watchers else ""),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.json"))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--machines", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/runs/live"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--backlog", type=int, default=DEFAULT_BACKLOG)
    parser.add_argument("--heartbeat", type=float, default=15.0)
    parser.add_argument("--no-stream", action="store_true", help="train without the viewer")
    args = parser.parse_args()

    config = Config.load(args.config if args.config.exists() else None)
    if args.rounds is not None:
        config.loop.rounds = args.rounds

    net = None
    if args.checkpoint is not None and args.checkpoint.exists():
        net, payload = load_checkpoint(args.checkpoint)
        print(f"loaded {args.checkpoint} (step {payload['step']})")
    elif args.checkpoint is not None:
        # Loud, because the alternative is a run that silently starts from noise and takes an
        # hour to look wrong.
        print(f"no checkpoint at {args.checkpoint} - starting from an untrained network")

    loop = Loop(config, args.out, net=net)
    started = time.perf_counter()
    finished = threading.Event()

    hub = None
    if not args.no_stream:
        # `animate` is what makes the viewer's Simulate button work: it re-simulates one
        # published machine with logging on and hands back the per-tick record. Nothing is
        # computed until somebody asks, so a run nobody watches costs exactly what it did before.
        hub = StreamHub(
            host=args.host, port=args.port, backlog=args.backlog, animate=animate
        )
        if hub.start():
            loop.hub = hub
        else:
            # A failed bind is almost always a second run, or a stale process, holding the
            # port. Training is still worth doing, so this warns and continues rather than
            # exiting - but it must not be silent, or the viewer just never connects and the
            # reason is invisible.
            print(f"  WARNING  could not open ws://{args.host}:{args.port}: {hub.error}")
            print("           training continues; the visualizer will receive nothing.")
            hub = None

    print(f"\n  run         {args.out}")
    if hub is not None:
        print(f"  stream      {hub.url}")
        print("  viewer      the Live Training page connects to this on its own")
        print("  simulate    click a machine, then Simulate, to watch it run")
    print(
        f"  plan        {config.loop.rounds} rounds x {config.loop.episodes_per_round} "
        f"episodes, k={config.search.k}, {config.search.simulations} simulator calls each"
    )
    print("  stop        Ctrl-C\n", flush=True)

    threading.Thread(
        target=heartbeat, args=(loop, started, finished, args.heartbeat), daemon=True
    ).start()

    try:
        loop.run(args.machines)
    except KeyboardInterrupt:
        # A stopped run is not a failed one: every round already saved its library, its
        # metrics row and its checkpoint, so what was found is on disk either way.
        print("\ninterrupted - the library and metrics up to the last round are saved")
    finally:
        finished.set()

    if hub is not None:
        # Hold the socket open so a viewer can still connect and browse the run that just
        # ended - the backlog it would replay is already in memory.
        print(f"\ntraining finished. {hub.url} still serving - Ctrl-C to quit.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        hub.stop()
    print("\ndone.")


if __name__ == "__main__":
    main()
