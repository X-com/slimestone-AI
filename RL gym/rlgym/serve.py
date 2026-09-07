"""Training with a dashboard - one process, one double-click.

The loop already prints a round row every few minutes and writes everything else to
`metrics.jsonl` and `library/`. That is the right design for a run you come back to, and the
wrong one for a run you are watching: a round takes minutes, so the console looks frozen, and
the discoveries themselves - the actual product - are only visible by opening a JSON file.

This adds two things and changes nothing about the training:

    a dashboard    every discovered machine as it is admitted, downloadable as fixture JSON
    a heartbeat    a console line every few seconds, so a live run is distinguishable from a
                   hung one WITHOUT waiting a whole round to find out

**Stdlib only.** `pyproject.toml` declares no dependencies and this does not change that -
`http.server` is enough for one viewer on localhost, and a web framework here would be a
dependency the project carries forever for a page with one table on it.

**The training thread owns the data; the server only reads.** The loop runs in the main thread
so Ctrl-C stops it the way it always did, and the HTTP server runs as a daemon thread that reads
`Loop.store` and `Loop.metrics` live. Nothing is copied into a second place that could disagree
with the first.

Usage:
    py -m rlgym.serve --config configs/stage1.json --checkpoint data/runs/stage0/best.pt
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rlgym.config import Config
from rlgym.loop import Loop
from rlgym.train import load_checkpoint

DEFAULT_PORT = 8765
# Newest N machines sent to the page. The library is unbounded and the page polls every two
# seconds, so without a cap a long run spends real CPU serialising thousands of entries into
# a table nobody can read - measured at 66 seconds for one test that let the library run away.
# The full library is always on disk in `library/library.json`; this is a viewport, not a store.
LIBRARY_LIMIT = 200


# --- reading the live loop ------------------------------------------------------------------


def _snapshot(mapping: dict) -> list:
    """A stable list of a dict the training thread is still writing to.

    CPython raises RuntimeError if a dict changes size while it is being iterated, and the
    training thread admits machines whenever it likes. Retrying is enough - the window is a
    few microseconds wide and the next attempt lands outside it - and it keeps the dashboard
    from being a source of 500s in a run that is otherwise fine.
    """
    for _ in range(5):
        try:
            return list(mapping.values())
        except RuntimeError:
            continue
    return []


def state_of(loop: Loop, started: float, finished: threading.Event) -> dict[str, Any]:
    """Everything the page shows, read straight off the objects the loop is writing."""
    entries = _snapshot(loop.store.entries)
    # `config` is repeated verbatim in every metrics row on purpose (metrics.py), which is
    # right for a file that has to be answerable years later and pointless over the wire.
    rows = [
        {key: value for key, value in row.items() if key not in ("config", "params")}
        for row in loop.metrics.rows
        if row.get("kind") == "round"
    ]
    # Newest first: the discoveries are the point, and the seeds are always there.
    newest = sorted(entries, key=lambda e: (e.round_found, e.generation), reverse=True)
    return {
        "running": not finished.is_set(),
        "elapsed": round(time.perf_counter() - started, 1),
        "round": rows[-1]["round_index"] if rows else 0,
        "rounds": loop.config.loop.rounds,
        "replay": len(loop.replay.examples),
        "attempts": len(loop.store.seen),
        "library_total": len(entries),
        "library": [
            {
                "digest": entry.digest,
                "name": entry.name,
                "generation": entry.generation,
                "blocks": entry.blocks,
                "period": entry.period,
                "shift": list(entry.shift),
                "round_found": entry.round_found,
                "descendants": entry.descendants,
                "redundant_removed": entry.redundant_removed,
            }
            for entry in newest[:LIBRARY_LIMIT]
        ],
        "rows": rows,
    }


def _handler(loop: Loop, started: float, finished: threading.Event):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, kind: str, filename: str | None = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            if filename:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                payload = json.dumps(state_of(loop, started, finished)).encode("utf-8")
                return self._send(payload, "application/json")
            if path.startswith("/api/machine/"):
                digest = path.rsplit("/", 1)[-1]
                entry = loop.store.entries.get(digest)
                if entry is None:
                    self.send_error(404, "no machine with that digest")
                    return
                # The candidate as the fixture loader and the visualiser already read it, so a
                # discovery can be opened in the existing tooling instead of only looked at.
                payload = json.dumps(entry.candidate, indent=1).encode("utf-8")
                return self._send(
                    payload, "application/json", filename=f"{entry.name}.json"
                )
            self.send_error(404)

        def log_message(self, *args) -> None:
            """Silence per-request logging. The console belongs to the training run, and a
            dashboard polling once a second would bury the round rows completely."""

    return Handler


def serve(loop: Loop, port: int, started: float, finished: threading.Event):
    """Start the dashboard on a daemon thread and return the server."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(loop, started, finished))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --- the console heartbeat --------------------------------------------------------------


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
        print(
            f"  ...{round(time.perf_counter() - started):>5}s   "
            f"attempts {now[0]:>6}   library {now[1]:>4}   replay {len(loop.replay.examples):>6}",
            flush=True,
        )


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>slimestone training</title>
<style>
  :root { color-scheme: dark; --bg:#12141a; --panel:#1a1d26; --line:#2a2f3d; --dim:#8b93a7;
          --ink:#e6e9f0; --good:#7ee787; --warm:#f0b849; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px ui-monospace, "Cascadia Code", Consolas, monospace; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:20px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; letter-spacing:.14em; text-transform:uppercase; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:7px; }
  .live { background:var(--good); box-shadow:0 0 9px var(--good); }
  .done { background:var(--dim); }
  main { padding:24px; display:grid; gap:24px; grid-template-columns:1fr; max-width:1200px; }
  .cards { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
  .card b { display:block; font-size:26px; font-weight:600; margin-bottom:3px; }
  .card span { color:var(--dim); font-size:11px; letter-spacing:.1em; text-transform:uppercase; }
  h2 { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim);
       margin:0 0 10px; }
  .scroll { overflow-x:auto; background:var(--panel); border:1px solid var(--line);
            border-radius:8px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th { text-align:left; color:var(--dim); font-weight:500; font-size:11px;
       letter-spacing:.09em; text-transform:uppercase; }
  th, td { padding:8px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }
  tr:last-child td { border-bottom:0; }
  td.n { text-align:right; font-variant-numeric:tabular-nums; }
  a { color:var(--warm); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .empty { padding:26px 14px; color:var(--dim); }
  .seed { color:var(--dim); }
</style>
<header>
  <h1>slimestone training</h1>
  <div><span id="dot" class="dot done"></span><span id="status">connecting</span></div>
</header>
<main>
  <div class="cards" id="cards"></div>
  <section>
    <h2>Discovered machines <span id="shown"></span></h2>
    <div class="scroll"><table>
      <thead><tr>
        <th>name</th><th>gen</th><th class="n">blocks</th><th class="n">period</th>
        <th>shift</th><th class="n">stripped</th><th class="n">round</th>
        <th class="n">children</th><th></th>
      </tr></thead>
      <tbody id="library"><tr><td colspan="9" class="empty">waiting for the first round</td></tr></tbody>
    </table></div>
  </section>
  <section>
    <h2>Rounds</h2>
    <div class="scroll"><table>
      <thead><tr>
        <th class="n">round</th><th class="n">seconds</th><th class="n">model /1k</th>
        <th class="n">control /1k</th><th class="n">stripped</th><th class="n">stopped</th>
        <th class="n">depth</th><th class="n">library</th><th class="n">replay</th>
      </tr></thead>
      <tbody id="rounds"><tr><td colspan="9" class="empty">no round has finished yet</td></tr></tbody>
    </table></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const cell = (v, cls) => `<td class="${cls || ''}">${v}</td>`;

function cards(s) {
  const items = [
    ['round', `${s.round} / ${s.rounds}`], ['library', s.library_total],
    ['attempts', s.attempts], ['replay', s.replay],
    ['elapsed', `${Math.floor(s.elapsed / 60)}m ${Math.round(s.elapsed % 60)}s`],
  ];
  $('cards').innerHTML = items.map(([k, v]) =>
    `<div class="card"><b>${v}</b><span>${k}</span></div>`).join('');
}

function library(rows, total) {
  $('shown').textContent = total > rows.length ? `- newest ${rows.length} of ${total}` : '';
  if (!rows.length) return;
  $('library').innerHTML = rows.map(m => `<tr class="${m.generation ? '' : 'seed'}">` +
    cell(m.name) + cell(m.generation, 'n') + cell(m.blocks, 'n') + cell(m.period, 'n') +
    cell(m.shift.join(',')) + cell(m.redundant_removed, 'n') + cell(m.round_found, 'n') +
    cell(m.descendants, 'n') +
    cell(`<a href="/api/machine/${m.digest}">json</a>`) + '</tr>').join('');
}

function rounds(rows) {
  if (!rows.length) return;
  $('rounds').innerHTML = rows.slice().reverse().map(r => {
    const h = r.headline || {};
    return '<tr>' + cell(r.round_index, 'n') + cell(r.seconds, 'n') +
      cell((h.model ?? 0).toFixed(2), 'n') + cell((h.uninformed ?? 0).toFixed(2), 'n') +
      cell(h.redundant_stripped ?? 0, 'n') + cell(h.stopped ?? 0, 'n') +
      cell(h.mean_depth ?? '-', 'n') + cell((r.library || {}).size ?? 0, 'n') +
      cell(r.replay ?? 0, 'n') + '</tr>';
  }).join('');
}

async function tick() {
  try {
    const s = await (await fetch('/api/state')).json();
    $('dot').className = 'dot ' + (s.running ? 'live' : 'done');
    $('status').textContent = s.running ? 'training' : 'finished';
    cards(s); library(s.library, s.library_total); rounds(s.rows);
  } catch (e) {
    $('dot').className = 'dot done';
    $('status').textContent = 'server closed';
  }
}
tick();
setInterval(tick, 2000);
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.json"))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--machines", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/runs/live"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--heartbeat", type=float, default=15.0)
    parser.add_argument("--no-browser", action="store_true")
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
    server = serve(loop, args.port, started, finished)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"\n  dashboard   {url}")
    print(f"  run         {args.out}")
    print(
        f"  plan        {config.loop.rounds} rounds x {config.loop.episodes_per_round} "
        f"episodes, k={config.search.k}, {config.search.simulations} simulator calls each"
    )
    print("  stop        Ctrl-C\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

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

    print(f"\ntraining finished. dashboard still at {url} - Ctrl-C to quit.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
