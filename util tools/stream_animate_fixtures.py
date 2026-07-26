"""Standalone diagnostic: simulate every JSON fixture in `flying machines/animate/` and stream
them to the flyer-web-visualizer's /generator page, to verify the page's piston-push animation is
actually working - using known-good, hand-built fixtures instead of generator output, so a
rendering bug in the page isn't confused with a bug in what the generator produces.

Reuses stream_to_visualizer.py's exact wire format and directory-serving loop
(build_animation_record via handle_client) - fixtures are just simulated once into a scratch
shard directory, then served exactly the way an already-built corpus directory would be, so this
is testing the real client-side animation path, not a special case of it.

Run (from anywhere - paths below are resolved relative to this file, not the CWD):
    py "util tools/stream_animate_fixtures.py"
Then open the visualizer's /generator page (defaults to ws://localhost:8766) and Connect.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# This file lives in `util tools/`, a sibling of `transformer gym/` (which is where the
# `generator` package actually lives) - not a descendant of it, so the path inserted here must
# name `transformer gym/` explicitly rather than just walking up parent directories.
_TRANSFORMER_GYM = Path(__file__).resolve().parent.parent / "transformer gym"
sys.path.insert(0, str(_TRANSFORMER_GYM))  # so `import generator` resolves
import generator  # noqa: F401,E402

from genetic_ml.compact_format import json_file_to_compact  # noqa: E402
from stream_to_visualizer import handle_client  # noqa: E402
from verify_simulation_data import EXE, MSYS_BIN  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "flying machines" / "animate"

HOST = "localhost"
PORT = 8766  # same default the /generator page's URL field already shows - Connect just works
POLL_INTERVAL = 1.0
BATCH = 20


def _load_id(json_path: Path) -> int:
    return int(json.loads(json_path.read_text(encoding="utf-8").splitlines()[0])["id"])


def simulate_fixture(json_path: Path, workdir: Path) -> Path:
    """Mirrors verify_simulation_data.run_fixture, but for an arbitrary json_path instead of a
    name looked up in that function's hardcoded FIXTURE_DIR (these fixtures live in a different
    folder: flying machines/animate/, not flying machines/json/)."""
    dat = workdir / f"{json_path.stem}.dat"
    json_file_to_compact(json_path, dat)

    base = workdir / f"{json_path.stem}.simlog"
    env = os.environ.copy()
    env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
    env["MCP1122_CPP_NO_Y_OFFSET"] = "1"
    subprocess.run(
        [str(EXE), str(dat), "--simulation-data", str(base)],
        env=env, check=True, stdout=subprocess.DEVNULL,
    )

    cid = _load_id(json_path)
    out = workdir / f"{json_path.stem}-{cid}.simlog"
    if not out.exists():
        raise FileNotFoundError(f"expected log not produced: {out}")
    return out


def build_shard_dir() -> Path:
    """Simulates every fixture in FIXTURE_DIR into a scratch shard directory, one {stem}.simlog
    per fixture - handle_client/build_animation_record only care about the .simlog extension; the
    fixture's own name becomes the wire record's "name" field via path.stem.

    Raw simulation intermediates (the exe's own {stem}-{id}.dat/.simlog naming) are written into a
    nested _raw/ subdirectory, never straight into shard_dir - handle_client's glob("*.simlog") is
    non-recursive, so shard_dir only ever sees the one curated {stem}.simlog copy per fixture, not
    both names for the same machine."""
    fixtures = sorted(FIXTURE_DIR.glob("*.json"))
    if not fixtures:
        raise SystemExit(f"no .json fixtures found in {FIXTURE_DIR}")

    shard_dir = Path(tempfile.mkdtemp(prefix="animate_fixtures_"))
    atexit.register(shutil.rmtree, shard_dir, ignore_errors=True)
    raw_dir = shard_dir / "_raw"
    raw_dir.mkdir()

    for json_path in fixtures:
        print(f"simulating {json_path.name} ...", flush=True)
        try:
            out = simulate_fixture(json_path, raw_dir)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}", flush=True)
            continue
        shard_path = shard_dir / f"{json_path.stem}.simlog"
        shard_path.write_bytes(out.read_bytes())
        print(f"  -> {shard_path.name}", flush=True)
    return shard_dir


async def main() -> None:
    shard_dir = build_shard_dir()
    print(flush=True)

    async with serve(
        lambda ws: handle_client(ws, shard_dir, POLL_INTERVAL, BATCH),
        HOST,
        PORT,
        max_size=None,
    ):
        print(f"streaming {shard_dir} on ws://{HOST}:{PORT}  (Ctrl-C to stop)", flush=True)
        print("open the visualizer's /generator page and Connect (defaults to this URL)", flush=True)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
