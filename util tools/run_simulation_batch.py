"""WebSocket server that runs EVERY schematic in "flying machines/json" through the C++
simulator with --simulation-data and streams all of them (no validCycle filtering - every
fixture that loads is shown, working or not) to the flyer-web-visualizer's Simulated tab.
Nothing is written to the visualizer's public/ folder; everything goes over the socket.

Why genetic_ml.compact_format is imported here: it is not GA-specific logic, it is the one
shared binary-encoding function this script has no reason to reimplement - encode_candidate
turns a candidate dict into both the exe's compact .dat input AND the same wire-format bytes
the visualizer already decodes client-side (parseCompactData in src/lib/data.ts), so the format
here is identical to every other producer in this repo (main_ga.py's StreamHub,
scripts/stream_flyers.py) instead of a second implementation to keep in sync by hand.

Each fixture's own "id" field (assigned by util tools/schematic_to_stream_json.py, restarting
from 1 per export batch) is NOT unique across the whole flying machines/json folder - e.g.
simple_observer_engine.json and test_flyer_23.json are both id 15. The exe names each
candidate's simulation_data file after that id (batch-15.simlog), so two fixtures sharing an id
would silently overwrite each other's log. This script assigns every fixture a fresh, guaranteed
-unique id for the run (its 1-based position in the sorted fixture list) before encoding, so the
id is only ever used as this run's internal handle - display names still come from the filename.

Wire format, sent on every connection:
    1. one text (JSON) message: {"names": {"<id>": "<fixture filename stem>", ...}} - the compact
       binary candidate format carries no name field, so labels are sent out-of-band like this.
    2. one binary message, tag 0x01 + concatenated compact-format candidate records (all fixtures)
    3. one or more binary messages, each tag 0x02 + repeated {int32 id, uint32 byteLength,
       <byteLength bytes of simulation_data>} - fixtures' logs chunked to a byte budget per
       message (SIMLOG_CHUNK_BYTES), never splitting a single fixture's log across two messages
The visualizer decodes 0x01 with parseSimulatedCandidates (after stripping the tag byte) and every
0x02 message with parseSimulatedSimLogs (plural - it splits a chunk's concatenated entries apart;
it's called once per chunk, so multiple chunks just mean multiple calls).

Why chunked instead of one-per-fixture or one giant message: sending 54 separate small binary WS
messages back-to-back was observed to arrive at a real browser WebSocket client as 1-2 oversized,
corrupted messages (frame boundaries lost); sending everything as a single very large message (tens
of MB) was observed to simply never arrive at all in a constrained/sandboxed browser environment.
Both were confirmed correct at the wire level with a plain Node WebSocket client - this is a
transport/environment quirk with either extreme, not a framing or decode bug. Chunking to a modest
byte budget per message avoids both failure modes.

Run:  py "util tools/run_simulation_batch.py" [--host H] [--port P]
Then open the visualizer, go to the Simulated tab, and Connect (default ws://localhost:8765).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "cpp simulator" / "build" / "cpp_simulator_stream.exe"
FIXTURE_DIR = REPO / "flying machines" / "json" / "subset"
GENETIC_ML = REPO / "genetic algorithm"
MSYS_BIN = r"C:\msys64\ucrt64\bin"

TAG_CANDIDATES = b"\x01"
TAG_SIMLOG = b"\x02"
_SIMLOG_ENTRY_HEADER = struct.Struct("<iI")  # id (int32) + byteLength (uint32) per entry
SIMLOG_CHUNK_BYTES = 2_000_000  # per-message budget; see module docstring for why chunked at all


def _build_backlog() -> tuple[str, bytes, list[bytes]]:
    """Runs every *.json fixture once and returns (names_json, candidates_frame, [simlog_chunk, ...])."""
    sys.path.insert(0, str(GENETIC_ML))
    from genetic_ml.compact_format import encode_candidate

    fixtures = sorted(FIXTURE_DIR.glob("*.json"))
    if not fixtures:
        raise FileNotFoundError(f"no .json fixtures found in {FIXTURE_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        entries = []
        dat_paths = []
        for i, json_path in enumerate(fixtures, start=1):
            candidate = json.loads(json_path.read_text(encoding="utf-8").splitlines()[0])
            candidate["id"] = i  # de-duplicate ids across the whole folder, see module docstring
            dat_path = workdir / f"{json_path.stem}.dat"
            dat_path.write_bytes(encode_candidate(candidate))
            dat_paths.append(dat_path)
            entries.append({"name": json_path.stem, "candidate": candidate})

        # tracePathForCandidate (main.cpp) names per-candidate files "<stem>-<id><ext>" by
        # splitting off the LAST dot in the base filename - give it one so files come out
        # "batch-<id>.simlog" instead of extension-less "batch-<id>".
        sim_base = workdir / "simlogs" / "batch.simlog"
        env = os.environ.copy()
        env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
        # None of these fixtures have negative y (see run-test-json-flyers.bat); skip the
        # default +64 offset so logged/streamed coordinates match the source JSON exactly.
        env["MCP1122_CPP_NO_Y_OFFSET"] = "1"
        proc = subprocess.run(
            [str(EXE), *[str(p) for p in dat_paths], "--simulation-data", str(sim_base)],
            env=env, capture_output=True, text=True, check=True,
        )
        result_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if len(result_lines) != len(entries):
            raise RuntimeError(
                f"expected {len(entries)} result line(s), got {len(result_lines)} - stdout/fixture "
                "order assumption broke (see run-test-json-flyers.bat for the same assumption)"
            )

        candidate_bytes = []
        simlog_entries = []
        names = {}
        print(f"{'fixture':<32} {'ok':<6} {'validCycle':<11}")
        for entry, line in zip(entries, result_lines):
            r = json.loads(line)
            print(f"{entry['name']:<32} {str(r.get('ok')):<6} {str(r.get('validCycle')):<11}")
            candidate_bytes.append(encode_candidate(entry["candidate"]))
            names[entry["candidate"]["id"]] = entry["name"]

            simlog_path = workdir / "simlogs" / f"batch-{entry['candidate']['id']}.simlog"
            if simlog_path.exists():
                log_bytes = simlog_path.read_bytes()
                simlog_entries.append(
                    _SIMLOG_ENTRY_HEADER.pack(entry["candidate"]["id"], len(log_bytes)) + log_bytes
                )
            else:
                print(f"  warning: {entry['name']} produced no .simlog file")

        candidates_frame = TAG_CANDIDATES + b"".join(candidate_bytes)
        names_json = json.dumps({"names": names})

        # Chunk to a byte budget, never splitting one fixture's entry across two chunks.
        simlog_chunks = []
        current: list[bytes] = []
        current_size = 0
        for entry_bytes in simlog_entries:
            if current and current_size + len(entry_bytes) > SIMLOG_CHUNK_BYTES:
                simlog_chunks.append(TAG_SIMLOG + b"".join(current))
                current, current_size = [], 0
            current.append(entry_bytes)
            current_size += len(entry_bytes)
        if current:
            simlog_chunks.append(TAG_SIMLOG + b"".join(current))

        return names_json, candidates_frame, simlog_chunks


async def _serve_client(ws, names_json: str, candidates_frame: bytes, simlog_chunks: list[bytes]) -> None:
    peer = getattr(ws, "remote_address", "?")
    print(f"[+] client {peer} connected")
    try:
        await ws.send(names_json)
        await ws.send(candidates_frame)
        for chunk in simlog_chunks:
            await ws.send(chunk)
        print(f"    sent candidates + simulation_data ({len(simlog_chunks)} chunk(s))")
        await ws.wait_closed()
    except ConnectionClosed:
        pass
    finally:
        print(f"[-] client {peer} disconnected")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not EXE.exists():
        print(f"error: exe not built: {EXE}\nbuild it via 'cpp simulator/build-cpp.bat' first.")
        raise SystemExit(1)

    print("running all fixtures once...")
    names_json, candidates_frame, simlog_chunks = _build_backlog()
    total_bytes = sum(len(c) for c in simlog_chunks)
    print(f"backlog ready: {total_bytes} simulation_data byte(s) in {len(simlog_chunks)} chunk(s)")

    async with serve(
        lambda ws: _serve_client(ws, names_json, candidates_frame, simlog_chunks),
        args.host, args.port, max_size=None,
    ):
        print(f"streaming on ws://{args.host}:{args.port}  (Ctrl-C to stop)")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
