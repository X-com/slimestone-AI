#!/usr/bin/env python3
"""Simulate the training loop's live machine stream for the visualizer's /live page.

Streams the compact ".data" binary (header <iiiiI> = id,trigger xyz,block_count; per block
<iiiI> = x,y,z,state; records concatenated) over a WebSocket. On each client connection it
sends the first half of the file as one binary frame, then 100 new records every 30s, then
idles until the file's records are exhausted. Stand-in until the real RL/GA loop feeds this.

Run:  python3 scripts/stream_flyers.py      (then open the app's /live page and click Connect)
"""
import argparse
import asyncio
import struct
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from dev_tls import build_ssl_context, tls_hint

_HEADER = struct.Struct("<iiiiI")  # id, trigger_x, trigger_y, trigger_z, block_count -> 20 bytes
_BLOCK_SIZE = 16  # <iiiI> per block
# The backend sibling directory is named "genetic-ml" in some checkouts and "genetic algorithm"
# in others (they've diverged) - try both, falling back to the first if neither exists yet.
_CANDIDATE_FILES = [
    Path(__file__).resolve().parents[2] / "genetic-ml/data/compact-working/flyers.data",
    Path(__file__).resolve().parents[2] / "genetic algorithm/data/compact-working/flyers.data",
]
_DEFAULT_FILE = next((p for p in _CANDIDATE_FILES if p.exists()), _CANDIDATE_FILES[0])


def split_records(buf: bytes) -> list[bytes]:
    """Split concatenated compact records into a list of per-record byte slices."""
    records: list[bytes] = []
    off, n = 0, len(buf)
    while off < n:
        if off + _HEADER.size > n:
            raise ValueError(f"truncated header at byte {off}")
        block_count = _HEADER.unpack_from(buf, off)[4]
        size = _HEADER.size + block_count * _BLOCK_SIZE
        if off + size > n:
            raise ValueError(f"truncated record at byte {off}")
        records.append(buf[off : off + size])
        off += size
    return records


async def stream(ws, records, initial, batch, interval):
    peer = getattr(ws, "remote_address", "?")
    print(f"[+] client {peer} connected")
    try:
        await ws.send(b"".join(records[:initial]))
        print(f"    sent initial {initial} records")
        i = initial
        while i < len(records):
            await asyncio.sleep(interval)
            chunk = records[i : i + batch]
            await ws.send(b"".join(chunk))
            i += len(chunk)
            print(f"    sent batch of {len(chunk)} ({i}/{len(records)})")
        print("    file exhausted; idling (socket stays open)")
        await ws.wait_closed()
    except ConnectionClosed:
        pass
    finally:
        print(f"[-] client {peer} disconnected")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, default=_DEFAULT_FILE)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch", type=int, default=100, help="records per live batch")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between batches")
    ap.add_argument("--initial", type=int, default=None, help="initial records (default: half)")
    ap.add_argument(
        "--no-tls", action="store_true", help="serve plain ws:// instead of wss:// (default: TLS on)"
    )
    args = ap.parse_args()

    raw = args.file.read_bytes()
    records = split_records(raw)
    # self-check: the split must account for every byte, exactly.
    assert sum(len(r) for r in records) == len(raw), "record split lost/gained bytes"
    initial = args.initial if args.initial is not None else len(records) // 2

    print(
        f"{args.file} -> {len(records)} records ({len(raw)} bytes); "
        f"initial {initial}, then +{args.batch} every {args.interval:g}s"
    )
    ssl_context = None if args.no_tls else build_ssl_context()
    async with serve(
        lambda ws: stream(ws, records, initial, args.batch, args.interval),
        args.host,
        args.port,
        max_size=None,
        ssl=ssl_context,
    ):
        scheme = "ws" if ssl_context is None else "wss"
        print(f"streaming on {scheme}://{args.host}:{args.port}  (Ctrl-C to stop)")
        if ssl_context is not None:
            print(tls_hint(args.host, args.port))
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
