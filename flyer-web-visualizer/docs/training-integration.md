# Streaming live machines to the `/live` dashboard

The visualizer's **Live Training** page connects to a WebSocket and renders machines as they
arrive. To feed it from the real training loop you host a small WS server that:

1. **On each new connection** sends the current historical backlog (one binary frame).
2. **Then streams** new machines as the loop discovers them (one frame per batch).

The frontend resets its view on every connection, so the server **must** resend the full backlog
each time a client connects (this is what makes the frontend's *auto-reconnect* checkbox work).

## Wire format (already implemented on both ends)

A frame is just concatenated compact records — the same `.data` bytes on disk. Reuse the existing
encoder; no new format:

```python
from genetic_ml.compact_format import encode_candidate   # candidate -> bytes (20B header + 16B/block)

frame = b"".join(encode_candidate(c) for c in candidates)  # one WS binary frame = one "batch"
```

The browser decodes each frame with `parseCompactData` (`src/lib/data.ts`) as one batch. There is
**no outer length prefix** — records are self-delimiting, so a frame may hold 1 or 10,000 machines.

`scripts/stream_flyers.py` is an offline stand-in that does exactly this from a saved file; the steps
below wire the same behavior into the live loop.

## 1. Declare the server once, with the backlog (top of training)

Run the WS server in a background thread so it never blocks the (synchronous) GA/RL loop. Drop this
helper in as e.g. `genetic_ml/stream_hub.py`:

```python
# genetic_ml/stream_hub.py
import asyncio, threading
from websockets.asyncio.server import serve

class StreamHub:
    """Background-thread WebSocket hub: backfills the backlog on connect, fans out live frames."""
    def __init__(self, backlog: bytes = b"", host="localhost", port=8765):
        self._backlog = bytearray(backlog)   # every record sent so far, in wire format
        self._clients: set[asyncio.Queue] = set()
        self._host, self._port = host, port
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        await serve(self._client, self._host, self._port, max_size=None)
        print(f"[stream] ws://{self._host}:{self._port}")

    async def _client(self, ws):
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.add(q)
            backlog = bytes(self._backlog)
        try:
            if backlog:
                await ws.send(backlog)            # (1) historical backfill, one frame
            while True:
                await ws.send(await q.get())      # (2) live batches
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(q)

    def publish(self, frame: bytes):
        """Thread-safe. Append to the backlog and fan a batch out to all connected clients."""
        if not frame:
            return
        with self._lock:
            self._backlog += frame
            clients = list(self._clients)
        for q in clients:
            self._loop.call_soon_threadsafe(q.put_nowait, frame)
```

Wire it in near the top of `main_ga.py` (the on-disk `flyers.data` is already in wire format, so the
backlog is just its bytes):

```python
from genetic_ml.stream_hub import StreamHub

COMPACT_FILE = COMPACT_DIR / "flyers.data"
backlog = COMPACT_FILE.read_bytes() if COMPACT_FILE.exists() else b""
hub = StreamHub(backlog=backlog)   # backfill included with the declaration
hub.start()
```

## 2. Stream new machines from the loop

Wherever the loop records a working discovery (in `ga_loop.run_ga`, alongside
`resolved_working_writer.save(child)`), publish the same candidates as a batch:

```python
# after a generation's working machines are found:
new_working = [child for child, res in offspring if res.working and _is_new(child)]
if new_working:
    hub.publish(b"".join(encode_candidate(c) for c in new_working))
```

Batch granularity is up to you — publish per generation, or accumulate and publish every N
discoveries (mirroring `CompactWorkingWriter`'s `flush_every`). Each `publish` becomes one batch the
dashboard shows at the top; the bottom pager keeps the full history.

## Notes

- **Backlog grows in memory.** `StreamHub` keeps every published frame so late joiners get full
  history. For very long runs, cap it (e.g. keep the last K records) or backfill from disk on connect
  instead of an in-memory buffer.
- **One hub, many clients.** Multiple browser tabs can watch the same run; each gets the backlog then
  the same live batches.
- **Thread safety.** The loop only ever calls `hub.publish(...)` — it's the single synchronous entry
  point; everything async happens on the hub's own thread.
- **RL vs GA.** The loop here is a GA, but the contract is identical for an RL trainer: encode each
  emitted machine's candidate and `publish` it.
