"""Streaming discoveries to the flyer-web-visualizer's Live Training page.

The viewer already exists and is far better than anything worth rebuilding here:
`flyer-web-visualizer/src/TrainingDashboard.svelte` renders machines in 3D with orbit and zoom,
split latest-batch and history viewports, paging, click-to-select and auto-reconnect. This file
is only the producer side, and it follows `flyer-web-visualizer/docs/training-integration.md`
rather than inventing a protocol.

**The wire format needed no work.** `parseCompactData` in the viewer reads a 20-byte `<iiiiI>`
header plus 16-byte `<iiiI>` blocks, which is byte-for-byte what `game.py:encode_candidate`
already emits. Geometry is republished unchanged.

**Two frame kinds on one socket**, told apart by type on the client (`e.data` is an ArrayBuffer
for binary and a string for text):

    binary   concatenated compact records - the machines themselves
    text     one JSON object, {"machines": {<candidate id>: {...}}, "run": {...}}

and one request kind travelling the other way, `{"want": "animation", "id": N}`, answered with
`{"animation": {...}, "id": N}` to that viewer alone. Animations are built **on demand** rather
than attached to every discovery: a machine is fully described by its blocks, so the record is
reproducible from what the viewer already has, and the training loop pays nothing for animations
nobody asks to watch.

The second exists because the compact format carries geometry and nothing else, so period,
shift, generation, round found and the redundant-block count have nowhere else to travel. The
join key is the compact header's `id`, which this file assigns from a monotonic counter - the
loop itself always uses `cid=0`, so without that every machine would collide on id 0.

**Backlog is capped, deliberately.** The viewer resets its view on every connection, so the
server must resend history or auto-reconnect shows an empty page. The integration doc flags the
consequence in its own Notes: an uncapped backlog is resent in full on every reconnect, and it
lands on the client's `machines.push(...batch)`, which spreads into arguments and throws
`RangeError: Maximum call stack size exceeded` on a large enough one (measured in V8: fine at 100,000, RangeError at 200,000). That is a
crash rather than a slowdown, so the cap is not a nicety.

**`websockets` is this project's one dependency.** Hand-rolling RFC 6455 is small for a
send-only server on localhost and stops being small the moment it is not: idle keepalive (a
proxy or NAT closes idle connections after 30-60s, and the failure looks like the training
having crashed), `permessage-deflate` (block records are repetitive int32s that compress
enormously), and the likelihood that a decentralised deployment inverts the direction so this
becomes a *client*, where every frame must be masked.
"""
from __future__ import annotations

import asyncio
import io
import json
import threading
from collections import deque
from typing import Any, Callable, Iterable

from rlgym.game import Candidate, decode_candidate, encode_candidate

DEFAULT_PORT = 8765
# Records kept for replay to a late or reconnecting client. Matched to the viewer's own history
# cap (MAX_HISTORY_PAGES * PAGE = 2,500 in TrainingDashboard.svelte) - sending more than the
# client will retain is pure cost at both ends.
DEFAULT_BACKLOG = 2000


class StreamHub:
    """A WebSocket server on a background thread, so it never blocks the synchronous loop.

    Every public method is safe to call from the training thread; nothing here touches the
    event loop directly except through `call_soon_threadsafe`.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        backlog: int = DEFAULT_BACKLOG,
        animate: Callable[[Candidate], dict] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        # Candidate -> the JSON animatedScene.ts plays (rlgym.animation.animate). Optional so a
        # test, or a run that only wants the stream, needs no simulator.
        self.animate = animate
        # (record bytes, metadata dict) pairs, newest last. One deque rather than two, so the
        # two halves cannot drift out of step as it evicts.
        self._backlog: deque[tuple[bytes, dict]] = deque(maxlen=backlog)
        self._run: dict[str, Any] = {}
        self._clients: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._ready = threading.Event()
        self._next_id = 0
        self.error: BaseException | None = None

    # --- lifecycle --------------------------------------------------------------------

    def start(self, timeout: float = 5.0) -> bool:
        """Bring the server up, returning whether it is actually listening.

        Waits for the bind rather than assuming it: a port already in use is the common case
        (a second training run, or a stale process), and a hub that silently failed to bind
        looks exactly like a viewer that will not connect.
        """
        if self._thread is not None:
            return self.error is None
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self.error is None and self._ready.is_set()

    def _serve_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(lambda: self._loop.create_task(self._listen()))
        self._loop.run_forever()

    async def _listen(self) -> None:
        try:
            from websockets.asyncio.server import serve

            # max_size=None: a backlog frame legitimately runs to megabytes, and the default
            # 1 MB cap would drop exactly the frame a reconnecting client needs most.
            self._server = await serve(
                self._client, self.host, self.port, max_size=None
            )
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            self.error = exc
        finally:
            self._ready.set()

    async def _shutdown(self) -> None:
        """Release the client handlers, then close the listener.

        The sentinel is load-bearing. Each handler parks on `queue.get()`, which closing the
        socket does NOT wake - so `wait_closed()` waits on handlers that will never return, and
        a 0.25s test suite became an 18s one before this was added.
        """
        with self._lock:
            clients = list(self._clients)
        for queue in clients:
            queue.put_nowait(None)
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=1.5)
            except Exception:  # noqa: BLE001 - already going down; nothing to salvage
                pass

    def stop(self) -> None:
        """Close the listener, then the loop, then join.

        Closing the server first is not tidiness: stopping the loop out from under a live
        acceptor leaves pending overlapped I/O, which Windows reports as a wall of
        "Task was destroyed but it is pending" and one "handle is invalid" traceback. That
        noise is indistinguishable from a real failure the next time something does go wrong.
        """
        if self._thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=2.0)
        except Exception:  # noqa: BLE001 - a hub that will not close must not block the caller
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    # --- serving one client -----------------------------------------------------------

    async def _client(self, websocket) -> None:
        """Backfill, then follow.

        The viewer clears its history on every open, so the backlog has to be resent or
        auto-reconnect would repopulate an empty page. Sent as ONE binary frame plus one JSON
        frame, which is what the client's per-frame batching expects.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.add(queue)
            history = list(self._backlog)
            run = dict(self._run)
        # The send side parks on `queue.get()` forever, so listening needs its own task. Replies
        # go back onto this client's own queue: an animation was asked for by one viewer and is
        # of no interest to the others.
        reader = asyncio.create_task(self._read(websocket, queue))
        try:
            if history:
                await websocket.send(b"".join(record for record, _ in history))
                await websocket.send(
                    json.dumps(
                        {
                            "machines": {str(m["id"]): m for _, m in history},
                            "run": run,
                            "backfill": True,
                        }
                    )
                )
            while True:
                item = await queue.get()
                if item is None:
                    break  # shutdown sentinel - see _shutdown
                binary, text = item
                # A reply carries no geometry; sending an empty binary frame would make the
                # client rebuild its scene from zero machines.
                if binary is not None:
                    await websocket.send(binary)
                await websocket.send(text)
        except Exception:  # noqa: BLE001 - a client vanishing is normal, never fatal here
            pass
        finally:
            reader.cancel()
            with self._lock:
                self._clients.discard(queue)

    async def _read(self, websocket, queue: asyncio.Queue) -> None:
        try:
            async for message in websocket:
                reply = await self._answer(message)
                if reply is not None:
                    queue.put_nowait((None, reply))
        except Exception:  # noqa: BLE001 - same as the send side: a client leaving is not news
            pass

    async def _answer(self, message: Any) -> str | None:
        """One request from one viewer. Returns the JSON to send back, or None to ignore it.

        Unknown requests are ignored rather than refused: the viewer and this file are versioned
        separately, and a newer page asking for something this run cannot do should degrade to a
        button that does nothing, not a dropped connection.
        """
        if not isinstance(message, str):
            return None
        try:
            request = json.loads(message)
        except ValueError:
            return None
        if not isinstance(request, dict) or request.get("want") != "animation":
            return None
        cid = request.get("id")
        if self.animate is None:
            return json.dumps({"animation": None, "id": cid, "error": "this run has no simulator"})
        candidate = self._candidate(cid)
        if candidate is None:
            return json.dumps({"animation": None, "id": cid, "error": f"no machine with id {cid}"})
        try:
            # In a thread: this spawns the simulator and takes long enough that doing it inline
            # would stall every other viewer's frames behind one person clicking a button.
            record = await asyncio.get_running_loop().run_in_executor(
                None, self.animate, candidate
            )
        except Exception as exc:  # noqa: BLE001 - reported to the viewer, never fatal here
            return json.dumps({"animation": None, "id": cid, "error": str(exc)})
        return json.dumps({"animation": record, "id": cid})

    def _candidate(self, cid: Any) -> Candidate | None:
        """Recover a published machine from the backlog. The bytes already on the wire are the
        candidate, so nothing extra has to be retained to make this answerable."""
        with self._lock:
            record = next((r for r, m in self._backlog if m.get("id") == cid), None)
        return None if record is None else decode_candidate(io.BytesIO(record))

    # --- publishing -------------------------------------------------------------------

    def publish(
        self, machines: Iterable[tuple[Candidate, dict]], run: dict | None = None
    ) -> int:
        """Send a batch of (candidate, metadata) pairs. Thread-safe. Returns how many went.

        Called once per round rather than once per discovery: the viewer's top viewport is
        captioned "Latest batch", and a batch of one would make it flicker on every machine
        found while telling you no more than the console already does.
        """
        batch = list(machines)
        if not batch:
            # Still worth forwarding run stats on a barren round - "nothing found" is a result,
            # and a stats strip frozen at the last good round would misreport it.
            if run is not None:
                self._publish_run(run)
            return 0

        records: list[bytes] = []
        meta: dict[str, dict] = {}
        with self._lock:
            for candidate, info in batch:
                stream_id = self._next_id
                self._next_id += 1
                # The loop always builds candidates with cid=0, so the id is assigned here and
                # the SAME id is what the metadata is keyed by. Encoding a copy keeps the
                # caller's candidate untouched.
                record = encode_candidate({**candidate, "id": stream_id})
                entry = {**info, "id": stream_id}
                records.append(record)
                meta[str(stream_id)] = entry
                self._backlog.append((record, entry))
            if run is not None:
                self._run = dict(run)
            payload = json.dumps({"machines": meta, "run": dict(self._run)})
            clients = list(self._clients)

        frame = (b"".join(records), payload)
        for queue in clients:
            self._loop.call_soon_threadsafe(queue.put_nowait, frame)
        return len(records)

    def _publish_run(self, run: dict) -> None:
        """Run stats with no machines. An empty binary frame decodes to zero machines, which
        the client already handles, so the shape stays uniform rather than needing its own case."""
        with self._lock:
            self._run = dict(run)
            payload = json.dumps({"machines": {}, "run": dict(self._run)})
            clients = list(self._clients)
        for queue in clients:
            self._loop.call_soon_threadsafe(queue.put_nowait, (b"", payload))

    # --- introspection, for tests and the console -------------------------------------

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def backlog_size(self) -> int:
        with self._lock:
            return len(self._backlog)
