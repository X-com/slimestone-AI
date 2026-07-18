"""Background-thread WebSocket hub for streaming live discoveries to the flyer-web-visualizer's
/live page. See flyer-web-visualizer/docs/training-integration.md for the wire format and the
full contract this implements - backfill the connecting client with everything published so far
in one frame, then fan out each later publish() as its own frame.
"""
from __future__ import annotations

import asyncio
import ssl
import threading

from websockets.asyncio.server import serve

from genetic_ml.dev_tls import tls_hint


class StreamHub:
    """Runs the WS server on its own thread so it never blocks the (synchronous) GA/RL loop.

    # ponytail: backlog is an unbounded in-memory buffer - fine for a bounded run, but a
    # continuous one (main_rl.py's ITERATIONS=None) will grow it forever. Cap it (keep the last
    # K records) or backfill from disk on connect instead, if a long run's memory use matters.
    """

    def __init__(
        self,
        backlog: bytes = b"",
        host: str = "localhost",
        port: int = 8765,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._backlog = bytearray(backlog)
        self._clients: set[asyncio.Queue] = set()
        self._host, self._port = host, port
        self._ssl_context = ssl_context
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._serve())
        self._loop.run_forever()

    async def _serve(self) -> None:
        await serve(self._client, self._host, self._port, max_size=None, ssl=self._ssl_context)
        scheme = "wss" if self._ssl_context is not None else "ws"
        print(f"[stream] {scheme}://{self._host}:{self._port}")
        if self._ssl_context is not None:
            print(tls_hint(self._host, self._port))

    async def _client(self, ws) -> None:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.add(q)
            backlog = bytes(self._backlog)
        try:
            if backlog:
                await ws.send(backlog)  # (1) historical backfill, one frame
            while True:
                await ws.send(await q.get())  # (2) live batches
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(q)

    def publish(self, frame: bytes) -> None:
        """Thread-safe. Append to the backlog and fan a batch out to all connected clients."""
        if not frame:
            return
        with self._lock:
            self._backlog += frame
            clients = list(self._clients)
        for q in clients:
            self._loop.call_soon_threadsafe(q.put_nowait, frame)
