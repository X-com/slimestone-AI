"""Exercises StreamHub's actual threading/asyncio/queue fan-out over a real socket, rather than
mocking it away - that's where the genuine concurrency risk lives (backlog snapshot vs. client
registration, call_soon_threadsafe fan-out from a foreign thread)."""
from __future__ import annotations

import asyncio

from websockets.asyncio.client import connect

from genetic_ml.stream_hub import StreamHub

_PORT = 18765


async def _connect_with_retries(url: str, attempts: int = 50):
    for _ in range(attempts):
        try:
            return await connect(url)
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"could not connect to {url}")


async def _client_gets_backlog_then_live_publishes() -> None:
    hub = StreamHub(backlog=b"existing-backlog", host="localhost", port=_PORT)
    hub.start()

    ws = await _connect_with_retries(f"ws://localhost:{_PORT}")
    try:
        backlog_frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
        assert backlog_frame == b"existing-backlog"

        hub.publish(b"live-batch")
        live_frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
        assert live_frame == b"live-batch"
    finally:
        await ws.close()


def test_client_gets_backlog_then_live_publishes():
    asyncio.run(_client_gets_backlog_then_live_publishes())


async def _publish_before_any_client_connects_joins_the_backlog() -> None:
    hub = StreamHub(host="localhost", port=_PORT + 1)
    hub.start()
    hub.publish(b"before-connect")

    ws = await _connect_with_retries(f"ws://localhost:{_PORT + 1}")
    try:
        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
        assert frame == b"before-connect"
    finally:
        await ws.close()


def test_publish_before_any_client_connects_is_still_in_the_backlog():
    asyncio.run(_publish_before_any_client_connects_joins_the_backlog())
