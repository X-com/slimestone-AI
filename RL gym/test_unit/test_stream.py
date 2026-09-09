"""The discovery stream - rlgym/stream.py.

What matters here is that a real browser can read what this sends, so the tests connect a real
WebSocket client to a real server rather than calling methods on the hub. Two things they are
specifically guarding:

  * **the wire format the viewer already speaks** - `parseCompactData` in the visualizer reads a
    20-byte header plus 16 bytes per block, and a producer that drifts from that fails as an
    empty page rather than as an error;
  * **the backlog cap** - the viewer resets on every connect, so history must be resent, and the
    integration doc's own Notes warn that an uncapped backlog is what breaks a long run.
"""
from __future__ import annotations

import asyncio
import json
import struct

import pytest

from rlgym.game import encode_candidate
from rlgym.stream import StreamHub

websockets = pytest.importorskip("websockets", reason="the stream needs websockets")

HEADER = struct.Struct("<iiiiI")  # exactly what data.ts:parseCompactData expects
BLOCK = struct.Struct("<iiiI")


@pytest.fixture
def hub():
    """A hub on an OS-chosen port, so the suite never collides with a real training run."""
    hub = StreamHub(host="127.0.0.1", port=0)
    assert hub.start(), f"hub failed to bind: {hub.error}"
    # Port 0 means the OS picked one; ask the server what it actually got.
    hub.port = hub._server.sockets[0].getsockname()[1]
    yield hub
    hub.stop()


def candidate(cid: int = 0, blocks: int = 3):
    return {
        "id": cid,
        "trigger": {"x": 1, "y": 2, "z": 3},
        "blocks": [{"x": i, "y": 0, "z": 0, "state": 20} for i in range(blocks)],
    }


async def _collect(url: str, frames: int, timeout: float = 5.0):
    async with websockets.connect(url) as ws:
        return [await asyncio.wait_for(ws.recv(), timeout) for _ in range(frames)]


def collect(hub: StreamHub, frames: int):
    return asyncio.run(_collect(hub.url, frames))


# --- the wire format --------------------------------------------------------------------


def test_a_frame_is_records_the_visualizer_can_parse(hub):
    """Decoded the way data.ts does it, byte offset by byte offset. If this drifts, the viewer
    shows an empty page rather than an error, so it is checked here and not there."""
    hub.publish([(candidate(blocks=2), {"name": "m"})])
    binary, _ = collect(hub, 2)

    assert isinstance(binary, bytes)
    cid, tx, ty, tz, count = HEADER.unpack_from(binary, 0)
    assert (tx, ty, tz) == (1, 2, 3)
    assert count == 2
    assert len(binary) == HEADER.size + count * BLOCK.size
    x, y, z, state = BLOCK.unpack_from(binary, HEADER.size)
    assert (x, y, z, state) == (0, 0, 0, 20)


def test_several_machines_ride_in_one_frame(hub):
    """The records are self-delimiting with no outer length prefix, so one frame is one batch
    of any size - which is what makes the viewer's 'Latest batch' viewport meaningful."""
    hub.publish([(candidate(blocks=1), {}), (candidate(blocks=4), {}), (candidate(blocks=2), {})])
    binary, _ = collect(hub, 2)

    sizes, offset = [], 0
    while offset < len(binary):
        count = HEADER.unpack_from(binary, offset)[4]
        sizes.append(count)
        offset += HEADER.size + count * BLOCK.size
    assert sizes == [1, 4, 2]
    assert offset == len(binary), "the split must account for every byte, exactly"


def test_the_encoder_is_the_projects_own(hub):
    """Not a second implementation. game.py already emits this format and round-trips it."""
    machine = candidate(cid=7, blocks=3)
    hub.publish([(machine, {})])
    binary, _ = collect(hub, 2)
    assert binary == encode_candidate({**machine, "id": 0}), "ids are reassigned by the hub"


# --- metadata ---------------------------------------------------------------------------


def test_metadata_arrives_keyed_by_the_id_in_the_record(hub):
    """The compact format carries geometry only, so everything else rides a second JSON frame.
    The join key has to be the id inside the binary record or the two cannot be matched up."""
    hub.publish([(candidate(), {"name": "gen1-abc", "period": 20, "redundant_removed": 2})])
    binary, text = collect(hub, 2)

    stream_id = HEADER.unpack_from(binary, 0)[0]
    payload = json.loads(text)
    entry = payload["machines"][str(stream_id)]
    assert entry["name"] == "gen1-abc"
    assert entry["period"] == 20
    assert entry["redundant_removed"] == 2


def test_ids_are_unique_across_batches(hub):
    """The loop builds every candidate with cid=0. Without reassignment every machine would
    collide on id 0 and the whole run would share one metadata entry."""
    hub.publish([(candidate(), {"name": "a"}), (candidate(), {"name": "b"})])
    hub.publish([(candidate(), {"name": "c"})])
    binary, text = collect(hub, 2)  # a reconnect gets the whole backlog

    ids, offset = [], 0
    while offset < len(binary):
        cid, _, _, _, count = HEADER.unpack_from(binary, offset)
        ids.append(cid)
        offset += HEADER.size + count * BLOCK.size
    assert len(set(ids)) == 3, "ids collided"
    names = [json.loads(text)["machines"][str(i)]["name"] for i in ids]
    assert names == ["a", "b", "c"]


def test_run_stats_ride_along(hub):
    hub.publish([(candidate(), {})], run={"round": 3, "rounds": 10, "library": 12})
    _, text = collect(hub, 2)
    assert json.loads(text)["run"] == {"round": 3, "rounds": 10, "library": 12}


def test_a_barren_round_still_reports(hub):
    """"Nothing found this round" is a result. A stats strip frozen at the last good round
    would show it as still current, which is worse than showing zero."""
    hub.publish([], run={"round": 4, "library": 12})
    hub.publish([(candidate(), {})], run={"round": 5, "library": 13})
    _, text = collect(hub, 2)
    assert json.loads(text)["run"]["round"] == 5


# --- the backlog ------------------------------------------------------------------------


def test_a_late_client_is_backfilled(hub):
    """The viewer clears its history on every open, so history must be resent or auto-reconnect
    repopulates an empty page."""
    hub.publish([(candidate(), {"name": "before"})])
    binary, text = collect(hub, 2)
    assert len(binary) > 0
    assert [m["name"] for m in json.loads(text)["machines"].values()] == ["before"]
    assert json.loads(text)["backfill"] is True


def test_the_backlog_is_capped(hub):
    """The integration doc's own Notes warn about this, and the failure is not a slowdown: the
    client appends a frame with `machines.push(...batch)`, which spreads into arguments and
    throws RangeError on a large enough one (measured in V8: fine at 100,000, RangeError at 200,000).
    An uncapped backlog eventually crashes the page rather than slowing it."""
    hub._backlog = type(hub._backlog)(maxlen=5)
    for index in range(20):
        hub.publish([(candidate(), {"name": f"m{index}"})])
    assert hub.backlog_size == 5

    binary, text = collect(hub, 2)
    names = [m["name"] for m in json.loads(text)["machines"].values()]
    assert names == ["m15", "m16", "m17", "m18", "m19"], "the cap must drop the OLDEST"

    count = 0
    offset = 0
    while offset < len(binary):
        blocks = HEADER.unpack_from(binary, offset)[4]
        count += 1
        offset += HEADER.size + blocks * BLOCK.size
    assert count == 5, "geometry and metadata disagree about what the backlog holds"


def test_publishing_with_nobody_listening_is_harmless(hub):
    """The common case: training starts before anyone opens the page."""
    assert hub.clients == 0
    assert hub.publish([(candidate(), {})]) == 1
    assert hub.backlog_size == 1


# --- failure modes ----------------------------------------------------------------------


def test_a_port_already_in_use_is_reported_not_swallowed(hub):
    """`serve.py` prints a warning and trains anyway. It can only do that if `start()` tells the
    truth about having bound - a hub that failed silently looks exactly like a viewer that will
    not connect."""
    second = StreamHub(host="127.0.0.1", port=hub.port)
    try:
        assert second.start() is False
        assert second.error is not None
    finally:
        second.stop()


def test_publish_returns_what_it_sent(hub):
    assert hub.publish([]) == 0
    assert hub.publish([(candidate(), {}), (candidate(), {})]) == 2


# --- requests from the viewer ------------------------------------------------------------
#
# The Simulate button asks for one machine's animation. A stub stands in for the real
# `rlgym.animation.animate` here: what these guard is the request path - routing, lookup and
# failure handling - not the projection, which test_animation.py covers against the simulator.


@pytest.fixture
def answering_hub():
    """A hub that answers animation requests by echoing back what it was asked to animate."""
    seen = []

    def fake_animate(candidate):
        seen.append(candidate)
        return {"blocks": candidate["blocks"], "events": [], "terminationTick": 7}

    hub = StreamHub(host="127.0.0.1", port=0, animate=fake_animate)
    assert hub.start(), f"hub failed to bind: {hub.error}"
    hub.port = hub._server.sockets[0].getsockname()[1]
    hub.seen = seen
    yield hub
    hub.stop()


async def _ask(url: str, request: dict, skip: int = 0, timeout: float = 5.0):
    async with websockets.connect(url) as ws:
        for _ in range(skip):  # the backfill frames a fresh connection always gets first
            await asyncio.wait_for(ws.recv(), timeout)
        await ws.send(json.dumps(request))
        return await asyncio.wait_for(ws.recv(), timeout)


def ask(hub: StreamHub, request: dict, skip: int = 0):
    return asyncio.run(_ask(hub.url, request, skip))


def test_a_viewer_can_ask_for_one_machine_s_animation(answering_hub):
    """The machine is recovered from the backlog - the bytes already sent ARE the candidate, so
    nothing extra is retained to make this answerable."""
    answering_hub.publish([(candidate(blocks=2), {"name": "m"})])
    reply = json.loads(ask(answering_hub, {"want": "animation", "id": 0}, skip=2))

    assert reply["id"] == 0
    assert reply["animation"]["terminationTick"] == 7
    assert len(reply["animation"]["blocks"]) == 2
    assert answering_hub.seen[0]["id"] == 0, "the hub's own stream id, which is what was published"


def test_an_answer_carries_no_geometry(answering_hub):
    """A reply is one TEXT frame. Pairing it with an empty binary frame - the shape a publish
    uses - would make the client rebuild its scene from zero machines."""
    answering_hub.publish([(candidate(), {})])
    reply = ask(answering_hub, {"want": "animation", "id": 0}, skip=2)
    assert isinstance(reply, str)


def test_an_unknown_id_is_answered_not_ignored(answering_hub):
    """Silence is indistinguishable from a hung simulator, and the button would spin forever."""
    answering_hub.publish([(candidate(), {})])
    reply = json.loads(ask(answering_hub, {"want": "animation", "id": 999}, skip=2))
    assert reply["animation"] is None
    assert "999" in reply["error"]


def test_a_hub_with_no_animator_says_so(hub):
    """`--no-stream` aside, a hub can legitimately be built without a simulator. The viewer has
    to hear that rather than wait."""
    hub.publish([(candidate(), {})])
    reply = json.loads(ask(hub, {"want": "animation", "id": 0}, skip=2))
    assert reply["animation"] is None and reply["error"]


def test_a_failing_animation_is_reported_as_a_message(hub_that_fails):
    """A simulator that refuses one machine must not take the connection - or the run - with it."""
    hub_that_fails.publish([(candidate(), {})])
    reply = json.loads(ask(hub_that_fails, {"want": "animation", "id": 0}, skip=2))
    assert reply["animation"] is None
    assert "no simulator here" in reply["error"]


@pytest.fixture
def hub_that_fails():
    def boom(candidate):
        raise RuntimeError("no simulator here")

    hub = StreamHub(host="127.0.0.1", port=0, animate=boom)
    assert hub.start(), f"hub failed to bind: {hub.error}"
    hub.port = hub._server.sockets[0].getsockname()[1]
    yield hub
    hub.stop()


def test_an_unrecognised_request_is_ignored_without_dropping_the_client(answering_hub):
    """The page and this file are versioned separately. A newer viewer asking for something this
    run cannot do should get a button that does nothing, not a closed socket - so the connection
    stays live and the next publish still arrives."""

    async def scenario():
        async with websockets.connect(answering_hub.url) as ws:
            await ws.send("this is not json")
            await ws.send(json.dumps({"want": "something-else", "id": 0}))
            answering_hub.publish([(candidate(blocks=5), {})])
            binary = await asyncio.wait_for(ws.recv(), 5.0)
            await asyncio.wait_for(ws.recv(), 5.0)
            return binary

    binary = asyncio.run(scenario())
    assert HEADER.unpack_from(binary, 0)[4] == 5
