"""The dashboard - rlgym/serve.py.

The point of these is that the server reads a **live** Loop, on another thread, while the
training thread is still writing to it. That is the only genuinely new failure mode this file
introduces, so it is what most of this tests: a dashboard that 500s during a healthy run, or
that hands back a machine that is not the one asked for, is worse than no dashboard.

Nothing here trains or simulates. A Loop is constructed and its store is written directly,
which is exactly what the training thread does to it.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from rlgym.config import Config
from rlgym.loop import Loop
from rlgym.serve import LIBRARY_LIMIT, PAGE, serve, state_of


@pytest.fixture
def live(tmp_path, engine):
    """A Loop with one seeded machine, its server running, as the real thing would be."""
    config = Config()
    config.loop.rounds = 3
    loop = Loop(config, tmp_path / "run")
    loop.store.seed("engine", engine.to_candidate(cid=0), period=10, shift=(1, 0, 0))

    finished = threading.Event()
    # Port 0 asks the OS for a free one. A fixed port makes the suite fail when anything else
    # on the machine - including a real training run - happens to hold it.
    server = serve(loop, 0, time.perf_counter(), finished)
    yield loop, finished, f"http://127.0.0.1:{server.server_address[1]}"
    finished.set()
    server.shutdown()


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def test_the_page_is_served(live):
    _, _, base = live
    body = get(base + "/").decode("utf-8")
    assert body == PAGE
    assert "/api/state" in body, "the page cannot refresh without calling the API"


def test_state_reports_the_library(live):
    loop, _, base = live
    state = json.loads(get(base + "/api/state"))
    assert state["running"] is True
    assert state["rounds"] == 3
    assert len(state["library"]) == 1
    assert state["library"][0]["period"] == 10


def test_state_stops_saying_running_when_the_loop_ends(live):
    loop, finished, base = live
    finished.set()
    assert json.loads(get(base + "/api/state"))["running"] is False


def test_the_config_is_not_shipped_to_the_browser(live):
    """Every metrics row repeats the whole config (metrics.py). That is right for a file
    answerable years later and pointless on a 2-second poll - a 10-round run would send the
    same nested config ten times per request."""
    loop, _, base = live
    loop.metrics.log("round", round_index=1, seconds=1.0, headline={}, library={}, replay=0)
    row = json.loads(get(base + "/api/state"))["rows"][0]
    assert "config" not in row
    assert row["round_index"] == 1


def test_only_round_rows_reach_the_page(live):
    """`step` and `eval` rows are far more numerous and the page has nowhere to put them."""
    loop, _, base = live
    loop.metrics.log("step", step=1)
    loop.metrics.log("round", round_index=1, seconds=1.0, headline={}, library={}, replay=0)
    rows = json.loads(get(base + "/api/state"))["rows"]
    assert [row["kind"] for row in rows] == ["round"]


def test_a_machine_downloads_as_a_fixture(live):
    """A discovery is only useful if it opens in the tooling that already exists, so the
    download is the candidate in fixture form rather than a bespoke shape."""
    loop, _, base = live
    digest = next(iter(loop.store.entries))
    candidate = json.loads(get(f"{base}/api/machine/{digest}"))
    assert candidate == loop.store.entries[digest].candidate
    assert {"id", "trigger", "blocks"} <= set(candidate)


def test_an_unknown_machine_is_a_404_not_a_crash(live):
    _, _, base = live
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(base + "/api/machine/" + "0" * 64)
    assert caught.value.code == 404


def test_state_survives_the_library_growing_underneath_it(live, engine):
    """The one failure this file can genuinely introduce.

    CPython raises RuntimeError if a dict changes size mid-iteration, and the training thread
    admits machines whenever it likes. A dashboard that 500s during a healthy run would look
    exactly like the run itself having failed.
    """
    loop, finished, base = live
    stop = threading.Event()

    def churn() -> None:
        index = 0
        while not stop.is_set():
            index += 1
            cells = dict(engine.cells)
            cells[(50 + index, 0, 0)] = 20  # a cell far outside the machine, so the hash differs
            loop.store.admit(engine.to_candidate(cells, cid=0), None, 1, 10, (0, 0, 0))

    writer = threading.Thread(target=churn, daemon=True)
    writer.start()
    try:
        for _ in range(25):
            state = json.loads(get(base + "/api/state"))
            assert isinstance(state["library"], list)
    finally:
        stop.set()
        writer.join(timeout=5)

    # ...and the payload stays bounded while the library does not. This test is what found
    # that: letting the writer run free made one poll serialise every entry, and 25 polls took
    # 66 seconds. A dashboard is a viewport; library.json on disk is the store.
    final = json.loads(get(base + "/api/state"))
    assert final["library_total"] > len(final["library"])
    assert len(final["library"]) == LIBRARY_LIMIT


def test_state_of_needs_no_server(live):
    """The reader is a plain function of the loop, so what the page shows can be asserted
    without a socket - which is what keeps these tests from being flaky."""
    loop, finished, _ = live
    state = state_of(loop, time.perf_counter(), finished)
    assert state["round"] == 0 and state["attempts"] == 0
    assert state["library"][0]["name"] == "engine"
