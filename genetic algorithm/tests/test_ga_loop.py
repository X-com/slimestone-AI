from __future__ import annotations

import io

from genetic_ml.compact_format import decode_candidate, read_compact_file
from genetic_ml.compact_working_writer import CompactWorkingWriter
from genetic_ml.ga_loop import GAConfig, run_ga

_SEED = {
    "id": 1,
    "trigger": {"x": 0, "y": 0, "z": 0},
    "blocks": [
        {"x": 0, "y": 0, "z": 0, "state": 33},  # piston
        {"x": 0, "y": 1, "z": 0, "state": 152},  # redstone block
        {"x": 0, "y": 2, "z": 0, "state": 165},  # slime
    ],
}


class _StubPool:
    """Fakes genetic_ml.simulator_pool.SimulatorPool - run_ga only ever uses it as a context
    manager plus run_all(), both of which this fakes without touching the real simulator."""

    def __init__(self, working: bool) -> None:
        self.working = working
        self.crash_count = 0
        self.hang_count = 0

    def __enter__(self) -> "_StubPool":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run_all(self, candidates: list[dict]) -> list[dict]:
        return [{"validCycle": self.working} for _ in candidates]


class _StubHub:
    """Fakes genetic_ml.stream_hub.StreamHub's publish(frame) - run_ga only ever calls that one
    method."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def publish(self, frame: bytes) -> None:
        self.frames.append(frame)


def _decode_frame(frame: bytes) -> list[dict]:
    stream = io.BytesIO(frame)
    candidates = []
    while True:
        candidate = decode_candidate(stream)
        if candidate is None:
            return candidates
        candidates.append(candidate)


def test_a_generations_working_discoveries_publish_as_one_frame(tmp_path):
    working_writer = CompactWorkingWriter(tmp_path / "compact")
    hub = _StubHub()

    run_ga(
        simulator_config=None,
        ga_config=GAConfig(population_capacity=4, offspring_per_lineage=2, generations=1, seed=1),
        seed_candidates=[_SEED],
        working_hashes_path=str(tmp_path / "working_hashes.log"),
        not_working_hashes_path=str(tmp_path / "not_working_hashes.log"),
        working_writer=working_writer,
        pool=_StubPool(working=True),
        stream_hub=hub,
    )

    working_writer.flush()
    saved = read_compact_file(tmp_path / "compact" / "flyers.data")
    assert saved  # the stub pool reports every mutant as working, so there's something to check

    assert len(hub.frames) == 1  # one generation -> one publish, not one per discovery
    published = _decode_frame(hub.frames[0])
    assert {c["id"] for c in published} == {c["id"] for c in saved}


def test_a_generation_with_no_discoveries_never_publishes(tmp_path):
    hub = _StubHub()

    run_ga(
        simulator_config=None,
        ga_config=GAConfig(population_capacity=4, offspring_per_lineage=2, generations=1, seed=1),
        seed_candidates=[_SEED],
        working_hashes_path=str(tmp_path / "working_hashes.log"),
        not_working_hashes_path=str(tmp_path / "not_working_hashes.log"),
        pool=_StubPool(working=False),
        stream_hub=hub,
    )

    assert hub.frames == []
