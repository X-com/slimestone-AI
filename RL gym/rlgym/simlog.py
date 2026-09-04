"""Producing .simlog records for a batch of candidates.

Separate from sim.py because the two answer different questions and cost different amounts.
sim.py wants a verdict and nothing else, which is all Milestone 1 needed. This wants the full
record, which means --simulation-data, one file per candidate, and a temporary directory to put
them in.

That file-per-candidate cost is what DECISIONS.md point 25 proposes to remove by having the C++
write records to stdout. Not needed yet: Stage 0 writes a few thousand files once, which is
seconds of churn. It matters when the loop runs continuously at ~1,000/sec for days.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from rlgym.game import Candidate, encode_candidate
from rlgym.record import Record
from rlgym.sim import DEFAULT_SIMULATOR, SimConfig, _prepend_matching_runtime_dir


@contextmanager
def records_for(
    candidates: Sequence[Candidate], config: SimConfig | None = None
) -> Iterator[dict[int, Record]]:
    """Simulate every candidate with logging on, yielding {candidate id: Record}.

    The records are deleted when the context exits - they are a cache, regenerable at about a
    millisecond each (point 19), so nothing here is precious.
    """
    config = (config or SimConfig()).validated()
    if not candidates:
        yield {}
        return

    env = os.environ.copy()
    env["MCP1122_CPP_MAX_TICKS"] = str(config.max_ticks)
    env["MCP1122_CPP_STRUCTURAL_VERIFY"] = "1" if config.structural_verify else "0"
    _prepend_matching_runtime_dir(env)

    with tempfile.TemporaryDirectory(prefix="rlgym-simlog-") as tmp:
        base = Path(tmp) / "run"
        payload = b"".join(encode_candidate(c) for c in candidates)
        result = subprocess.run(
            [str(config.simulator_path), "--simulation-data", str(base)],
            input=payload,
            capture_output=True,
            env=env,
            cwd=str(Path(config.simulator_path).parent),
            timeout=config.timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"simulator exited {result.returncode}: {result.stderr.decode(errors='replace')[:400]}. "
                "An instant exit with empty stderr usually means the wrong MinGW runtime is on "
                "PATH - see the note in sim.py."
            )
        records: dict[int, Record] = {}
        for candidate in candidates:
            path = base.with_name(f"{base.name}-{candidate['id']}")
            if path.exists():
                records[candidate["id"]] = Record.load(path)
        yield records


def record_for(candidate: Candidate, config: SimConfig | None = None) -> Record:
    """One candidate, one record. Convenience for tests and single-machine work."""
    with records_for([candidate], config) as records:
        if candidate["id"] not in records:
            raise RuntimeError(
                f"no record produced for candidate id={candidate['id']} - the simulator "
                "refused it before logging began (a destroyed trigger does this)."
            )
        return records[candidate["id"]]
