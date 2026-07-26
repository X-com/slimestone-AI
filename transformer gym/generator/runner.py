"""Runs one in-memory Candidate dict through the C++ simulator and returns the decoded
.simlog bytes. Same exe/env/output-naming convention as verify_simulation_data.run_fixture,
just starting from a candidate dict instead of a named JSON fixture on disk - generators build
candidates in memory, so there's no fixture file to look one up by name.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import generator  # noqa: F401  (sys.path shim, must run before the imports below)
from genetic_ml.compact_format import encode_candidate
from verify_simulation_data import EXE, MSYS_BIN

Candidate = dict[str, Any]


class SimRunError(RuntimeError):
    pass


def simulate(candidate: Candidate, workdir: Path | None = None, timeout: float = 30.0) -> bytes:
    """Runs `candidate` through the exe with --simulation-data and returns the raw .simlog bytes.
    Raises SimRunError if the exe isn't built, the run fails (candidate malformed, non-piston
    trigger, etc.), or it exceeds `timeout` seconds - callers (the gen_* modules, corpus.py)
    should catch this per-candidate and skip/retry, since a generator producing an occasional
    invalid or pathologically slow candidate is expected, not fatal. The timeout matters more for
    generators that can produce large/dense boards (e.g. wave_function/) than for the small
    constructive ones, but applies uniformly since any generator could hit a rare slow case."""
    if not EXE.exists():
        raise SimRunError(f"exe not built: {EXE}")

    def _run(dir_path: Path) -> bytes:
        dat = dir_path / f"c{candidate['id']}.dat"
        dat.write_bytes(encode_candidate(candidate))
        base = dir_path / f"c{candidate['id']}.simlog"
        env = os.environ.copy()
        env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
        env["MCP1122_CPP_NO_Y_OFFSET"] = "1"
        try:
            result = subprocess.run(
                [str(EXE), str(dat), "--simulation-data", str(base)],
                env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise SimRunError(f"candidate {candidate['id']}: timed out after {timeout}s") from e
        if result.returncode != 0:
            raise SimRunError(f"candidate {candidate['id']}: {result.stderr.strip()[:500]}")
        out = dir_path / f"c{candidate['id']}-{candidate['id']}.simlog"
        if not out.exists():
            raise SimRunError(f"candidate {candidate['id']}: expected log not produced: {out}")
        return out.read_bytes()

    if workdir is not None:
        return _run(workdir)
    with tempfile.TemporaryDirectory() as tmp:
        return _run(Path(tmp))
