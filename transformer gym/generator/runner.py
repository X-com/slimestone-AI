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


def simulate(candidate: Candidate, workdir: Path | None = None) -> bytes:
    """Runs `candidate` through the exe with --simulation-data and returns the raw .simlog bytes.
    Raises SimRunError if the exe isn't built or the run fails (candidate malformed, non-piston
    trigger, etc.) - callers (the gen_* modules) should catch this per-candidate and skip/retry,
    since a generator producing an occasional invalid candidate is expected, not fatal."""
    if not EXE.exists():
        raise SimRunError(f"exe not built: {EXE}")

    def _run(dir_path: Path) -> bytes:
        dat = dir_path / f"c{candidate['id']}.dat"
        dat.write_bytes(encode_candidate(candidate))
        base = dir_path / f"c{candidate['id']}.simlog"
        env = os.environ.copy()
        env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
        env["MCP1122_CPP_NO_Y_OFFSET"] = "1"
        result = subprocess.run(
            [str(EXE), str(dat), "--simulation-data", str(base)],
            env=env, capture_output=True, text=True,
        )
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
