"""Driving the C++ simulator - md/ALPHAZERO.md Part 8, DECISIONS.md point 25.

One long-lived process reads compact-binary candidates from stdin and writes one JSON verdict
line per candidate to stdout (main.cpp:57 processStream). This module owns that protocol.

The verdict already carries everything Milestone 1 needs, so no sim-log decoding is involved:

    validCycle              the reward
    period, shift           the cheap half of the cargo detector
    elapsedNs               per-candidate timing, which measures the ~1ms assumption directly

Deadlock note: stdin and stdout are both pipes with finite buffers, so writing a whole batch
before reading anything will hang once stdout fills. A reader thread drains stdout continuously
while the main thread writes, which is why batching is safe here at any size.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rlgym.game import Candidate, encode_candidate

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATOR = _REPO_ROOT / "cpp simulator" / "build" / "cpp_simulator_stream.exe"

# cpp_simulator_stream.exe is dynamically linked against the msys64/ucrt64 MinGW runtime and
# does not bundle those DLLs next to the exe. If the launching shell has a DIFFERENT MinGW
# runtime earlier on PATH - Git for Windows ships one at ...\Git\mingw64\bin - Windows loads
# the wrong libstdc++-6.dll and the process dies instantly, with no stderr, on every single
# candidate. That failure mode is indistinguishable from "nothing works" and depends entirely
# on which terminal launched Python, so it is fixed here rather than left to the environment.
# Inherited from genetic_ml/simulator_process.py; do not drop it.
_MINGW_RUNTIME_DIR = Path(r"C:\msys64\ucrt64\bin")


def _prepend_matching_runtime_dir(env: dict[str, str], runtime_dir: Path = _MINGW_RUNTIME_DIR) -> None:
    if runtime_dir.is_dir():
        env["PATH"] = f"{runtime_dir};{env.get('PATH', '')}"


@dataclass(frozen=True)
class SimConfig:
    simulator_path: Path = DEFAULT_SIMULATOR
    max_ticks: int = 6000
    # Structural (piston-usage) verification, the C++ default: the trigger must be an observer
    # or an already-powered piston, and validity is decided by whether every piston completes
    # exactly one extend+retract. Flying machines are designed around observer triggering, so
    # this is the correct mode - see genetic_ml/config.py for the alternative.
    structural_verify: bool = True
    timeout_seconds: float = 60.0

    def validated(self) -> "SimConfig":
        if not Path(self.simulator_path).exists():
            raise FileNotFoundError(
                f"Simulator executable does not exist: {self.simulator_path}. "
                "Build it with cpp simulator/build-cpp.bat."
            )
        return self


class SimulatorProcess:
    """One long-lived simulator process. Not thread-safe; use one per thread."""

    def __init__(self, config: SimConfig | None = None, worker_index: int = 0) -> None:
        self.config = (config or SimConfig()).validated()
        self.worker_index = worker_index
        self.process: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes] | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        env = os.environ.copy()
        env["MCP1122_CPP_MAX_TICKS"] = str(self.config.max_ticks)
        env["MCP1122_CPP_STRUCTURAL_VERIFY"] = "1" if self.config.structural_verify else "0"
        _prepend_matching_runtime_dir(env)

        # Binary mode throughout: candidates are raw compact-format bytes, so this pipe cannot
        # be text mode. Verdicts come back as JSON text lines which we decode ourselves.
        self.process = subprocess.Popen(
            [str(self.config.simulator_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            cwd=str(Path(self.config.simulator_path).parent),
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()

    def _drain_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._lines is not None
        try:
            for line in self.process.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(b"")  # sentinel: stream closed

    def simulate_batch(self, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
        """Send every candidate, then collect exactly that many verdicts, in order.

        Order is guaranteed by the C++ side: processStream is a strict read-one/print-one loop,
        so the Nth line answers the Nth candidate. The verdict also carries the id we sent,
        which is checked below - a mismatch means the stream desynchronised and every label
        after it would be silently attached to the wrong machine.
        """
        if not candidates:
            return []
        self.start()
        assert self.process is not None and self.process.stdin is not None
        assert self._lines is not None

        if self.process.poll() is not None:
            raise RuntimeError(self._dead_process_message())

        payload = b"".join(encode_candidate(c) for c in candidates)

        def _write() -> None:
            try:
                self.process.stdin.write(payload)  # type: ignore[union-attr]
                self.process.stdin.flush()  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass

        writer = threading.Thread(target=_write, daemon=True)
        writer.start()

        results: list[dict[str, Any]] = []
        for index in range(len(candidates)):
            try:
                line = self._lines.get(timeout=self.config.timeout_seconds)
            except queue.Empty:
                self._kill()
                raise TimeoutError(
                    f"Worker {self.worker_index} produced {index} of {len(candidates)} verdicts "
                    f"within {self.config.timeout_seconds}s; killed as hung"
                ) from None
            if line == b"":
                raise RuntimeError(self._dead_process_message())
            try:
                result = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Worker {self.worker_index} returned invalid JSON: {line!r}"
                ) from exc
            expected = candidates[index]["id"]
            if result.get("id") != expected:
                raise RuntimeError(
                    f"Verdict stream desynchronised at position {index}: expected id "
                    f"{expected}, got {result.get('id')}. Every later label would be attached "
                    "to the wrong machine."
                )
            results.append(result)
        writer.join(timeout=1.0)
        return results

    def simulate(self, candidate: Candidate) -> dict[str, Any]:
        return self.simulate_batch([candidate])[0]

    def write_raw(self, payload: bytes) -> None:
        """Push arbitrary bytes at the simulator. For protocol tests only."""
        self.start()
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(payload)
        self.process.stdin.flush()

    def read_raw_line(self, timeout: float | None = None) -> bytes:
        """One verdict line straight off the drain queue.

        Never read self.process.stdout directly once start() has run: the drain thread is
        already reading it, and two readers on one pipe interleave character by character,
        producing output that looks corrupted rather than raced.
        """
        self.start()
        assert self._lines is not None
        try:
            return self._lines.get(timeout=timeout or self.config.timeout_seconds)
        except queue.Empty:
            return b""

    def close_stdin(self) -> None:
        """Signal end-of-stream without tearing the process down, so the final verdicts can
        still be read. For protocol tests only."""
        if self.process is not None and self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass

    def _kill(self) -> None:
        if self.process is None:
            return
        try:
            self.process.kill()
            self.process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            self.process = None

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)
        finally:
            self.process = None

    def _dead_process_message(self) -> str:
        assert self.process is not None
        stderr = ""
        if self.process.stderr is not None:
            try:
                stderr = self.process.stderr.read().decode("utf-8", errors="replace")
            except OSError:
                stderr = ""
        return (
            f"Simulator worker {self.worker_index} exited with code {self.process.returncode}. "
            f"stderr={stderr!r}. An instant exit with empty stderr usually means the wrong "
            f"MinGW runtime is on PATH - see the note at the top of this file."
        )

    def __enter__(self) -> "SimulatorProcess":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def simulate_all(
    candidates: Sequence[Candidate],
    workers: int = 12,
    max_chunk: int = 512,
    config: SimConfig | None = None,
) -> list[dict[str, Any]]:
    """Run every candidate across a pool of processes, preserving input order.

    Threads are enough despite the GIL: the work happens in separate OS processes and the
    Python side only blocks on pipe I/O, which releases the lock.

    Chunk size is derived from the workload rather than fixed, because a fixed chunk silently
    caps parallelism on small runs - 988 candidates at a fixed 512 is two chunks, so ten of
    twelve workers would sit idle and the measured throughput would be a sixth of the truth.
    """
    if not candidates:
        return []
    workers = max(1, min(workers, len(candidates)))
    chunk = max(1, min(max_chunk, (len(candidates) + workers - 1) // workers))
    chunks = [candidates[i : i + chunk] for i in range(0, len(candidates), chunk)]

    local = threading.local()

    def _run(batch: Sequence[Candidate]) -> list[dict[str, Any]]:
        proc = getattr(local, "proc", None)
        if proc is None:
            proc = SimulatorProcess(config, worker_index=threading.get_ident() % 1000)
            local.proc = proc
        return proc.simulate_batch(batch)

    procs: list[SimulatorProcess] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run, chunks))
    finally:
        for proc in procs:
            proc.close()
    return [item for part in parts for item in part]


def iter_verdicts(results: Iterable[dict[str, Any]]) -> Iterable[tuple[int, float]]:
    """(id, reward) pairs. Reward is a float from the start - see game.py."""
    for result in results:
        yield result["id"], 1.0 if result.get("validCycle") else 0.0
