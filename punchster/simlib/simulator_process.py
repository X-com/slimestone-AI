from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from .compact_format import encode_candidate
from .config import SimulatorRunConfig

# cpp_simulator_stream.exe is dynamically linked against the msys64/ucrt64 MinGW
# runtime (see ../cpp simulator/build-cpp.bat, which builds with
# C:\msys64\ucrt64\bin\g++.exe) and does not bundle those DLLs next to the exe. If
# whatever shell launches this Python process has a *different* MinGW runtime
# earlier on PATH - Git for Windows bundles its own at ...\Git\mingw64\bin, for
# example - Windows loads the wrong libstdc++-6.dll/libgcc_s_seh-1.dll and the
# process dies instantly with no stderr on every single candidate. That failure mode
# is indistinguishable from "nothing works" and depends entirely on which terminal
# happens to launch this (VS Code's integrated terminal, Code Runner, a plain
# shell...), so it's fixed here rather than left to the caller's environment.
_MINGW_RUNTIME_DIR = Path(r"C:\msys64\ucrt64\bin")


def _prepend_matching_runtime_dir(env: dict[str, str], runtime_dir: Path = _MINGW_RUNTIME_DIR) -> None:
    if runtime_dir.is_dir():
        env["PATH"] = f"{runtime_dir};{env.get('PATH', '')}"


class SimulatorProcess:
    def __init__(self, config: SimulatorRunConfig, worker_index: int) -> None:
        self.config = config.validated()
        self.worker_index = worker_index
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None:
            return

        command = [str(self.config.simulator_path)]
        if self.config.mode == "debug-piston-helper":
            command.append("--debug-piston-helper")
        elif self.config.mode == "debug-piston-move":
            command.append("--debug-piston-move")

        env = os.environ.copy()
        env["MCP1122_CPP_MAX_TICKS"] = str(self.config.max_ticks)
        _prepend_matching_runtime_dir(env)

        # Binary mode (no text=True/encoding) - candidates go to cpp's stdin as raw compact-
        # format bytes (see compact_format), not JSON, so this pipe cannot be text
        # mode. Results still come back as a JSON text line on stdout; _read_line() decodes
        # that line itself since text-mode line-splitting isn't available in binary mode.
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            cwd=str(Path(self.config.simulator_path).parent),
        )

    def simulate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self.process is not None
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        if self.process.poll() is not None:
            raise RuntimeError(self._dead_process_message())

        self.process.stdin.write(encode_candidate(candidate))
        self.process.stdin.flush()

        line = self._read_line(candidate)
        if line == b"":
            raise RuntimeError(self._dead_process_message())

        try:
            result = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Worker {self.worker_index} returned invalid JSON: {line!r}") from exc

        if not isinstance(result, dict):
            raise RuntimeError(f"Worker {self.worker_index} returned a non-object result: {line!r}")
        return result

    def _read_line(self, candidate: dict[str, Any]) -> bytes:
        """Read one response line, bounded by simulation_timeout_seconds. max_ticks
        only bounds the simulated tick loop - a malformed mutant can still hang the
        simulator in an unbounded block-update cascade before tick counting even
        starts, so a wall-clock timeout on the read itself is the only thing that
        catches that case."""

        assert self.process is not None
        assert self.process.stdout is not None

        timeout = self.config.simulation_timeout_seconds
        if timeout is None:
            return self.process.stdout.readline()

        line_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                line_queue.put(self.process.stdout.readline())  # type: ignore[union-attr]
            except (OSError, ValueError):
                line_queue.put(b"")

        threading.Thread(target=_reader, daemon=True).start()
        try:
            return line_queue.get(timeout=timeout)
        except queue.Empty:
            self._kill_hung_process()
            raise TimeoutError(
                f"Simulator worker {self.worker_index} did not respond within "
                f"{timeout}s for candidate id={candidate.get('id')!r}; killed as hung"
            ) from None

    def _kill_hung_process(self) -> None:
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
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2.0)
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
            f"Simulator worker {self.worker_index} exited with code "
            f"{self.process.returncode}. stderr={stderr!r}"
        )

    def __enter__(self) -> "SimulatorProcess":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()