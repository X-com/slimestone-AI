"""Times the full flyers.data batch through 4 parallel cpp extract workers vs the GPU
kernel, and prints how they compare.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from genetic_ml.compact_format import encode_candidate, read_compact_file
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.simulator_pool import SimulatorPool

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
FLYERS_FILE = PROJECT_ROOT / "data" / "compact-working" / "flyers.data"

CPP_EXE = REPO_ROOT / "cpp simulator" / "build" / "cpp_simulator_stream.exe"
GPU_EXE = REPO_ROOT / "gpu simulator" / "build" / "gpu_kernel_stream.exe"

WORKER_COUNT = 4  # number of parallel cpp extract worker processes
MAX_TICKS = 6000


def bench_cpu_pool(candidates: list[dict]) -> float:
    config = SimulatorRunConfig(simulator_path=CPP_EXE, worker_count=WORKER_COUNT, max_ticks=MAX_TICKS)
    with SimulatorPool(config) as pool:
        started = time.perf_counter()
        results = pool.run_all(candidates)
        elapsed = time.perf_counter() - started
    ok = sum(1 for r in results if r.get("ok"))
    print(f"cpu pool ({WORKER_COUNT} worker processes): {len(results)} results, {ok} ok, "
          f"{elapsed:.2f}s, {len(results) / elapsed:.1f} candidates/sec")
    return elapsed


def bench_gpu_kernel(compact_bytes: bytes, expected_count: int) -> float:
    env = os.environ.copy()
    started = time.perf_counter()
    proc = subprocess.run(
        [str(GPU_EXE)], input=compact_bytes, capture_output=True, cwd=str(GPU_EXE.parent), env=env
    )
    elapsed = time.perf_counter() - started
    lines = [line for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) != expected_count:
        print(f"  warning: expected {expected_count} result lines, got {len(lines)} "
              f"(stderr: {proc.stderr.decode('utf-8', errors='replace')[:500]!r})")
    ok = sum(1 for line in lines if '"ok":true' in line)
    print(f"gpu kernel: {len(lines)} results, {ok} ok, {elapsed:.2f}s, {len(lines) / elapsed:.1f} candidates/sec")
    return elapsed


def main() -> None:
    if not CPP_EXE.exists():
        raise FileNotFoundError(f"cpp engine not built: {CPP_EXE}")
    if not GPU_EXE.exists():
        raise FileNotFoundError(f"GPU kernel not built: {GPU_EXE} (run build-cuda.bat first)")

    candidates = read_compact_file(FLYERS_FILE)
    if not candidates:
        raise RuntimeError(f"no candidates found in {FLYERS_FILE}")
    print(f"loaded {len(candidates)} candidates from {FLYERS_FILE}")

    compact_bytes = b"".join(encode_candidate(c) for c in candidates)

    cpu_elapsed = bench_cpu_pool(candidates)
    gpu_elapsed = bench_gpu_kernel(compact_bytes, len(candidates))

    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float("inf")
    print(f"cpu pool: {cpu_elapsed:.2f}s | gpu kernel: {gpu_elapsed:.2f}s | speedup (cpu/gpu): {speedup:.2f}x")


if __name__ == "__main__":
    main()
