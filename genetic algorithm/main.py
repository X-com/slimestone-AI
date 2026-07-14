from __future__ import annotations

from pathlib import Path

from genetic_ml.candidate_io import load_candidates_from_glob
from genetic_ml.config import SimulatorRunConfig
from genetic_ml.result_store import write_dataset_jsonl
from genetic_ml.simulator_pool import SimulatorPool


# Edit these values in VS Code, then run this file.
PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_EXE = (
    PROJECT_ROOT.parent
    / "java-mcp simulator"
    / "cpp extract"
    / "build"
    / "mcp1122_cpp_stream.exe"
)
INPUT_GLOB = PROJECT_ROOT / "data" / "working" / "*.json"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "outputs" / "cpp_results.jsonl"

WORKER_COUNT = 4
MAX_TICKS = 6000
SIMULATOR_MODE = "simulate"  # simulate, debug-piston-helper, or debug-piston-move
INCLUDE_CANDIDATE_IN_OUTPUT = True


def main() -> None:
    config = SimulatorRunConfig(
        simulator_path=SIMULATOR_EXE,
        worker_count=WORKER_COUNT,
        max_ticks=MAX_TICKS,
        mode=SIMULATOR_MODE,
    )

    candidates = load_candidates_from_glob(INPUT_GLOB)
    print(f"Loaded {len(candidates)} candidates from {INPUT_GLOB}")
    print(f"Starting {config.worker_count} simulator worker(s): {config.simulator_path}")

    with SimulatorPool(config) as pool:
        results = pool.run_all(candidates)

    write_dataset_jsonl(
        OUTPUT_JSONL,
        candidates,
        results,
        simulator_path=config.simulator_path,
        include_candidate=INCLUDE_CANDIDATE_IN_OUTPUT,
    )

    ok = sum(1 for result in results if result.get("ok") is True)
    working = sum(1 for result in results if result.get("working") is True)
    print(f"Wrote {len(results)} results to {OUTPUT_JSONL}")
    print(f"ok={ok} working={working}")


if __name__ == "__main__":
    main()

