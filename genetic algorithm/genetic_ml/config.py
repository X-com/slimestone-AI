from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


def simulator_exe_name(base: str = "cpp_simulator_stream") -> str:
    """Platform-correct simulator executable filename - build-cpp.sh/build-cpp.bat both name it
    `base` on Linux/macOS and `base.exe` on Windows; every main*.py's SIMULATOR_EXE should build
    its path through this instead of hardcoding the extension."""
    return base + ".exe" if platform.system() == "Windows" else base


@dataclass(frozen=True)
class SimulatorRunConfig:
    simulator_path: Path
    worker_count: int = 1
    max_ticks: int = 6000
    mode: str = "simulate"
    startup_timeout_seconds: float = 5.0
    simulation_timeout_seconds: float | None = None
    # Selects the C++ simulator's cycle-verification method (see cpp simulator/src/simulator.cpp
    # Simulator::simulate). True (default): structural (piston-usage) verification - trigger must
    # be an observer or an already-powered piston (an unpowered piston trigger fails instantly, no
    # cold start), and validity is decided by whether every piston completes exactly one
    # extend+retract without a spent piston ever pushing a block that had never moved before. All
    # flying-machine structures are designed around observer triggering, so this is the correct
    # default. False: burnout + settle + compareFinalToInitial, trigger can be a cold piston
    # (artificially charged) or an already-powered one.
    structural_verify: bool = True

    def validated(self) -> "SimulatorRunConfig":
        simulator_path = Path(self.simulator_path)
        if not simulator_path.exists():
            raise FileNotFoundError(f"Simulator executable does not exist: {simulator_path}")
        if self.worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if self.max_ticks < 1:
            raise ValueError("max_ticks must be at least 1")
        if self.mode not in {"simulate", "debug-piston-helper", "debug-piston-move"}:
            raise ValueError(f"Unsupported simulator mode: {self.mode}")
        return SimulatorRunConfig(
            simulator_path=simulator_path,
            worker_count=self.worker_count,
            max_ticks=self.max_ticks,
            mode=self.mode,
            startup_timeout_seconds=self.startup_timeout_seconds,
            simulation_timeout_seconds=self.simulation_timeout_seconds,
            structural_verify=self.structural_verify,
        )

