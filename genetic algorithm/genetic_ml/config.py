from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulatorRunConfig:
    simulator_path: Path
    worker_count: int = 1
    max_ticks: int = 6000
    mode: str = "simulate"
    startup_timeout_seconds: float = 5.0
    simulation_timeout_seconds: float | None = None

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
        )

