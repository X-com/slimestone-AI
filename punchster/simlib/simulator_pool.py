from __future__ import annotations

import queue
import threading
from typing import Any

from .config import SimulatorRunConfig
from .failure_log import FailureLogger
from .simulator_process import SimulatorProcess


class SimulatorPool:
    def __init__(self, config: SimulatorRunConfig, failure_logger: FailureLogger | None = None) -> None:
        self.config = config.validated()
        self.failure_logger = failure_logger
        self.workers = [
            SimulatorProcess(self.config, worker_index=index)
            for index in range(self.config.worker_count)
        ]
        # crash_count/hang_count are always tracked (cheap, in-memory) so a caller can report
        # how many candidates crashed/hung the simulator even without failure_logger configured.
        # failure_logger, when given, additionally writes each one to disk as a repro case - see
        # failure_log.py. Both are updated from worker threads in _worker_loop, so
        # guarded by a lock.
        self.crash_count = 0
        self.hang_count = 0
        self._counts_lock = threading.Lock()

    def __enter__(self) -> "SimulatorPool":
        for worker in self.workers:
            worker.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for worker in self.workers:
            worker.close()

    def run_all(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        work_queue: queue.Queue[tuple[int, dict[str, Any]] | None] = queue.Queue()
        results: list[dict[str, Any] | None] = [None] * len(candidates)

        for index, candidate in enumerate(candidates):
            work_queue.put((index, candidate))
        for _ in self.workers:
            work_queue.put(None)

        threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(worker, work_queue, results),
                daemon=True,
            )
            for worker in self.workers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return [result for result in results if result is not None]

    def _worker_loop(
        self,
        worker: SimulatorProcess,
        work_queue: queue.Queue[tuple[int, dict[str, Any]] | None],
        results: list[dict[str, Any] | None],
    ) -> None:
        while True:
            item = work_queue.get()
            if item is None:
                return

            index, candidate = item
            try:
                results[index] = worker.simulate(candidate)
            except BaseException as exc:
                # A crash- or hang-inducing candidate (a real risk once a GA starts
                # mutating into territory the simulator doesn't handle yet) must not
                # take down the rest of the batch. Record it as a failed result and
                # restart the worker process so the queue keeps draining.
                kind = "hang" if isinstance(exc, TimeoutError) else "crash"
                results[index] = {
                    "id": candidate.get("id"),
                    "ok": False,
                    "burnin_ticks": 0,
                    "burnin_shift": {"x": 0, "y": 0, "z": 0},
                    "cycle_ticks": 0,
                    "cycle_shift": {"x": 0, "y": 0, "z": 0},
                    "timeout": False,
                    "errorCode": "worker_hung" if kind == "hang" else "worker_crashed",
                    "error": str(exc),
                }
                with self._counts_lock:
                    if kind == "hang":
                        self.hang_count += 1
                    else:
                        self.crash_count += 1
                if self.failure_logger is not None:
                    try:
                        path = self.failure_logger.log(kind, candidate)
                        print(f"[{kind}] candidate id={candidate.get('id')} -> {path}")
                    except OSError:
                        pass
                try:
                    worker.close()
                except Exception:
                    pass