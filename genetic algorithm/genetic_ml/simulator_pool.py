from __future__ import annotations

import json
import queue
import threading
from typing import Any

from genetic_ml.config import SimulatorRunConfig
from genetic_ml.failure_log import FailureLogger
from genetic_ml.simulator_process import SimulatorProcess
from genetic_ml.three_way_compare import dump_cpp_trace_for_crash


class SimulatorPool:
    def __init__(self, config: SimulatorRunConfig, failure_logger: FailureLogger | None = None) -> None:
        self.config = config.validated()
        self.failure_logger = failure_logger
        self.workers = [
            SimulatorProcess(self.config, worker_index=index)
            for index in range(self.config.worker_count)
        ]

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
                args=(worker, work_queue, results, self.failure_logger),
                daemon=True,
            )
            for worker in self.workers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return [result for result in results if result is not None]

    @staticmethod
    def _worker_loop(
        worker: SimulatorProcess,
        work_queue: queue.Queue[tuple[int, dict[str, Any]] | None],
        results: list[dict[str, Any] | None],
        failure_logger: FailureLogger | None,
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
                    "working": False,
                    "errorCode": "worker_hung" if kind == "hang" else "worker_crashed",
                    "error": str(exc),
                }
                if failure_logger is not None:
                    try:
                        path = failure_logger.log(kind, candidate)
                        print(f"[{kind}] candidate id={candidate.get('id')} -> {path}")
                    except OSError:
                        path = None
                    if path is not None:
                        try:
                            candidate_json = json.dumps(candidate, separators=(",", ":"))
                            dump_cpp_trace_for_crash(candidate.get("id"), candidate_json, path.parent)
                        except Exception as trace_exc:
                            print(f"  crash-trace dump failed: {trace_exc}")
                try:
                    worker.close()
                except Exception:
                    pass

