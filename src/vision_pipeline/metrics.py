from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
from typing import Dict, List


@dataclass
class FrameMetrics:
    latencies_ms: List[float]
    frames_processed: int
    started_at: float
    ended_at: float

    def summary(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {
                "frames": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "effective_fps": 0.0,
            }

        sorted_lat = sorted(self.latencies_ms)
        p95_idx = max(int(len(sorted_lat) * 0.95) - 1, 0)
        elapsed = max(self.ended_at - self.started_at, 1e-6)

        return {
            "frames": float(self.frames_processed),
            "latency_p50_ms": float(median(sorted_lat)),
            "latency_p95_ms": float(sorted_lat[p95_idx]),
            "effective_fps": float(self.frames_processed / elapsed),
        }


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
