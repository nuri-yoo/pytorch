# mypy: allow-untyped-defs
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .config import _PERF_SAMPLE_COUNT


@dataclass
class LauncherStats:
    """Per-launcher timing statistics for incremental autotuning."""

    timings: list[float] = field(default_factory=list)
    filtered: bool = False  # eliminated: too slow, invalid config, or done timing
    dispatch_count: int = 0

    def add_timing(self, elapsed_ms: float) -> None:
        bisect.insort(self.timings, elapsed_ms)

    def mean(self) -> float:
        """Mean of the fastest _PERF_SAMPLE_COUNT runs (or all if fewer)."""
        if not self.timings:
            return float("inf")
        n = min(len(self.timings), _PERF_SAMPLE_COUNT)
        return sum(self.timings[:n]) / n

    @property
    def sample_count(self) -> int:
        return len(self.timings)
