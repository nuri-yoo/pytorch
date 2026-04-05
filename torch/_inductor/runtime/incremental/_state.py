# mypy: allow-untyped-defs
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import torch

from .config import (
    _INITIAL_THRESHOLD,
    _MAX_SAMPLES_PER_LAUNCHER,
    _MIN_SAMPLES_BEFORE_FILTER,
    _THRESHOLD_MULTIPLIERS,
)
from ._resolver import (
    _acquire_global_event_resolver,
    _global_event_queue,
    _release_global_event_resolver,
)
from ._stats import LauncherStats
from ._utils import log

if TYPE_CHECKING:
    from typing import Any, Callable

    LauncherType = Any


class IncrementalAutotuneState:
    """Central state for incremental autotuning within a CachingAutotuner instance.

    Manages a round-robin of launcher candidates, CUDA event timing
    via a daemon thread, progressive filtering, and convergence detection.

    Thread safety: the event resolver daemon calls _resolve_timing() from its
    thread, while the main thread drives dispatch().  A single lock serializes
    access to shared mutable state.
    """

    def __init__(
        self,
        launchers: list = (),
        pre_launch_fn: Callable | None = None,
        post_launch_fn: Callable[[], None] | None = None,
        on_convergence_fn: Callable[[IncrementalAutotuneState], None] | None = None,
        on_cleanup_fn: Callable[[IncrementalAutotuneState], None] | None = None,
    ) -> None:
        self._launcher_stats: dict[int, LauncherStats] = {}
        self._round_robin: deque[LauncherType] = deque()
        self.best_launcher: LauncherType | None = None

        self._lock = threading.RLock()
        self._pre_launch_fn = pre_launch_fn
        self._post_launch_fn = post_launch_fn
        self._on_convergence_fn = on_convergence_fn
        self._on_cleanup_fn = on_cleanup_fn

        self._pending_events: int = 0

        # Set by the event resolver daemon if it encounters an error.
        # Re-raised on the main thread at the next dispatch() call.
        self._background_error: Exception | None = None

        # Accumulated wall-clock time of all dispatch() calls before
        # convergence. Used to report autotuning overhead.
        self._total_dispatch_ns: int = 0

        self._resolver_released = False
        _acquire_global_event_resolver()
        for launcher in launchers:
            self._add_launcher(launcher)

    # -- Round-robin ----------------------------------------------------------

    def _add_launcher(self, launcher: LauncherType) -> None:
        launcher_id = id(launcher)
        assert launcher_id not in self._launcher_stats
        self._launcher_stats[launcher_id] = LauncherStats()
        self._round_robin.append(launcher)
        log.debug("Incremental autotune: state id=%d queued launcher id=%d", id(self), id(launcher))

    def _launcher_is_active(self, launcher: LauncherType) -> bool:
        """Return True if launcher has not been filtered. Lock must be held."""
        return not self._launcher_stats[id(launcher)].filtered

    def _next_launcher(self) -> LauncherType:
        """Pop and return the next active launcher. Lock must be held."""
        while self._round_robin:
            launcher = self._round_robin.popleft()
            if self._launcher_is_active(launcher):
                return launcher
            log.debug(
                "Incremental autotune: state id=%d skipping filtered launcher id=%d", id(self), id(launcher)
            )
        raise RuntimeError(
            "No active launchers available for incremental autotune"
        )

    def _mark_filtered(self, launcher: LauncherType) -> None:
        """Mark launcher as filtered. Lock must be held."""
        log.debug("Incremental autotune: state id=%d marking launcher id=%d as filtered", id(self), id(launcher))
        self._launcher_stats[id(launcher)].filtered = True

    # -- Dispatch ----------------------------------------------------------

    def dispatch(self, *args, stream, **kwargs):
        """Dispatch a kernel launch with incremental autotuning.

        On convergence, calls on_convergence_fn which may either finalize
        (real convergence) or submit more configs to continue tuning.
        """
        _t0 = time.time_ns()
        result = self._dispatch(*args, stream=stream, **kwargs)
        self._total_dispatch_ns += time.time_ns() - _t0
        return result

    def _dispatch(self, *args, stream, **kwargs):
        with self._lock:
            if self._background_error is not None:
                raise self._background_error

            if self.converged:
                log.debug("Incremental autotune: state id=%d converged", id(self))
                if self._on_convergence_fn is not None:
                    self._on_convergence_fn(self)
                # Check again: on_convergence_fn may have added more launchers.
                if self.converged:
                    log.debug(
                        "Incremental autotune: state id=%d still converged after"
                        " on_convergence_fn, shutting down",
                        id(self),
                    )
                    self.shutdown()
                return self._launch(self.best_launcher, *args, stream=stream, **kwargs)

            while True:
                try:
                    launcher = self._next_launcher()
                except RuntimeError:
                    if self.best_launcher is not None:
                        log.debug(
                            "Incremental autotune: state id=%d all launchers exhausted,"
                            " falling back to best launcher id=%d",
                            id(self),
                            id(self.best_launcher),
                        )
                        return self._launch(
                            self.best_launcher, *args, stream=stream, **kwargs
                        )
                    raise

                try:
                    result = self._launch(launcher, *args, stream=stream, **kwargs)
                except Exception as e:
                    if "invalid configuration" not in str(e).lower():
                        raise
                    log.debug(
                        "Incremental autotune: state id=%d launcher id=%d"
                        " filtered — invalid configuration at launch",
                        id(self),
                        id(launcher),
                    )
                    self._mark_filtered(launcher)
                    continue

                if self.best_launcher is None:
                    self.best_launcher = launcher
                    log.debug(
                        "Incremental autotune: state id=%d initial best launcher id=%d",
                        id(self),
                        id(launcher),
                    )

                return result

    def _launch(self, launcher, *args, stream, **kwargs):
        """Launch a kernel with pre/post-launch hooks and optional CUDA event timing."""
        timed = self._launcher_is_active(launcher)
        log.debug(
            "Incremental autotune: state id=%d launching launcher id=%d timed=%s",
            id(self),
            id(launcher),
            timed,
        )
        try:
            if self._pre_launch_fn is not None:
                self._pre_launch_fn(launcher, *args, stream=stream, **kwargs)
            if timed:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            result = launcher(*args, **kwargs, stream=stream)
            if timed:
                end_event.record()
        finally:
            if self._post_launch_fn is not None:
                self._post_launch_fn()
        if timed:
            stats = self._launcher_stats[id(launcher)]
            stats.dispatch_count += 1
            if stats.dispatch_count >= _MAX_SAMPLES_PER_LAUNCHER:
                log.debug(
                    "Incremental autotune: state id=%d launcher id=%d"
                    " reached max samples (%d), filtering",
                    id(self),
                    id(launcher),
                    _MAX_SAMPLES_PER_LAUNCHER,
                )
                self._mark_filtered(launcher)
            else:
                self._round_robin.append(launcher)
            self._pending_events += 1
            _global_event_queue.put((self, launcher, start_event, end_event))
        return result

    # -- Event resolution (handled by global daemon) -----------------------

    @property
    def best_mean(self) -> float:
        if self.best_launcher is None:
            return float("inf")
        return self._launcher_stats[id(self.best_launcher)].mean()

    def _resolve_timing(self, launcher: LauncherType, elapsed_ms: float) -> None:
        """Called by the global event resolver daemon after synchronizing a CUDA event."""
        with self._lock:
            stats = self._launcher_stats[id(launcher)]
            stats.add_timing(elapsed_ms)
            mean = stats.mean()
            log.debug(
                "Incremental autotune: state id=%d launcher id=%d"
                " sample %d = %.3f ms (mean=%.3f ms)",
                id(self),
                id(launcher),
                stats.sample_count,
                elapsed_ms,
                mean,
            )
            if not stats.filtered and mean < self.best_mean:
                prev_launcher_id = id(self.best_launcher) if self.best_launcher is not None else 0
                prev_mean = self.best_mean
                self.best_launcher = launcher
                log.debug(
                    "Incremental autotune: state id=%d new best launcher id=%d"
                    " (mean=%.3f ms) replacing launcher id=%d (mean=%.3f ms)",
                    id(self),
                    id(launcher),
                    mean,
                    prev_launcher_id,
                    prev_mean,
                )
            self._apply_threshold_filter(launcher)
            self._pending_events -= 1

    def _apply_threshold_filter(self, launcher: LauncherType) -> None:
        """Filter launcher if its mean exceeds threshold * best_mean. Lock must be held."""
        if self.best_launcher is None or launcher is self.best_launcher:
            return
        stats = self._launcher_stats[id(launcher)]
        if stats.filtered or stats.sample_count < _MIN_SAMPLES_BEFORE_FILTER:
            return
        launcher_mean = stats.mean()
        # Threshold decays based on how many timings this launcher has,
        # so launchers with more data are filtered more aggressively.
        threshold = 1.0 + (_INITIAL_THRESHOLD - 1.0) * _THRESHOLD_MULTIPLIERS[stats.sample_count - 1]
        if launcher_mean > threshold * self.best_mean:
            log.debug(
                "Incremental autotune: state id=%d launcher id=%d"
                " filtered after %d samples"
                " (mean=%.3f ms > threshold %.3fx * best %.3f ms)",
                id(self),
                id(launcher),
                stats.sample_count,
                launcher_mean,
                threshold,
                self.best_mean,
            )
            self._mark_filtered(launcher)

    # -- Convergence -------------------------------------------------------

    @property
    def converged(self) -> bool:
        if self._pending_events > 0:
            return False
        if len(self._round_robin) == 0:
            assert self.best_launcher is not None, (
                "deque empty with no best launcher — all configs were rejected at launch"
            )
            return True
        return False

    # -- Cleanup -----------------------------------------------------------

    def _release_resolver(self) -> None:
        if not self._resolver_released:
            self._resolver_released = True
            _release_global_event_resolver()

    def __del__(self) -> None:
        try:
            if self._on_cleanup_fn is not None:
                self._on_cleanup_fn(self)
                self._on_cleanup_fn = None
        except Exception:
            pass
        finally:
            self._release_resolver()

    def shutdown(self) -> None:
        self._release_resolver()
