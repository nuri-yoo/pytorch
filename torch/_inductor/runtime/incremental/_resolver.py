# mypy: allow-untyped-defs
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import queue
import threading

from ._utils import log

# Global event resolution daemon shared by all IncrementalAutotuneState
# instances. CUDA events must be resolved sequentially, so a single daemon
# thread processes events from all instances.
_global_event_queue: queue.Queue = queue.Queue()
_global_resolver_lock = threading.Lock()
_global_resolver_refcount = 0
# Unique sentinel per daemon generation — prevents a new daemon from
# consuming an old daemon's shutdown signal.
_global_resolver_sentinel: object | None = None
# The running resolver daemon thread, or None if no daemon is active.
# Set to None by the daemon itself (under _global_resolver_lock) before exiting.
_global_resolver_thread: threading.Thread | None = None


def _acquire_global_event_resolver() -> None:
    """Start the global event resolver daemon if needed."""
    global _global_resolver_refcount, _global_resolver_sentinel, _global_resolver_thread
    with _global_resolver_lock:
        _global_resolver_refcount += 1
        if _global_resolver_thread is not None:
            log.debug(
                "Incremental autotune event resolver already running (thread id=%d)",
                _global_resolver_thread.ident,
            )
            return
        _global_resolver_sentinel = object()
        t = threading.Thread(
            target=_global_event_resolver_loop,
            args=(_global_resolver_sentinel,),
            daemon=True,
            name="autotune-event-resolver",
        )
        t.start()
        _global_resolver_thread = t
        log.debug(
            "Incremental autotune event resolver started (thread id=%d, sentinel id=%d)",
            t.ident,
            id(_global_resolver_sentinel),
        )


def _release_global_event_resolver() -> None:
    """Shut down the daemon when no states remain."""
    global _global_resolver_refcount
    with _global_resolver_lock:
        _global_resolver_refcount -= 1
        if _global_resolver_refcount == 0 and _global_resolver_thread is not None:
            log.debug(
                "Incremental autotune event resolver shutting down"
                " (thread id=%d, sentinel id=%d)",
                _global_resolver_thread.ident,
                id(_global_resolver_sentinel),
            )
            _global_event_queue.put(_global_resolver_sentinel)


def _global_event_resolver_loop(sentinel: object) -> None:
    global _global_resolver_thread
    log.debug("Incremental autotune event resolver started")
    background_error = None

    def _set_background_error(state, error) -> None:
        with state._lock:
            if state._background_error is None:
                state._background_error = error
                log.debug(
                    "Incremental autotune: set background error on state id=%d: %s",
                    id(state),
                    error,
                )
            else:
                log.debug(
                    "Incremental autotune: background error already set on state id=%d, skipping",
                    id(state),
                )

    while True:
        item = _global_event_queue.get()
        if item is sentinel:
            with _global_resolver_lock:
                _global_resolver_thread = None
            log.debug(
                "Incremental autotune event resolver stopped (sentinel id=%d)",
                id(sentinel),
            )
            break
        state, launcher, start_event, end_event = item
        if background_error is not None:
            _set_background_error(state, background_error)
            continue
        try:
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            state._resolve_timing(launcher, elapsed_ms)
        except Exception as exc:
            log.debug(
                "Incremental autotune: exception resolving timing"
                " for state id=%d, launcher id=%d: %s",
                id(state),
                id(launcher),
                exc,
            )
            background_error = RuntimeError("Incremental autotune event resolver failed")
            background_error.__cause__ = exc
            _set_background_error(state, background_error)
