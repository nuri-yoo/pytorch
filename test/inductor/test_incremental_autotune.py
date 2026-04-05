# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Owner(s): ["module: inductor"]

import threading
from unittest.mock import MagicMock, patch

from torch._inductor.runtime.incremental import IncrementalAutotuneState
from torch._inductor.runtime.incremental._stats import LauncherStats
from torch._inductor.runtime.incremental.config import _PERF_SAMPLE_COUNT
from torch._inductor.test_case import run_tests, TestCase


def _make_launcher(name="launcher"):
    launcher = MagicMock()
    launcher.config = f"config:{name}"
    return launcher


class LauncherStatsTest(TestCase):
    def test_mean_empty(self):
        stats = LauncherStats()
        self.assertEqual(stats.mean(), float("inf"))

    def test_sample_count(self):
        stats = LauncherStats()
        stats.add_timing(1.0)
        stats.add_timing(2.0)
        self.assertEqual(stats.sample_count, 2)

    def test_add_timing_sorted(self):
        stats = LauncherStats()
        stats.add_timing(3.0)
        stats.add_timing(1.0)
        stats.add_timing(2.0)
        self.assertEqual(stats.timings, [1.0, 2.0, 3.0])

    def test_mean_top_n(self):
        stats = LauncherStats()
        # Add _PERF_SAMPLE_COUNT + 5 timings; mean should use only the top N fastest.
        for i in range(_PERF_SAMPLE_COUNT + 5):
            stats.add_timing(float(i + 1))
        expected = sum(range(1, _PERF_SAMPLE_COUNT + 1)) / _PERF_SAMPLE_COUNT
        self.assertAlmostEqual(stats.mean(), expected)

    def test_mean_fewer_than_n(self):
        stats = LauncherStats()
        stats.add_timing(2.0)
        stats.add_timing(4.0)
        self.assertAlmostEqual(stats.mean(), 3.0)


class IncrementalAutotuneStateTest(TestCase):
    def test_add_launcher_and_next_launcher_order(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = IncrementalAutotuneState(launchers=[a, b])
        self.assertIs(state._next_launcher(), a)
        self.assertIs(state._next_launcher(), b)
        state.shutdown()

    def test_next_launcher_skips_filtered(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = IncrementalAutotuneState(launchers=[a, b])
        state._mark_filtered(a)
        self.assertIs(state._next_launcher(), b)
        state.shutdown()

    def test_next_launcher_all_filtered_raises(self):
        a = _make_launcher("a")
        state = IncrementalAutotuneState(launchers=[a])
        state._mark_filtered(a)
        with self.assertRaises(RuntimeError):
            state._next_launcher()
        state.shutdown()

    def test_mark_filtered_idempotent(self):
        launcher = _make_launcher()
        state = IncrementalAutotuneState(launchers=[launcher])
        state._mark_filtered(launcher)
        self.assertTrue(state._launcher_stats[id(launcher)].filtered)
        state._mark_filtered(launcher)  # should not raise or change anything
        self.assertTrue(state._launcher_stats[id(launcher)].filtered)
        state.shutdown()

    def test_converged_false_pending_events(self):
        launcher = _make_launcher()
        state = IncrementalAutotuneState(launchers=[launcher])
        state.best_launcher = launcher
        state._round_robin.clear()
        state._pending_events = 1
        self.assertFalse(state.converged)
        state.shutdown()

    def test_converged_false_nonempty_deque(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = IncrementalAutotuneState(launchers=[a, b])
        state.best_launcher = a
        self.assertFalse(state.converged)
        state.shutdown()

    def test_converged_true(self):
        launcher = _make_launcher()
        state = IncrementalAutotuneState(launchers=[launcher])
        state.best_launcher = launcher
        state._round_robin.clear()
        self.assertTrue(state.converged)
        state.shutdown()

    def test_resolve_timing_updates_best(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = IncrementalAutotuneState(launchers=[a, b])
        state._pending_events = 2
        state._launcher_stats[id(a)].add_timing(5.0)
        state.best_launcher = a

        state._resolve_timing(b, 2.0)

        self.assertIs(state.best_launcher, b)
        self.assertAlmostEqual(state.best_mean, 2.0)
        self.assertEqual(state._pending_events, 1)
        state.shutdown()

    def test_resolve_timing_does_not_demote_filtered(self):
        # A filtered launcher shouldn't become best even if it has the fastest time.
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = IncrementalAutotuneState(launchers=[a, b])
        state._pending_events = 1
        state._launcher_stats[id(a)].add_timing(5.0)
        state.best_launcher = a
        state._launcher_stats[id(b)].filtered = True

        state._resolve_timing(b, 1.0)

        self.assertIs(state.best_launcher, a)
        state.shutdown()

    def test_apply_threshold_filter_slow_launcher(self):
        # With _MIN_SAMPLES_BEFORE_FILTER=3, _INITIAL_THRESHOLD=2.5,
        # _THRESHOLD_DECAY_EXP=0.1, _MAX_SAMPLES_PER_LAUNCHER=50:
        # threshold after 3 samples = 1 + 1.5*(1 - (2/49)^0.1) ≈ 1.411.
        # slow mean 2.0ms vs best 1.0ms exceeds 1.411x — should be filtered.
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        state = IncrementalAutotuneState(launchers=[best, slow])
        state.best_launcher = best
        state._launcher_stats[id(best)].add_timing(1.0)
        for _ in range(3):
            state._launcher_stats[id(slow)].add_timing(2.0)
        with state._lock:
            state._apply_threshold_filter(slow)
        self.assertTrue(state._launcher_stats[id(slow)].filtered)
        state.shutdown()

    def test_apply_threshold_no_filter_below_threshold(self):
        # threshold after 3 samples ≈ 1.411; candidate mean 1.1ms < 1.411ms — not filtered.
        best = _make_launcher("best")
        candidate = _make_launcher("candidate")
        state = IncrementalAutotuneState(launchers=[best, candidate])
        state.best_launcher = best
        state._launcher_stats[id(best)].add_timing(1.0)
        for _ in range(3):
            state._launcher_stats[id(candidate)].add_timing(1.1)
        with state._lock:
            state._apply_threshold_filter(candidate)
        self.assertFalse(state._launcher_stats[id(candidate)].filtered)
        state.shutdown()

    def test_apply_threshold_not_enough_samples(self):
        # Only 1 sample, need _MIN_SAMPLES_BEFORE_FILTER=3 before filtering.
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        state = IncrementalAutotuneState(launchers=[best, slow])
        state.best_launcher = best
        state._launcher_stats[id(best)].add_timing(1.0)
        state._launcher_stats[id(slow)].add_timing(10.0)
        with state._lock:
            state._apply_threshold_filter(slow)
        self.assertFalse(state._launcher_stats[id(slow)].filtered)
        state.shutdown()

    def test_launchers_added_at_construction(self):
        launchers = [_make_launcher(f"l{i}") for i in range(3)]
        state = IncrementalAutotuneState(launchers=launchers)
        self.assertEqual(len(state._round_robin), 3)
        state.shutdown()

    def test_dispatch_round_robin_and_convergence(self):
        """dispatch() iterates launchers round-robin and calls on_convergence when done."""
        converged = threading.Event()
        converged_launcher = [None]

        def on_convergence(state):
            converged_launcher[0] = state.best_launcher
            converged.set()

        a = _make_launcher("a")
        b = _make_launcher("b")
        a.return_value = "result_a"
        b.return_value = "result_b"

        state = IncrementalAutotuneState(
            launchers=[a, b],
            on_convergence_fn=on_convergence,
        )

        mock_event = MagicMock()

        def fake_put(item):
            if not isinstance(item, tuple):
                return
            s, launcher, _start, _end = item
            # Simulate event resolver: resolve timing and decrement counter.
            s._resolve_timing(launcher, 1.0)

        with patch(
            "torch._inductor.runtime.incremental._state._MAX_SAMPLES_PER_LAUNCHER", 2
        ), patch(
            "torch._inductor.runtime.incremental._state.torch.cuda.Event",
            return_value=mock_event,
        ), patch(
            "torch._inductor.runtime.incremental._state._global_event_queue"
        ) as mock_queue:
            mock_queue.put.side_effect = fake_put
            # Dispatch enough times to exhaust both launchers (max_samples=2 each).
            for _ in range(4):
                state.dispatch(stream=0)
            # Next dispatch triggers convergence check -> on_convergence_fn.
            state.dispatch(stream=0)

        self.assertTrue(converged.is_set())
        self.assertIsNotNone(converged_launcher[0])

    def test_on_cleanup_called_on_del(self):
        cleanup_called = [False]

        def on_cleanup(state):
            cleanup_called[0] = True

        state = IncrementalAutotuneState(on_cleanup_fn=on_cleanup)
        state.__del__()
        self.assertTrue(cleanup_called[0])


if __name__ == "__main__":
    run_tests()
