# mypy: allow-untyped-defs
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

# Number of fastest samples used to compute a launcher's mean timing.
_PERF_SAMPLE_COUNT = 25

# Minimum number of timing samples a launcher must have before it can be
# filtered out based on the threshold.
_MIN_SAMPLES_BEFORE_FILTER = 3

# A launcher is filtered if its mean exceeds threshold * best_mean.
# The threshold starts at _INITIAL_THRESHOLD and decays as a concave function
# of samples collected (aggressive early, gradual later), reaching 1.0 at
# _MAX_SAMPLES_PER_LAUNCHER:
#   threshold = 1 + (_INITIAL_THRESHOLD - 1) * (1 - ((n-1) / (MAX-1)) ** _THRESHOLD_DECAY_EXP)
# At n=1 (first sample), threshold == _INITIAL_THRESHOLD; at n=MAX, threshold == 1.0.
# Smaller _THRESHOLD_DECAY_EXP = more aggressive early filtering.
_INITIAL_THRESHOLD = 2.5
_THRESHOLD_DECAY_EXP = 0.1

# Maximum number of timed dispatches per launcher before it is filtered out.
_MAX_SAMPLES_PER_LAUNCHER = 50

# Precomputed threshold scale factors for sample counts 1.._MAX_SAMPLES_PER_LAUNCHER.
# Index i corresponds to sample_count == i+1.  Avoids repeated pow() calls.
_THRESHOLD_MULTIPLIERS: tuple[float, ...] = tuple(
    1.0 - ((n - 1) / (_MAX_SAMPLES_PER_LAUNCHER - 1)) ** _THRESHOLD_DECAY_EXP
    for n in range(1, _MAX_SAMPLES_PER_LAUNCHER + 1)
)
