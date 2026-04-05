# mypy: allow-untyped-defs
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import functools

import torch._utils_internal
from torch._logging import getArtifactLogger

log = getArtifactLogger(__name__, "incremental")

# Bump this to enable incremental autotuning for users who have
# config.incremental_autotune = True (or the env var set to "1").
# The JK "pytorch/inductor:incremental_autotune_version" must be
# <= this value for the feature to activate.
_INCREMENTAL_AUTOTUNE_VERSION = 1


@functools.lru_cache(maxsize=None)
def jk_passes() -> bool:
    """Return True if the JK gate allows incremental autotuning.

    In OSS builds, justknobs_getval_int is unavailable and JK always passes.
    In fbcode, the JK acts as a killswitch: bump above _INCREMENTAL_AUTOTUNE_VERSION to disable.
    """
    try:
        val = torch._utils_internal.justknobs_getval_int(
            "pytorch/inductor:incremental_autotune_version"
        )
    except AttributeError:
        return True
    return _INCREMENTAL_AUTOTUNE_VERSION >= val
