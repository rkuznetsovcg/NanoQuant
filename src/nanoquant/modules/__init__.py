# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NanoQuant modules package."""

from .linear import NanoQuantLinear

__all__ = ["NanoQuantLinear", "NanoQuantModel", "NanoQuantConfigDataclass"]


def __getattr__(name):
    # hub imports the reconstruction code, which itself imports linear.
    # Resolve these public exports only when requested to avoid that cycle.
    if name in {"NanoQuantModel", "NanoQuantConfigDataclass"}:
        from . import hub
        return getattr(hub, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
