"""Vendored mem0 with smartcomment tracing.

Absolute imports inside this tree use ``from mem0.xxx``. Register this package as
the top-level ``mem0`` name before loading those modules so they resolve here,
not the PyPI ``mem0`` wheel.
"""

from __future__ import annotations

import sys

_pkg = sys.modules[__name__]
_prev = sys.modules.get("mem0")
if _prev is not None and _prev is not _pkg:
    for _name in list(sys.modules):
        if _name == "mem0" or _name.startswith("mem0."):
            del sys.modules[_name]
sys.modules["mem0"] = _pkg

__version__ = "vendored"

from .memory.main import AsyncMemory, Memory  # noqa
