"""
Memory System - A LangGraph-based memory system with retrieval and storage layers.
"""
# EverMemOS uses absolute-style imports internally (e.g., `from memory_layer...`,
# `from api_specs...`), which assume the evermemos root is a top-level search path.
# Add it to sys.path so these internal imports resolve correctly.
import sys
from pathlib import Path

_EVERMEMOS_ROOT = str(Path(__file__).parent)
if _EVERMEMOS_ROOT not in sys.path:
    sys.path.insert(0, _EVERMEMOS_ROOT)

__version__ = "1.0.0"
__author__ = "Memory System Team"
