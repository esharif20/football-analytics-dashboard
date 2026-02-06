"""
Compatibility shim: re-exports from the package __init__.py so that
``from pipeline import Mode`` works when ``src/`` is on sys.path.

The canonical definitions live in ``src/__init__.py``.
"""

from __init__ import Mode, get_frame_generator  # noqa: F401

__all__ = ["Mode", "get_frame_generator"]
