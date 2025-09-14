"""Single-hit Poincare section detection.

This module provides single-hit Poincare section detection for individual trajectories,
useful for finding specific section crossings.
"""

from .backend import _plane_crossing_factory, _SingleHitBackend, find_crossing

__all__ = [
    "_SingleHitBackend",
    "find_crossing",
    "_plane_crossing_factory",
]
