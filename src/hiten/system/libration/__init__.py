"""Convenience re-exports for libration-point classes.

This shortcut allows users to do for example::

>>> from hiten.system.libration import L1Point, L4Point
>>> L1 = L1Point(system)
>>> L4 = L4Point(system)
"""

from .base import LibrationPoint
from .collinear import CollinearPoint, L1Point, L2Point, L3Point
from .triangular import L4Point, L5Point, TriangularPoint

__all__ = [
    "LibrationPoint",
    "CollinearPoint",
    "TriangularPoint",
    "L1Point",
    "L2Point",
    "L3Point",
    "L4Point",
    "L5Point",
]
