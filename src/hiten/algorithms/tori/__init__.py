"""Invariant tori computation for the circular restricted three-body problem.

This module provides comprehensive tools for computing 2D invariant tori that
bifurcate from periodic orbits in the circular restricted three-body problem.
The implementation supports both linear approximation methods.
The module provides:
"""

from .base import _InvariantTori as InvariantTori
from .base import _ToriCorrectionConfig as ToriCorrectionConfig
from .base import _Torus as Torus

__all__ = [
    "InvariantTori",
    "ToriCorrectionConfig",
    "Torus",
]