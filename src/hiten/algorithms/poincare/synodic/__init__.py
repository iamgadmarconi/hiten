"""Synodic Poincare maps for precomputed trajectories.

This module provides synodic Poincare map computation for precomputed trajectories,
enabling analysis of existing orbit data.
"""

from .base import SynodicMap
from .config import _SynodicMapConfig

__all__ = [
    "SynodicMap",
    "_SynodicMapConfig",
]
