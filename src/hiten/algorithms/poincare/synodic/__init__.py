"""Synodic Poincare maps for precomputed trajectories.

This module provides synodic Poincare map computation for precomputed trajectories,
enabling analysis of existing orbit data.
"""

from .base import SynodicMapPipeline
from .config import SynodicMapConfig
from .engine import _SynodicEngine
from .interfaces import _SynodicInterface
from .options import SynodicMapOptions
from .types import SynodicMapResults

__all__ = [
    # Main interface
    "SynodicMapPipeline",
    # Configuration (compile-time structure)
    "SynodicMapConfig",
    # Options (runtime tuning)
    "SynodicMapOptions",
    # Results
    "SynodicMapResults",
    # Engine and Interface
    "_SynodicEngine",
    "_SynodicInterface",
]
