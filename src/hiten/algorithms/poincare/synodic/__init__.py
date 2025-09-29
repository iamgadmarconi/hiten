"""Synodic Poincare maps for precomputed trajectories.

This module provides synodic Poincare map computation for precomputed trajectories,
enabling analysis of existing orbit data.
"""

from .base import _SynodicMapFacade
from .config import _SynodicMapConfig
from .engine import _SynodicEngine
from .interfaces import _SynodicInterface
from .types import SynodicMapResults

__all__ = [
    "_SynodicMapConfig",
    "SynodicMapResults",
    "_SynodicMapFacade",
    "_SynodicEngine",
    "_SynodicInterface",
]
