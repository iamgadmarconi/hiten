"""Synodic Poincare maps for precomputed trajectories.

This module provides synodic Poincare map computation for precomputed trajectories,
enabling analysis of existing orbit data.
"""

from .backend import _DetectionSettings, _SynodicDetectionBackend
from .base import SynodicMap
from .config import (_get_section_config, _SynodicMapConfig,
                     _SynodicSectionConfig)
from .engine import _SynodicEngine, _SynodicEngineConfigAdapter
from .events import _AffinePlaneEvent
from .strategies import _NoOpStrategy

__all__ = [
    "SynodicMap",
    "_SynodicMapConfig",
    "_SynodicSectionConfig",
    "_SynodicDetectionBackend",
    "_SynodicEngine",
    "_SynodicEngineConfigAdapter",
    "_AffinePlaneEvent",
    "_NoOpStrategy",
    "_DetectionSettings",
    "_get_section_config",
]
