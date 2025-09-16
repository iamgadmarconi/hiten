"""Synodic Poincare maps for precomputed trajectories.

This module provides synodic Poincare map computation for precomputed trajectories,
enabling analysis of existing orbit data.
"""

from .backend import (_DetectionSettings, _SynodicDetectionBackend,
                     _project_batch, _compute_event_values, _is_vectorizable_plane_event,
                     _on_surface_indices, _crossing_indices_and_alpha, _refine_hits_linear,
                     _refine_hits_cubic, _order_and_dedup_hits, _detect_with_segment_refine)
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
    "_project_batch",
    "_compute_event_values",
    "_is_vectorizable_plane_event",
    "_on_surface_indices",
    "_crossing_indices_and_alpha",
    "_refine_hits_linear",
    "_refine_hits_cubic",
    "_order_and_dedup_hits",
    "_detect_with_segment_refine",
]
