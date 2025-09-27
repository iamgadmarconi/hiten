"""Types for synodic Poincare maps.

This module provides the types for synodic Poincare maps.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from hiten.algorithms.poincare.core.types import _MapResults

if TYPE_CHECKING:
    from hiten.algorithms.poincare.synodic.config import _SynodicSectionConfig, _SynodicMapConfig


class SynodicMapResults(_MapResults):
    """User-facing results for synodic sections (extends 
    :class:`~hiten.algorithms.poincare.core.types._MapResults`).
    """
    
    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str], times: np.ndarray | None = None, trajectory_indices: np.ndarray | None = None):
        super().__init__(points, states, labels, times)
        self.trajectory_indices: np.ndarray | None = trajectory_indices


@dataclass(frozen=True)
class _SynodicMapProblem:
    """Problem definition for a synodic section run.

    Attributes
    ----------
    plane_coords : tuple[str, str]
        Labels of the plane projection axes.
    direction : {1, -1, None}
        Crossing direction filter.
    n_workers : int
        Parallel worker count to use in the engine.
    trajectories : Sequence[tuple[np.ndarray, np.ndarray]] | None
        Optional pre-bound trajectories.
    section_cfg : _SynodicSectionConfig
        Section configuration containing section parameters.
    map_cfg : _SynodicMapConfig
        Map configuration containing detection parameters.
    """

    plane_coords: Tuple[str, str]
    direction: Optional[int]
    n_workers: int
    section_cfg: "_SynodicSectionConfig"
    map_cfg: "_SynodicMapConfig"
    trajectories: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None


