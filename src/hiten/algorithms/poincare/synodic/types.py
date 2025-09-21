"""Types for synodic Poincare maps.

This module provides the types for synodic Poincare maps.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from hiten.algorithms.poincare.core.types import _MapResults


class SynodicMapResults(_MapResults):
    """User-facing results for synodic sections (extends 
    :class:`~hiten.algorithms.poincare.core.types._MapResults`).
    """
    pass


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
    """

    plane_coords: Tuple[str, str]
    direction: int | None
    n_workers: int
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]] | None = None


