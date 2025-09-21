"""Problem and Results types for center manifold Poincare maps.

This module defines the problem and results types for center manifold Poincare maps.
The problem type contains the parameters for the center manifold map computation,
while the results type contains the computed section.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from hiten.algorithms.poincare.core.types import _MapResults


@dataclass(frozen=True)
class _CenterManifoldMapProblem:
    """Immutable problem definition for a center manifold map run.

    Attributes
    ----------
    section_coord : str
        Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').
    energy : float
        Energy level (nondimensional units).
    dt : float
        Integration time step.
    n_iter : int
        Number of return-map iterations to compute.
    n_workers : int
        Number of parallel workers.
    """

    section_coord: str
    energy: float
    dt: float
    n_iter: int
    n_workers: int


class CenterManifoldMapResults(_MapResults):
    """User-facing results that behave as a _Section with extra helpers."""
    def __init__(self, points: np.ndarray, states: np.ndarray, labels: Tuple[str, str], times: Optional[np.ndarray] = None):
        super().__init__(points, states, labels, times)

    def project(self, axes: Tuple[str, str]) -> np.ndarray:
        idx1 = self.labels.index(axes[0])
        idx2 = self.labels.index(axes[1])
        return self.points[:, (idx1, idx2)]


