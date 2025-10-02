"""Problem and Results types for center manifold Poincare maps.

This module defines the problem and results types for center manifold Poincare maps.
The problem type contains the parameters for the center manifold map computation,
while the results type contains the computed section.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Callable, List

import numpy as np
from hiten.algorithms.poincare.core.types import _MapResults
from hiten.algorithms.types.core import _DomainPayload


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
    jac_H: List[np.ndarray]
    H_blocks: List[np.ndarray]
    clmo_table: List[np.ndarray]
    solve_missing_coord_fn: Callable[[str, dict[str, float]], Optional[float]] | None = None
    find_turning_fn: Callable[[str], float] | None = None


class CenterManifoldMapResults(_MapResults):
    """User-facing results that behave as a _Section with extra helpers.
    """
    def __init__(self, points: np.ndarray, states: np.ndarray, labels: Tuple[str, str], times: Optional[np.ndarray] = None):
        super().__init__(points, states, labels, times)

    def project(self, axes: Tuple[str, str]) -> np.ndarray:
        idx1 = self.labels.index(axes[0])
        idx2 = self.labels.index(axes[1])
        return self.points[:, (idx1, idx2)]


@dataclass(frozen=True)
class CenterManifoldDomainPayload(_DomainPayload):
    """Domain payload describing center manifold map outputs."""

    @classmethod
    def _from_mapping(cls, data: Mapping[str, object]) -> "CenterManifoldDomainPayload":
        return cls(data=data)

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.require("points"), dtype=float)

    @property
    def states(self) -> np.ndarray:
        return np.asarray(self.require("states"), dtype=float)

    @property
    def times(self) -> Optional[np.ndarray]:
        times = self.get("times")
        return None if times is None else np.asarray(times, dtype=float)

    @property
    def labels(self) -> Tuple[str, str]:
        return tuple(self.require("labels"))

    @property
    def section_coord(self) -> Optional[str]:
        return self.get("section_coord")
