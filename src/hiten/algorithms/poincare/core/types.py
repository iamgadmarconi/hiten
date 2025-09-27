"""Types for Poincare return map computation.

This module provides the types for Poincare return map computation.
"""

from typing import NamedTuple

import numpy as np


class _Section:
    """Immutable container for a single 2D return map slice.

    This class holds the data for a computed Poincare section, including
    the intersection points, state vectors, axis labels, and optional
    integration times.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Array of 2D intersection points in the section plane.
    states : ndarray, shape (n, k)
        Array of state vectors at the intersection points. The number
        of columns k depends on the backend implementation.
    labels : tuple[str, str]
        Labels for the two axes in the section plane (e.g., ("q2", "p2")).
    times : ndarray, optional, shape (n,)
        Array of absolute integration times at each intersection point.
        If None, times are not available.
    """

    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str], times: np.ndarray | None = None):
        self.points: np.ndarray = points       # (n, 2) plane coordinates
        self.states: np.ndarray = states       # (n, k) backend-specific state vectors
        self.labels: tuple[str, str] = labels  # axis labels (e.g. ("q2", "p2"))
        self.times: np.ndarray | None = times  # (n,) absolute integration times (optional)

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self):
        return f"_Section(points={len(self)}, labels={self.labels}, times={'yes' if self.times is not None else 'no'})"


class _SectionHit(NamedTuple):
    """Container for a single trajectory-section intersection.

    This named tuple holds all the information about a single
    intersection between a trajectory and a Poincare section.
    It provides both the full state vector and the 2D projection
    for efficient access.

    Parameters
    ----------
    time : float
        Absolute integration time (nondimensional units), signed
        according to propagation direction.
    state : ndarray, shape (n,)
        Full state vector at the crossing (immutable copy).
    point2d : ndarray, shape (2,)
        2D coordinates of the point in the section plane (e.g.,
        (q2, p2) or (x, x_dot)). Stored separately so callers
        do not have to re-project the full state vector.
    trajectory_index : int
        Index of the trajectory that produced this intersection
        within the input trajectory list.

    Notes
    -----
    This container is immutable and provides efficient access to
    both the full state information and the 2D projection. The
    2D coordinates are pre-computed to avoid repeated projection
    operations.

    All time units are in nondimensional units unless otherwise
    specified.
    """

    time: float
    state: np.ndarray  # shape (n,)
    point2d: np.ndarray  # shape (2,)
    trajectory_index: int

    def __repr__(self):
        return (f"SectionHit(t={self.time:.3e}, state={np.array2string(self.state, precision=3)}, "
                f"pt={np.array2string(self.point2d, precision=3)})")


class _MapResults(_Section):
    """Base results object for Poincare maps.

    Acts like `_Section` while serving as a common base for module-specific
    results (e.g., `CenterManifoldMapResults`, `SynodicMapResults`).
    """

    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str], times: np.ndarray | None = None):
        super().__init__(points, states, labels, times)
