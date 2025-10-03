"""Types for synodic Poincare maps.

This module provides the types for synodic Poincare maps.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from hiten.algorithms.poincare.core.types import _MapResults, _SectionHit
from hiten.algorithms.types.core import _DomainPayload


@dataclass
class SynodicBackendRequest:
    """Structured request for the synodic detection backend."""

    trajectories: Sequence[tuple[np.ndarray, np.ndarray]]
    normal: np.ndarray | Sequence[float]
    trajectory_indices: Sequence[int]
    offset: float = 0.0
    plane_coords: Tuple[str, str] = ("y", "vy")
    interp_kind: Literal["linear", "cubic"] = "linear"
    segment_refine: int = 0
    tol_on_surface: float = 1e-12
    dedup_time_tol: float = 1e-9
    dedup_point_tol: float = 1e-12
    max_hits_per_traj: int | None = None
    newton_max_iter: int = 4
    direction: Literal[1, -1, None] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynodicBackendResponse:
    """Structured response returned by the synodic detection backend.
    
    Attributes
    ----------
    hits : list[list[_SectionHit]]
        Raw section hits from backend detection.
    points : np.ndarray
        Processed 2D points array.
    states : np.ndarray
        Processed 6D states array.
    times : np.ndarray | None
        Processed times array.
    trajectory_indices : np.ndarray
        Trajectory index for each point.
    metadata : dict[str, Any]
        Additional metadata.
    """

    hits: list[list[_SectionHit]]
    points: np.ndarray
    states: np.ndarray
    times: np.ndarray | None
    trajectory_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class SynodicMapResults(_MapResults):
    """User-facing results for synodic sections (extends 
    :class:`~hiten.algorithms.poincare.core.types._MapResults`).
    """
    
    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str], times: np.ndarray | None = None, trajectory_indices: np.ndarray | None = None):
        super().__init__(points, states, labels, times)
        self.trajectory_indices: np.ndarray | None = trajectory_indices


@dataclass(frozen=True)
class SynodicMapDomainPayload(_DomainPayload):
    """Domain payload capturing synodic map results and metadata."""

    @classmethod
    def _from_mapping(cls, data: Mapping[str, object]) -> "SynodicMapDomainPayload":
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
    def trajectory_indices(self) -> Optional[np.ndarray]:
        idx = self.get("trajectory_indices")
        return None if idx is None else np.asarray(idx, dtype=int)

    @property
    def labels(self) -> Tuple[str, str]:
        return tuple(self.require("labels"))

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
    normal : Sequence[float] | np.ndarray
        Normal vector defining the section plane.
    offset : float
        Offset distance for the section plane.
    map_cfg : SynodicMapConfig
        Map configuration containing detection parameters.
    """

    plane_coords: Tuple[str, str]
    direction: Optional[int]
    n_workers: int
    normal: Sequence[float] | np.ndarray
    offset: float
    trajectories: Optional[Sequence[tuple[np.ndarray, np.ndarray]]]
    interp_kind: Literal["linear", "cubic"]
    segment_refine: int
    tol_on_surface: float
    dedup_time_tol: float
    dedup_point_tol: float
    max_hits_per_traj: int | None
    newton_max_iter: int
