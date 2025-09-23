"""
Interface classes for synodic Poincare map computation.

This module provides interface classes that abstract synodic Poincare
map computation for the synodic module. These interfaces handle the
conversion between synodic map configuration and the engine interface.
"""

from typing import Literal, Sequence, Tuple

import numpy as np

from hiten.algorithms.poincare.core.interfaces import _SectionInterface
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.events import _AffinePlaneEvent


class _SynodicEngineConfig:
    """Configuration adapter for synodic Poincare engine.

    This adapter class provides the interface expected by the base
    return map engine while adapting the synodic map configuration
    to the required format. It handles the translation between
    synodic-specific parameters and the generic engine interface.

    Parameters
    ----------
    cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The synodic map configuration to adapt.

    Attributes
    ----------
    _cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The original synodic map configuration.
    dt : float
        Time step (set to 0.0 for synodic maps since they use precomputed trajectories).
    n_iter : int
        Number of iterations (set to 1 for synodic maps).
    n_workers : int
        Number of parallel workers for batch processing.
    n_seeds : int
        Number of seeds (set to 0 for synodic maps since they use precomputed trajectories).

    Notes
    -----
    This adapter is necessary because synodic Poincare maps operate on
    precomputed trajectories rather than integrating from initial conditions.
    The adapter provides the interface expected by the base engine while
    setting appropriate values for the synodic use case.

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self, cfg: _SynodicMapConfig) -> None:
        self._cfg = cfg
        self.dt = 0.0
        self.n_iter = 1
        self.n_workers = cfg.n_workers
        self.n_seeds = 0

    def __repr__(self) -> str:
        return f"_SynodicEngineConfig(n_workers={self.n_workers})"


class _SynodicSectionInterface(_SectionInterface):
    """Section interface for synodic affine hyperplanes."""

    section_coord: str | None  # not a single coord; keep None
    plane_coords: tuple[str, str]
    normal: np.ndarray
    offset: float

    @staticmethod
    def from_normal(normal: Sequence[float], offset: float, plane_coords: Tuple[str, str]) -> "_SynodicSectionInterface":
        n_arr = np.asarray(normal, dtype=float)
        if n_arr.ndim != 1 or n_arr.size != 6 or not np.all(np.isfinite(n_arr)):
            raise ValueError("normal must be a finite 1-D array of length 6")
        return _SynodicSectionInterface(section_coord=None, plane_coords=(str(plane_coords[0]), str(plane_coords[1])), normal=n_arr, offset=float(offset))

    def build_event(self, *, direction: Literal[1, -1, None] = None) -> _AffinePlaneEvent:
        return _AffinePlaneEvent(normal=self.normal, offset=self.offset, direction=direction)