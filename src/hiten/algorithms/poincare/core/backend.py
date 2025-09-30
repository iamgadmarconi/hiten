"""Abstract base class for Poincare return map backends.

This module provides the abstract base class for implementing return map
backends in the Poincare section framework. Backends handle the numerical
integration and section crossing detection for computing Poincare maps.

The main class :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend` 
defines the interface that all concrete backends must implement, including the 
core `step_to_section` method and common functionality for root finding and bracket 
expansion.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from hiten.algorithms.types.core import _HitenBaseBackend


class _ReturnMapBackend(_HitenBaseBackend):
    """Abstract base class for Poincare return map backends.

    This class defines the interface that all concrete return map backends
    must implement. It provides common functionality for numerical integration,
    section crossing detection, and root finding.

    Parameters
    ----------
    forward : int, default=1
        Integration direction (1 for forward, -1 for backward).
    method : {'fixed', 'symplectic', 'adaptive'}, default='adaptive'
        Integration method to use.
    order : int, default=8
        Integration order for Runge-Kutta methods.
    pre_steps : int, default=1000
        Number of pre-integration steps for trajectory stabilization.
    refine_steps : int, default=3000
        Number of refinement steps for root finding.
    bracket_dx : float, default=1e-10
        Initial bracket size for root finding.
    max_expand : int, default=500
        Maximum bracket expansion iterations.

    Notes
    -----
    Subclasses must implement the `step_to_section` method to define
    how trajectories are integrated from one section crossing to the next.
    The backend handles the numerical integration and section crossing
    detection, while the engine layer manages iteration, caching, and
    parallel processing.

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self) -> None:

        self._section_cache = None
        self._grid_cache = None

    @abstractmethod
    def run(
        self,
        seeds: "np.ndarray",
        *,
        dt: float = 1e-2,
        forward: int = 1,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order: int = 8,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Propagate seeds to the next surface crossing.

        This abstract method must be implemented by concrete backends to
        define how trajectories are integrated from initial seeds to their
        next intersection with the Poincare section.

        Parameters
        ----------
        seeds : ndarray, shape (m, n)
            Array of initial states. The shape depends on the backend:
            - Center manifold backends: (m, 4) for [q2, p2, q3, p3]
            - Full state backends: (m, 6) for [q1, q2, q3, p1, p2, p3]
        dt : float, default=1e-2
            Integration time step (nondimensional units). Meaningful for
            Runge-Kutta methods, ignored for adaptive methods.

        Returns
        -------
        points : ndarray, shape (k, 2)
            Crossing coordinates in the section plane.
        states : ndarray, shape (k, n)
            State representation at the crossings. Shape matches input
            seeds but may have fewer rows if some trajectories don't
            reach the section.

        Notes
        -----
        This method performs a single step of the Poincare map, taking
        initial conditions and returning their next intersection with
        the section. The engine layer handles iteration and caching.
        """
