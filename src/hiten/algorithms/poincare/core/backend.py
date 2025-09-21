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
from typing import Callable, Literal

import numpy as np

from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.events import _SurfaceEvent
from hiten.algorithms.utils.exceptions import BackendError


class _ReturnMapBackend(ABC):
    """Abstract base class for Poincare return map backends.

    This class defines the interface that all concrete return map backends
    must implement. It provides common functionality for numerical integration,
    section crossing detection, and root finding.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
        Dynamical system providing the equations of motion.
    surface : :class:`~hiten.algorithms.poincare.core.events._SurfaceEvent`
        Poincare section surface definition.
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

    def __init__(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        forward: int = 1,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order: int = 8,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
    ) -> None:
        self._dynsys = dynsys
        self._surface = surface
        self._forward = 1 if forward >= 0 else -1
        self._method = method
        self._order = int(order)
        self._pre_steps = int(pre_steps)
        self._refine_steps = int(refine_steps)
        self._bracket_dx = float(bracket_dx)
        self._max_expand = int(max_expand)

        self._section_cache = None
        self._grid_cache = None

    # Each backend must implement a *single-step* worker that takes an array
    # of seeds and returns the crossings produced from those seeds. The engine
    # layer is then responsible for looping / caching / parallelism.

    @abstractmethod
    def step_to_section(
        self,
        seeds: "np.ndarray",
        *,
        dt: float = 1e-2,
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

    def _expand_bracket(
        self,
        f: "Callable[[float], float]",
        x0: float,
        *,
        dx0: float,
        grow: float,
        max_expand: int,
        crossing_test: "Callable[[float, float], bool]",
        symmetric: bool = True,
    ) -> tuple[float, float]:
        """Expand a bracket around a root of a scalar function.

        This method implements a robust bracket expansion algorithm for
        root finding. It starts from a reference point and expands the
        search interval until a root is bracketed.

        Parameters
        ----------
        f : callable
            Scalar function whose root is being searched for.
        x0 : float
            Reference point around which to start expanding the bracket.
        dx0 : float
            Initial half-width of the trial interval.
        grow : float
            Multiplicative factor applied to dx after every unsuccessful
            iteration.
        max_expand : int
            Maximum number of expansion attempts before giving up.
        crossing_test : callable
            A 2-argument predicate crossing_test(f_prev, f_curr) that returns
            True when the desired crossing is located inside (prev, curr).
        symmetric : bool, default=True
            If True, probe both the +dx and -dx directions; otherwise
            examine only the positive side.

        Returns
        -------
        tuple[float, float]
            Bracket (a, b) containing the root, with a < b.

        Raises
        ------
        BackendError
            If the root cannot be bracketed within max_expand iterations.

        Notes
        -----
        The algorithm starts with a small interval around x0 and expands
        it geometrically until a sign change is detected. If the function
        is already very close to zero at x0, a zero-length bracket is
        returned.
        """

        f0 = f(x0)

        # If we are already on the section (or very close) return a zero-length
        # bracket so the caller can decide what to do next.
        if abs(f0) < 1e-14:
            return (x0, x0)

        dx = dx0
        for _ in range(max_expand):
            # Probe +dx first (forward propagation).
            xr = x0 + dx
            fr = f(xr)
            if crossing_test(f0, fr):
                return (x0, xr) if x0 < xr else (xr, x0)

            if symmetric:
                xl = x0 - dx
                fl = f(xl)
                if crossing_test(f0, fl):
                    return (xl, x0) if xl < x0 else (x0, xl)

            dx *= grow

        raise BackendError("Failed to bracket root.")

    def on_iteration(self, iteration: int, seeds: "np.ndarray | None" = None) -> None:
        """Hook called at the start of each iteration in the engine."""
        return None

    def on_success(
        self,
        iteration: int,
        points: "np.ndarray",
        states: "np.ndarray",
        times: "np.ndarray | None" = None,
    ) -> None:
        """Hook called when an iteration produces crossings."""
        return None

    def on_failure(self, iteration: int) -> None:
        """Hook called when an iteration produces no crossings for a chunk."""
        return None

    def on_accept(
        self,
        points: "np.ndarray",
        states: "np.ndarray",
        times: "np.ndarray | None" = None,
    ) -> None:
        """Hook called after the engine aggregates final results."""
        return None
