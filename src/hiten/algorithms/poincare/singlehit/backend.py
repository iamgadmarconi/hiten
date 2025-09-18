"""Concrete backend implementation for single-hit Poincare sections.

This module provides a concrete implementation of the return map backend
for single-hit Poincare sections. It implements the generic surface-of-section
crossing search using numerical integration and root finding.

The main class :class:`~hiten.algorithms.poincare.singlehit.backend._SingleHitBackend` 
extends the abstract base class
to provide a complete implementation for finding single trajectory-section
intersections.
"""

from typing import Callable, Literal

import numpy as np
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                                   _SurfaceEvent)


class _SingleHitBackend(_ReturnMapBackend):
    """Concrete backend for single-hit Poincare section crossing search.

    This class implements the generic surface-of-section crossing search
    for single-hit Poincare sections. It extends the abstract base class
    to provide a complete implementation using numerical integration and
    root finding.

    The backend uses a two-stage approach:
    1. Coarse integration to get near the section
    2. Fine root finding to locate the exact crossing point

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol`
        The dynamical system providing the equations of motion.
    surface : :class:`~hiten.algorithms.poincare.core.events._SurfaceEvent`
        The Poincare section surface definition.
    forward : int, default=1
        Integration direction (1 for forward, -1 for backward).
    method : {'scipy', 'rk', 'symplectic', 'adaptive'}, default='scipy'
        Integration method to use.
    order : int, default=8
        Integration order for Runge-Kutta methods.
    pre_steps : int, default=1000
        Number of pre-integration steps for coarse integration.
    refine_steps : int, default=3000
        Number of refinement steps for root finding.
    bracket_dx : float, default=1e-10
        Initial bracket size for root finding.
    max_expand : int, default=500
        Maximum bracket expansion iterations.

    Notes
    -----
    This backend is optimized for single-hit computations where only
    the first intersection with the section is needed. It uses efficient
    root finding to locate the exact crossing point after coarse integration.

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        forward: int = 1,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
        order: int = 8,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
    ) -> None:
        super().__init__(
            dynsys=dynsys,
            surface=surface,
            forward=forward,
            method=method,
            order=order,
            pre_steps=pre_steps,
            refine_steps=refine_steps,
            bracket_dx=bracket_dx,
            max_expand=max_expand,
        )

    def step_to_section(
        self,
        seeds: np.ndarray,
        *,
        dt: float = 1e-2,
        t_guess: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the next crossing for every seed.

        This method implements the core functionality of the single-hit
        backend, finding the first intersection of each seed trajectory
        with the Poincare section.

        Parameters
        ----------
        seeds : ndarray, shape (m, 6)
            Array of initial state vectors [x, y, z, vx, vy, vz] in
            nondimensional units.
        dt : float, default=1e-2
            Integration time step (nondimensional units). Used for
            Runge-Kutta methods, ignored for adaptive methods.
        t_guess : float, optional
            Initial guess for the crossing time. If None, uses a
            default value based on the orbital period.

        Returns
        -------
        points : ndarray, shape (k, 2)
            Array of 2D intersection points in the section plane.
            Only includes trajectories that successfully cross the section.
        states : ndarray, shape (k, 6)
            Array of full state vectors at the intersection points.
            Shape matches points array.

        Notes
        -----
        This method processes each seed individually, finding the first
        intersection with the section. Trajectories that don't cross
        the section are excluded from the results.

        The method uses a two-stage approach:
        1. Coarse integration to get near the section
        2. Fine root finding to locate the exact crossing

        The 2D projection uses the first two coordinates as a fallback
        projection method.
        """
        pts, states = [], []
        for s in seeds:
            hit = self._cross(s, t_guess=t_guess)
            if hit is not None:
                pts.append(hit.point2d)
                states.append(hit.state.copy())

        if pts:
            return np.asarray(pts, float), np.asarray(states, float)
        return np.empty((0, 2)), np.empty((0, 6))

    def _value_at_time(self, state_ref: np.ndarray, t_ref: float, t_query: float):
        """Evaluate the surface function at a given time.

        Parameters
        ----------
        state_ref : ndarray, shape (6,)
            Reference state vector at time t_ref.
        t_ref : float
            Reference time (nondimensional units).
        t_query : float
            Query time (nondimensional units).

        Returns
        -------
        float
            Value of the surface function at the query time.

        Notes
        -----
        This method efficiently evaluates the surface function at a
        given time by either using the reference state directly (if
        times are very close) or by integrating from the reference
        state to the query time.

        The method uses the configured integration method and parameters
        for the integration step.
        """
        if np.isclose(t_query, t_ref, rtol=3e-10, atol=1e-10):
            return self._surface.value(state_ref)

        sol_seg = _propagate_dynsys(
            self._dynsys,
            state_ref,
            t_ref,
            t_query,
            forward=self._forward,
            steps=self._refine_steps,
            method=self._method,
            order=self._order,
        )
        state_final = sol_seg.states[-1]
        return self._surface.value(state_final)

    def _bracket_root(self, f: Callable[[float], float], x0: float):
        """Bracket a root of the surface function.

        Parameters
        ----------
        f : callable
            Function whose root is being searched for.
        x0 : float
            Reference point around which to expand the bracket.

        Returns
        -------
        tuple[float, float]
            Bracket (a, b) containing the root.

        Notes
        -----
        This method uses the parent class's bracket expansion algorithm
        with parameters optimized for surface crossing detection. The
        crossing test is provided by the surface event.
        """
        return self._expand_bracket(
            f,
            x0,
            dx0=self._bracket_dx,
            grow=np.sqrt(2),
            max_expand=self._max_expand,
            crossing_test=self._surface.is_crossing,
            symmetric=True,
        )

    def _cross(self, state0: np.ndarray, *, t_guess: float | None = None, t0_offset: float = 0.15):
        """Find a single crossing of a trajectory with the section.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
        t_guess : float, optional
            Initial guess for the crossing time. If None, uses a default
            value based on the orbital period.
        t0_offset : float, default=0.15
            Offset from the default time guess (nondimensional units).

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.events._SectionHit` or None
            Section hit object containing the crossing time, state, and
            2D projection. Returns None if no crossing is found.

        Notes
        -----
        This method implements the two-stage approach:
        1. Coarse integration to get near the section
        2. Fine root finding to locate the exact crossing

        The method uses Brent's method for root finding after bracketing
        the root using the surface crossing detection logic.

        The 2D projection uses the first two coordinates as a fallback
        projection method.
        """
        t0_z = float(t_guess) if t_guess is not None else (np.pi / 2.0 - t0_offset)

        sol_coarse = _propagate_dynsys(
            self._dynsys,
            state0,
            0.0,
            t0_z,
            forward=self._forward,
            steps=self._pre_steps,
            method=self._method,
            order=self._order,
        )
        state_mid = sol_coarse.states[-1]

        def _g(t: float):
            return self._value_at_time(state_mid, t0_z, t)

        a, b = self._bracket_root(_g, t0_z)

        root_t = root_scalar(_g, bracket=(a, b), method="brentq", xtol=1e-12).root

        sol_final = _propagate_dynsys(
            self._dynsys,
            state_mid,
            t0_z,
            root_t,
            forward=self._forward,
            steps=self._refine_steps,
            method=self._method,
            order=self._order,
        )
        state_cross = sol_final.states[-1].copy()

        # Fallback 2-D projection: first two coordinates
        point2d = state_cross[:2].copy()

        return _SectionHit(time=root_t, state=state_cross, point2d=point2d)


def find_crossing(dynsys, state0, surface, **kwargs):
    """Find a single crossing for a given state and surface.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol`
        The dynamical system providing the equations of motion.
    state0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
    surface : :class:`~hiten.algorithms.poincare.core.events._SurfaceEvent`
        The Poincare section surface definition.
    **kwargs
        Additional keyword arguments passed to the backend constructor.

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple of (points, states) arrays from the backend's step_to_section method.

    Notes
    -----
    This is a convenience function that creates a single-hit backend
    and finds the crossing for a single state vector. It's useful
    for simple crossing computations without needing to create a
    backend instance explicitly.
    """
    be = _SingleHitBackend(dynsys=dynsys, surface=surface, **kwargs)
    return be.step_to_section(np.asarray(state0, float))


def _plane_crossing_factory(coord: str, value: float = 0.0, direction: int | None = None):
    """Factory function for creating plane crossing functions.

    Parameters
    ----------
    coord : str
        Coordinate identifier for the plane (e.g., 'x', 'y', 'z').
    value : float, default=0.0
        Plane offset value (nondimensional units).
    direction : {1, -1, None}, optional
        Crossing direction filter.

    Returns
    -------
    callable
        A function that finds crossings for the specified plane.

    Notes
    -----
    This factory function creates specialized crossing functions for
    specific coordinate planes. The returned function takes a dynamical
    system and initial state and returns the crossing time and state.

    The returned function signature is:
    _section_crossing(*, dynsys, x0, forward=1, **kwargs) -> (time, state)
    """
    event = _PlaneEvent(coord=coord, value=value, direction=direction)

    def _section_crossing(*, dynsys, x0, forward: int = 1, **kwargs):
        # Ensure the seed state is treated as a full 6-D vector and find a single crossing
        be = _SingleHitBackend(dynsys=dynsys, surface=event, forward=forward)
        hit = be._cross(np.asarray(x0, float))  # compute single crossing
        return hit.time, hit.state

    return _section_crossing

# Predefined crossing functions for common coordinate planes
_x_plane_crossing = _plane_crossing_factory("x", 0.0, None)
_y_plane_crossing = _plane_crossing_factory("y", 0.0, None)
_z_plane_crossing = _plane_crossing_factory("z", 0.0, None)