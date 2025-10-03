"""Concrete backend implementation for single-hit Poincare sections.

This module provides a concrete implementation of the return map backend
for single-hit Poincare sections. It implements the generic
surface-of-section crossing search using numerical integration and root
finding.

The main class
:class:`~hiten.algorithms.poincare.singlehit.backend._SingleHitBackend`
extends the abstract base class
:class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend` to
provide a complete implementation for finding single
trajectory-section intersections.
"""

from typing import Literal

import numpy as np
from numba import njit, types

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.integrators.rk import RungeKutta
from hiten.algorithms.types.configs import EventConfig
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import _PlaneEvent, _SurfaceEvent
from hiten.algorithms.poincare.core.types import _SectionHit


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_x0(t: float, y: np.ndarray) -> float:
    return float(y[0])


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_y0(t: float, y: np.ndarray) -> float:
    return float(y[1])


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_z0(t: float, y: np.ndarray) -> float:
    return float(y[2])


# Cache for compiled plane event functions keyed by (index, offset)
_PLANE_EVENT_FN_CACHE = {}


def _get_cached_plane_event_fn(idx: int, offset: float):
    """Return a compiled g(t, y) = y[idx] - offset function, cached.

    Compiles once per (idx, offset) pair and reuses the CPUDispatcher so
    integrators receive a stable, already-compiled callable.
    """
    key = (int(idx), float(offset))
    fn = _PLANE_EVENT_FN_CACHE.get(key)
    if fn is not None:
        return fn

    i = int(idx)
    off = float(offset)

    @njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
    def _g(t: float, y: np.ndarray) -> float:
        return float(y[i] - off)

    _PLANE_EVENT_FN_CACHE[key] = _g
    return _g


class _SingleHitBackend(_ReturnMapBackend):
    """Concrete backend for single-hit Poincare section crossing search.

    This class implements the generic surface-of-section crossing search
    for single-hit Poincare sections. It extends the abstract base class
    to provide a complete implementation using numerical integration and
    root finding.

    The backend uses a two-stage approach:
    1. Coarse integration to get near the section
    2. Fine root finding to locate the exact crossing point

    Notes
    -----
    This backend is optimized for single-hit computations where only
    the first intersection with the section is needed. It uses efficient
    root finding to locate the exact crossing point after coarse
    integration.

    The backend is stateless - all dynamic system data must be passed to
    the step_to_section method as arguments.

    All time units are nondimensional unless otherwise specified.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def run(
        self,
        seeds: np.ndarray,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        dt: float = 1e-2,
        t_guess: float | None = None,
        forward: int = 1,
        method: Literal["fixed", "symplectic", "adaptive"] = "adaptive",
        order: int = 8,
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
            hit = self._cross(s, dynsys=dynsys, surface=surface, t_guess=t_guess, forward=forward)
            if hit is not None:
                pts.append(hit.point2d)
                states.append(hit.state.copy())

        if pts:
            return np.asarray(pts, float), np.asarray(states, float)
        return np.empty((0, 2)), np.empty((0, 6))

    def _cross_event_driven(self, state0: np.ndarray, *, dynsys: "_DynamicalSystemProtocol", surface: "_SurfaceEvent", t0: float, tmax: float, forward: int) -> _SectionHit | None:
        """Find a single crossing using event-driven integration.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial state at t=t0.
        t0 : float
            Start time.
        tmax : float
            Maximum time to search for a crossing.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.events._SectionHit` or None
            Crossing information, or ``None`` if no crossing before
            ``tmax``.
        """
        # Map surface to scalar event g(t,y)
        direction = getattr(surface, "direction", None)

        if isinstance(surface, _PlaneEvent):
            idx = int(getattr(surface, "index", 0))
            off = float(getattr(surface, "offset", 0.0))
            if off == 0.0 and idx in (0, 1, 2):
                if idx == 0:
                    event_fn = _g_x0
                elif idx == 1:
                    event_fn = _g_y0
                else:
                    event_fn = _g_z0
            else:
                event_fn = _get_cached_plane_event_fn(idx, off)
        else:
            raise TypeError("_SingleHitBackend requires a _PlaneEvent surface")

        # diagnostics disabled by default

        integrator = RungeKutta(order=853, rtol=1e-12, atol=1e-12)
        ev_dir = 0 if direction is None else int(direction)
        ev_cfg = EventConfig(direction=ev_dir, terminal=True)

        t_start = float(t0)
        # Avoid exactly-zero window start which may trigger an immediate event
        if t_start <= 0.0:
            t_start = 1e-12
        y_start = state0.astype(float, copy=True)

        sol_align = _propagate_dynsys(
            dynsys,
            y_start,
            0.0,
            t_start,
            forward=forward,
            steps=2,
            method="adaptive",
            order=8,
        )
        y_start = sol_align.states[-1].copy()

        span = float(max(0.0, tmax - t_start))
        times = np.array([0.0, span], dtype=float)
        # diagnostics disabled by default
        sol = integrator.integrate(dynsys, y_start, times, event_fn=event_fn, event_cfg=ev_cfg)
        t_hit_rel = float(sol.times[-1])
        y_hit = sol.states[-1].copy()
        t_hit = t_start + t_hit_rel
        if t_hit_rel < span and t_hit_rel >= 0.0:
            # diagnostics disabled by default
            return _SectionHit(time=t_hit, state=y_hit, point2d=y_hit[:2].copy(), trajectory_index=0)
        # diagnostics disabled by default
        return None

    def _cross(self, state0: np.ndarray, *, dynsys: "_DynamicalSystemProtocol", surface: "_SurfaceEvent", t_guess: float | None = None, t0_offset: float = 0.15, t_window: float | None = None, forward: int = 1):
        """Find a single crossing using event-driven integrators."""
        # If a hint is provided, confine search to a small window around it to keep branch selection consistent.
        if (t_guess is not None) and (t_window is not None) and (t_window > 0.0):
            t0 = float(max(t_guess - 0.5 * t_window, 0.0))
            tmax = float(t0 + t_window)
        else:
            # Start near half-period and look for the first crossing in a π window.
            if t_guess is not None:
                t_start = float(t_guess)
            else:
                t_start = float(np.pi / 2.0 - t0_offset)
            if t_start < 0.0:
                t_start = 0.0
            half_span = float(np.pi) * 0.5
            t0 = float(max(t_start - half_span, 0.0))
            tmax = float(t0 + 2.0 * half_span)
        # diagnostics disabled by default
        hit = self._cross_event_driven(np.asarray(state0, float), dynsys=dynsys, surface=surface, t0=t0, tmax=tmax, forward=forward)
        # If a hint was used and no hit was found, progressively widen the window (up to pi) then fall back
        if hit is None and t_guess is not None:
            span = float(t_window if (t_window is not None and t_window > 0.0) else 0.1)
            while span < float(np.pi):
                span *= 2.0
                t0_try = float(max(t_guess - 0.5 * span, 0.0))
                tmax_try = float(t0_try + span)
                # diagnostics disabled by default
                hit = self._cross_event_driven(np.asarray(state0, float), dynsys=dynsys, surface=surface, t0=t0_try, tmax=tmax_try, forward=forward)
                if hit is not None:
                    # diagnostics disabled by default
                    break
        # Final fallback: default half-period window
        if hit is None:
            if t_guess is not None:
                t_start = float(max(t_guess, 0.0))
            else:
                t_start = float(np.pi / 2.0 - t0_offset)
                if t_start < 0.0:
                    t_start = 0.0
            t0_fb = t_start
            tmax_fb = t_start + float(np.pi)
            # diagnostics disabled by default
            hit = self._cross_event_driven(np.asarray(state0, float), dynsys=dynsys, surface=surface, t0=t0_fb, tmax=tmax_fb, forward=forward)
        return hit



def find_crossing(dynsys, state0, surface, forward: int = 1, **kwargs):
    """Find a single crossing for a given state and surface.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
        The dynamical system providing the equations of motion.
    state0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
    surface : :class:`~hiten.algorithms.poincare.core.events._SurfaceEvent`
        The Poincare section surface definition.
    **kwargs
        Additional keyword arguments passed to the backend constructor.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Tuple of ``(points, states)`` arrays from the backend's
        ``step_to_section`` method.

    Notes
    -----
    This is a convenience function that creates a single-hit backend
    and finds the crossing for a single state vector. It's useful
    for simple crossing computations without needing to create a
    backend instance explicitly.
    """
    be = _SingleHitBackend(**kwargs)
    return be.run(np.asarray(state0, float), dynsys=dynsys, surface=surface, forward=forward)


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

    The returned function signature is::

        _section_crossing(*, dynsys, x0, forward=1, **kwargs) -> (time, state)
    """
    event = _PlaneEvent(coord=coord, value=value, direction=direction)
    # Attach explicit plane parameters for fast event selection downstream
    n = np.zeros(6, dtype=np.float64)
    if coord.lower() == "x":
        n[0] = 1.0
    elif coord.lower() == "y":
        n[1] = 1.0
    elif coord.lower() == "z":
        n[2] = 1.0
    else:
        raise ValueError(f"Unsupported plane coord '{coord}'. Must be one of 'x','y','z'.")
    # Provide attributes expected by the event-driven backend
    try:
        setattr(event, "normal", n)
        setattr(event, "offset", float(value))
    except Exception:
        pass

    def _section_crossing(*, dynsys, x0, forward: int = 1, t_guess: float | None = None, t_window: float | None = None):
        be = _SingleHitBackend()
        hit = be._cross(np.asarray(x0, float), dynsys=dynsys, surface=event, t_guess=t_guess, t_window=t_window, forward=forward)
        if hit is None:
            return None, None
        return hit.time, hit.state

    return _section_crossing

# Predefined crossing functions for common coordinate planes
_x_plane_crossing = _plane_crossing_factory("x", 0.0, None)
_y_plane_crossing = _plane_crossing_factory("y", 0.0, None)
_z_plane_crossing = _plane_crossing_factory("z", 0.0, None)