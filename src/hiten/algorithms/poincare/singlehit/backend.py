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
import time

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                                   _SurfaceEvent)
from hiten.algorithms.integrators import AdaptiveRK
from hiten.algorithms.integrators.configs import _EventConfig
from numba import njit, types


# --- Precompiled plane event functions (fast path for standard sections) ---
@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_x0(t: float, y: np.ndarray) -> float:
    return float(y[0])


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_y0(t: float, y: np.ndarray) -> float:
    return float(y[1])


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def _g_z0(t: float, y: np.ndarray) -> float:
    return float(y[2])


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

    def _cross_event_driven(self, state0: np.ndarray, *, t0: float, tmax: float) -> _SectionHit | None:
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
        _SectionHit or None
            Crossing information, or None if no crossing before tmax.
        """
        # Map surface to scalar event g(t,y)
        _t_total0 = time.perf_counter()
        _t_map0 = _t_total0
        direction = getattr(self._surface, "direction", None)
        normal_raw = getattr(self._surface, "normal", None)
        offset = getattr(self._surface, "offset", None)

        if isinstance(self._surface, _PlaneEvent) and normal_raw is not None and offset is not None:
            n_vec = np.asarray(normal_raw, dtype=np.float64)
            if not (float(offset) == 0.0 and n_vec.size >= 3):
                raise TypeError("Only axis-aligned zero-offset plane events are supported in event-driven backend")
            if n_vec[0] == 1.0 and n_vec[1] == 0.0 and n_vec[2] == 0.0:
                event_fn = _g_x0
                ev_name = "x==0"
            elif n_vec[0] == 0.0 and n_vec[1] == 1.0 and n_vec[2] == 0.0:
                event_fn = _g_y0
                ev_name = "y==0"
            elif n_vec[0] == 0.0 and n_vec[1] == 0.0 and n_vec[2] == 1.0:
                event_fn = _g_z0
                ev_name = "z==0"
            else:
                raise TypeError("Only axis-aligned zero-offset plane events are supported in event-driven backend")
        else:
            raise TypeError("_SingleHitBackend requires a plane event with normal and offset")

        # Build integrator and config
        _t_map1 = time.perf_counter()
        integrator = AdaptiveRK(order=8, rtol=1e-8, atol=1e-10)
        ev_dir = 0 if direction is None else int(direction)
        ev_cfg = _EventConfig(direction=ev_dir, terminal=True)
        _t_setup1 = time.perf_counter()

        # Diagnostics
        try:
            print(f"[SingleHit] surface normal={n_vec[:3]}, offset={float(offset):.3g}, direction={ev_dir}, event={ev_name}")
        except Exception:
            pass

        # Handle on-surface start to avoid immediate zero
        _t_g0_0 = time.perf_counter()
        g0 = event_fn(t0, state0)
        _t_g0_1 = time.perf_counter()
        t_start = float(t0)
        y_start = state0.astype(float, copy=True)
        print(f"[SingleHit] window t0={t0:.6g}, tmax={tmax:.6g}; g(t0)={g0:.3e}")
        print(f"[SingleHit][timing] map={(_t_map1 - _t_map0)*1e3:.2f} ms, setup={(_t_setup1 - _t_map1)*1e3:.2f} ms, g0_eval={( _t_g0_1 - _t_g0_0)*1e3:.2f} ms")
        if abs(g0) < 1e-12:
            # advance by epsilon time using RHS
            eps = 1e-9
            _t_eps0 = time.perf_counter()
            sol_eps = integrator.integrate(self._dynsys, y_start, np.array([t_start, t_start + eps], dtype=float))
            _t_eps1 = time.perf_counter()
            t_start = float(sol_eps.times[-1])
            y_start = sol_eps.states[-1].copy()
            g_start = event_fn(t_start, y_start)
            print(f"[SingleHit] on-surface start; advanced by {eps:.1e} to t={t_start:.6g}; g={g_start:.3e}")
            print(f"[SingleHit][timing] eps_integrate={(_t_eps1 - _t_eps0)*1e3:.2f} ms")

        # Integrate until event or tmax
        times = np.array([t_start, tmax], dtype=float)
        _t_int0 = time.perf_counter()
        sol = integrator.integrate(self._dynsys, y_start, times, event_fn=event_fn, event_cfg=ev_cfg)
        _t_int1 = time.perf_counter()
        t_hit = float(sol.times[-1])
        y_hit = sol.states[-1].copy()
        if t_hit < tmax and (t_hit - t_start) * 1.0 >= 0.0:
            g_hit = event_fn(t_hit, y_hit)
            print(f"[SingleHit] HIT at t={t_hit:.6g}, g={g_hit:.3e}, y[:3]={y_hit[:3]}")
            print(f"[SingleHit][timing] integrate={(_t_int1 - _t_int0)*1e3:.2f} ms, total={(time.perf_counter() - _t_total0)*1e3:.2f} ms")
            return _SectionHit(time=t_hit, state=y_hit, point2d=y_hit[:2].copy())
        print(f"[SingleHit] NO HIT before tmax; returned t={t_hit:.6g}, dt={t_hit - t_start:.3e}")
        print(f"[SingleHit][timing] integrate={(_t_int1 - _t_int0)*1e3:.2f} ms, total={(time.perf_counter() - _t_total0)*1e3:.2f} ms")
        return None

    def _cross(self, state0: np.ndarray, *, t_guess: float | None = None, t0_offset: float = 0.15):
        """Find a single crossing using event-driven integrators."""
        _t_cross0 = time.perf_counter()
        # Choose search horizon to avoid trivial early crossings:
        # Start near half-period and look for the first crossing in a π window.
        if t_guess is not None:
            t_start = float(t_guess)
        else:
            t_start = float(np.pi / 2.0 - t0_offset)
        if t_start < 0.0:
            t_start = 0.0
        t0 = t_start
        tmax = t_start + float(np.pi)
        print(f"[SingleHit] _cross window: t_start={t0:.6g} .. tmax={tmax:.6g} (t0_offset={t0_offset}, t_guess={t_guess})")

        hit = self._cross_event_driven(np.asarray(state0, float), t0=t0, tmax=tmax)
        print(f"[SingleHit][timing] _cross total={(time.perf_counter() - _t_cross0)*1e3:.2f} ms")
        return hit


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

    def _section_crossing(*, dynsys, x0, forward: int = 1, **kwargs):
        be = _SingleHitBackend(dynsys=dynsys, surface=event, forward=forward)
        hit = be._cross(np.asarray(x0, float))
        return hit.time, hit.state

    return _section_crossing

# Predefined crossing functions for common coordinate planes
_x_plane_crossing = _plane_crossing_factory("x", 0.0, None)
_y_plane_crossing = _plane_crossing_factory("y", 0.0, None)
_z_plane_crossing = _plane_crossing_factory("z", 0.0, None)