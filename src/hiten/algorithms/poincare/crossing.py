from typing import Callable, Literal

import numpy as np
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.base import (_DynamicalSystemProtocol,
                                            _propagate_dynsys)
from hiten.algorithms.poincare.events import (_PlaneEvent, _SectionHit,
                                              _SurfaceEvent)
from hiten.utils.log_config import logger


def _value_at_time(
    dynsys: _DynamicalSystemProtocol,
    state_ref: np.ndarray,
    t_ref: float,
    t_query: float,
    surface: _SurfaceEvent,
    *,
    forward: int = 1,
    steps: int = 500,
    method: Literal['scipy', 'rk', 'symplectic', 'adaptive'] = 'scipy',
    order: int = 8,
) -> float:
    """Return surface.value at absolute time *t_query* starting from *state_ref*.

    The helper integrates *only* the segment between ``t_ref`` and
    ``t_query`` so it can be used inside root-finding iterations without
    re-integrating the full trajectory each time.
    """
    if np.isclose(t_query, t_ref, rtol=3e-10, atol=1e-10):
        return surface.value(state_ref)

    sol_seg = _propagate_dynsys(
        dynsys,
        state_ref,
        t_ref,
        t_query,
        forward=forward,
        steps=steps,
        method=method,
        order=order,
    )
    state_final = sol_seg.states[-1]
    return surface.value(state_final)


def _bracket_root(
    f: Callable[[float], float],
    x0: float,
    surface,
    *,
    dx_init: float = 1e-10,
    max_expand: int = 500,
) -> tuple[float, float]:
    """Search outwards from *x0* until a sign change satisfying *surface* direction."""

    f0 = f(x0)
    if abs(f0) < 1e-14:
        return x0, x0  # already on the surface

    dx = dx_init
    for _ in range(max_expand):
        # right side
        x_r = x0 + dx
        fr = f(x_r)
        if surface.is_crossing(f0, fr):
            return (x0, x_r) if x0 < x_r else (x_r, x0)

        # left side
        x_l = x0 - dx
        fl = f(x_l)
        if surface.is_crossing(f0, fl):
            return (x_l, x0) if x_l < x0 else (x0, x_l)

        dx *= np.sqrt(2)

    raise RuntimeError("Failed to bracket root for section crossing.")


def find_crossing(
    dynsys: _DynamicalSystemProtocol,
    state0: np.ndarray,
    surface: _SurfaceEvent,
    *,
    t_guess: float | None = None,
    forward: Literal[1, -1] = 1,
    pre_steps: int = 1000,
    refine_steps: int = 3000,
    bracket_dx: float = 1e-10,
    max_expand: int = 500,
    method: Literal['scipy', 'rk', 'symplectic', 'adaptive'] = 'scipy',
    order: int = 8,
) -> _SectionHit:
    """Return the first intersection of the trajectory with *surface*.

    Parameters
    ----------
    dynsys
        Dynamical system to propagate.
    state0
        Initial state vector (1-D numpy array).
    surface
        Section definition.
    t_guess
        Coarse propagation time used to move *close* to the surface before root
        bracketing.  If *None*, uses the historical heuristic ``π/2 - 0.15``.
    forward
        +1 for forward integration, -1 for backward.
    pre_steps, refine_steps
        Number of integration steps for the coarse and refined propagations.
    bracket_dx, max_expand
        Parameters for the bracketing search.
    """

    t0_z = float(t_guess) if t_guess is not None else (np.pi / 2.0 - 0.15)

    logger.debug("Propagating from t=0 to t=%s for initial crossing guess", t0_z)
    sol_coarse = _propagate_dynsys(
        dynsys,
        state0,
        0.0,
        t0_z,
        forward=forward,
        steps=pre_steps,
        method=method,
        order=order,
    )
    state_mid = sol_coarse.states[-1]

    logger.debug("Bracketing root starting at t0_z=%s", t0_z)

    def _g(t: float) -> float:
        return _value_at_time(
            dynsys,
            state_mid,
            t0_z,
            t,
            surface,
            forward=forward,
            steps=refine_steps,
            method=method,
            order=order,
        )

    a, b = _bracket_root(_g, t0_z, surface, dx_init=bracket_dx, max_expand=max_expand)

    root_t = root_scalar(_g, bracket=(a, b), method="brentq", xtol=1e-12).root
    logger.debug("Found section crossing at t=%s", root_t)

    sol_final = _propagate_dynsys(
        dynsys,
        state_mid,
        t0_z,
        root_t,
        forward=forward,
        steps=refine_steps,
        method=method,
        order=order,
    )
    state_cross = sol_final.states[-1].copy()

    point2d = state_cross[:2].copy()

    return _SectionHit(time=root_t, state=state_cross, point2d=point2d)


def _plane_crossing_factory(coord: str, value: float = 0.0, direction: int | None = None):
    """Return a callable (dynsys, x0, forward=1 or -1) using find_crossing on a PlaneEvent.
    The *direction* argument controls which sign-change direction is accepted when detecting a crossing.
    """

    event = _PlaneEvent(coord=coord, value=value, direction=direction)

    def _section_crossing(*, dynsys, x0, forward: int = 1, **kwargs):
        hit = find_crossing(dynsys, np.asarray(x0, dtype=float), event, forward=forward)
        return hit.time, hit.state

    return _section_crossing

# Default to accepting crossings in either direction unless the caller needs a
# specific orientation.
_x_plane_crossing = _plane_crossing_factory("x", 0.0, None)
_y_plane_crossing = _plane_crossing_factory("y", 0.0, None)
_z_plane_crossing = _plane_crossing_factory("z", 0.0, None)

