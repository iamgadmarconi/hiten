"""Event detection utilities for in-loop, Numba-compatible integrators.

Notes
-----
This module is intentionally lightweight to minimize per-step overhead. All
heavy work (root refinement) is only activated when a sign change is detected
across an accepted step. The refinement uses a small RK4 local step kernel to
predict mid-point states; for DOP853 callers, this can later be replaced by a
dense-output evaluator without changing the public API here.
"""
from typing import Callable, Tuple

import numpy as np
from numba import njit

from hiten.algorithms.integrators.configs import _EventConfig
from hiten.algorithms.integrators.types import EventResult


@njit(cache=False, fastmath=True)
def _direction_allows(g0: float, g1: float, direction: int) -> bool:
    """Return True if the sign change (g0 -> g1) matches desired direction.

    direction = 0 allows any sign change; +1 requires increasing; -1 decreasing.
    Zeros at endpoints are considered crossings.
    """
    if g0 == 0.0 or g1 == 0.0:
        if direction == 0:
            return True
        # If exactly zero, look at a tiny bias via inequality
        if direction > 0:
            return g1 >= g0
        else:
            return g1 <= g0
    prod = g0 * g1
    if prod > 0.0:
        return False
    if direction == 0:
        return True
    # increasing crossing: g0 < 0 < g1
    if direction > 0:
        return (g0 < 0.0) and (g1 > 0.0)
    # decreasing crossing: g0 > 0 > g1
    return (g0 > 0.0) and (g1 < 0.0)


@njit(cache=False, fastmath=True)
def _rk4_local_step(f: Callable[[float, np.ndarray], np.ndarray], t0: float, y0: np.ndarray, h: float) -> np.ndarray:
    """Perform a single classical RK4 step of size h from (t0, y0).

    This local kernel is used exclusively during event-time refinement to
    predict mid-point states. It avoids coupling with large RK drivers and
    keeps the refinement routine fully Numba-compatible.
    """
    k1 = f(t0, y0)
    y_mid = y0 + 0.5 * h * k1
    k2 = f(t0 + 0.5 * h, y_mid)
    y_mid = y0 + 0.5 * h * k2
    k3 = f(t0 + 0.5 * h, y_mid)
    y_end = y0 + h * k3
    k4 = f(t0 + h, y_end)
    return y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=False, fastmath=True)
def _refine_bisection(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], float],
    ta: float,
    ya: np.ndarray,
    ga: float,
    tb: float,
    yb: np.ndarray,
    gb: float,
    tol: float,
    max_iter: int,
) -> Tuple[bool, float, np.ndarray, float]:
    """Refine an event time by bisection within [ta, tb].

    Uses a small RK4 local step from the left endpoint at each iteration to
    obtain the mid state. This is sufficient and robust for event localization
    and keeps dependencies minimal.
    """
    a_t = ta
    a_y = ya.copy()
    a_g = ga
    b_t = tb
    b_y = yb.copy()
    b_g = gb

    # Ensure bracket validity
    if not (a_g == 0.0 or b_g == 0.0 or a_g * b_g <= 0.0):
        return False, 0.0, ya, ga

    for _ in range(max_iter):
        mid_t = 0.5 * (a_t + b_t)
        h = mid_t - a_t
        if h == 0.0:
            # Degenerate (time tolerance reached)
            y_mid = a_y.copy()
        else:
            y_mid = _rk4_local_step(f, a_t, a_y, h)
        g_mid = g(mid_t, y_mid)

        # Convergence in time
        if np.abs(b_t - a_t) <= tol:
            return True, mid_t, y_mid, g_mid

        if g_mid == 0.0:
            return True, mid_t, y_mid, g_mid

        # Re-bracket
        if a_g * g_mid <= 0.0:
            # Keep left, move right to mid
            b_t = mid_t
            b_y = y_mid
            b_g = g_mid
        else:
            # Move left to mid
            a_t = mid_t
            a_y = y_mid
            a_g = g_mid

    # Best effort after max_iter
    mid_t = 0.5 * (a_t + b_t)
    h = mid_t - a_t
    if h == 0.0:
        y_mid = a_y.copy()
    else:
        y_mid = _rk4_local_step(f, a_t, a_y, h)
    g_mid = g(mid_t, y_mid)
    return True, mid_t, y_mid, g_mid


@njit(cache=False, fastmath=True)
def _check_and_refine_event(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], float],
    t0: float,
    y0: np.ndarray,
    t1: float,
    y1: np.ndarray,
    direction: int,
    tol: float,
    max_iter: int,
) -> Tuple[bool, float, np.ndarray, float]:
    """Detect sign change between (t0,y0) and (t1,y1) and refine event time.

    Returns (hit, t_event, y_event, g_event).
    """
    g0 = g(t0, y0)
    g1 = g(t1, y1)

    if not _direction_allows(g0, g1, direction):
        return False, 0.0, y0, g0

    ok, te, ye, ge = _refine_bisection(f, g, t0, y0, g0, t1, y1, g1, tol, max_iter)
    if not ok:
        return False, 0.0, y0, g0
    return True, te, ye, ge


def check_and_refine_event(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], float],
    t0: float,
    y0: np.ndarray,
    t1: float,
    y1: np.ndarray,
    cfg: _EventConfig,
) -> EventResult:
    """Thin Python wrapper calling the Numba kernel with config fields.

    Designed for minimal overhead: a single function call from the solver.
    """
    hit, te, ye, ge = _check_and_refine_event(
        f, g, t0, y0, t1, y1, int(cfg.direction), float(cfg.tol), int(cfg.max_iter)
    )
    if not hit:
        return EventResult(False, None, None, None)
    return EventResult(True, float(te), ye, float(ge))
