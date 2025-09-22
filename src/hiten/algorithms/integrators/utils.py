"""Shared Numba-compatible helpers for integrators.

This module deduplicates small routines used across multiple integrators,
such as directional crossing predicates and bisection bracket updates.
"""

import numba
import numpy as np

from hiten.algorithms.utils.config import FASTMATH


@numba.njit(cache=False, fastmath=FASTMATH)
def _event_crossed(g_prev: float, g_new: float, direction: int) -> bool:
    """Return True if a crossing consistent with direction occurred.

    Accepts endpoint zeros (g_new == 0.0) to catch exact hits at the right endpoint.
    direction: 0 -> any; >0 -> increasing; <0 -> decreasing.

    Parameters
    ----------
    g_prev : float
        The function value at the previous time.
    g_new : float
        The function value at the new time.
    direction : int
        The direction of the crossing.

    Returns
    -------
    bool
        True if a crossing consistent with direction occurred.
    """
    if direction == 0:
        return (g_prev < 0.0 and g_new > 0.0) or (g_prev > 0.0 and g_new < 0.0) or (g_new == 0.0)
    elif direction > 0:
        return (g_prev < 0.0 and g_new > 0.0) or (g_new == 0.0)
    else:
        return (g_prev > 0.0 and g_new < 0.0) or (g_new == 0.0)


@numba.njit(cache=False, fastmath=FASTMATH)
def _crossed_direction(g_left: float, g_mid: float, direction: int) -> bool:
    """Return True if the sign change from left to mid matches direction.

    This variant is used inside in-step refinement where bracket endpoints
    are maintained as [a,b] mapped to [g_left, g_right]. Endpoint zeros are
    handled by the caller using a |g_mid| <= gtol test.

    Parameters
    ----------
    g_left : float
        The function value at the left endpoint.
    g_mid : float
        The function value at the midpoint.
    direction : int
        The direction of the crossing.

    Returns
    -------
    bool
        True if the sign change from left to mid matches direction.
    """
    if direction == 0:
        return (g_left < 0.0 and g_mid > 0.0) or (g_left > 0.0 and g_mid < 0.0)
    elif direction > 0:
        return (g_left < 0.0 and g_mid > 0.0)
    else:
        return (g_left > 0.0 and g_mid < 0.0)


@numba.njit(cache=False, fastmath=FASTMATH)
def _bisection_update(a: float, b: float, g_left: float, mid: float, g_mid: float, crossed: bool):
    """Update bracket [a,b] and left value using a classic bisection step.

    Parameters
    ----------
    a, b : float
        Current bracket endpoints with a < b.
    g_left : float
        Function value at the left endpoint (corresponding to 'a').
    mid : float
        Midpoint (0.5 * (a + b)).
    g_mid : float
        Function value at the midpoint.
    crossed : bool
        Whether a sign-consistent crossing was found in [a, mid].

    Returns
    -------
    a_new, b_new, g_left_new : tuple[float, float, float]
        Updated bracket and left endpoint value consistent with the choice.
    """
    if crossed:
        b = mid
        # g_left unchanged
    else:
        a = mid
        g_left = g_mid
    return a, b, g_left


@numba.njit(cache=False, fastmath=FASTMATH)
def _bracket_converged(a: float, b: float, h: float, xtol: float) -> bool:
    """Return True if bracket size mapped to time is within tolerance.
    
    Parameters
    ----------
    a : float
        The left endpoint of the bracket.
    b : float
        The right endpoint of the bracket.
    h : float
        The step size.
    xtol : float
        The tolerance.

    Returns
    -------
    bool
        True if the bracket is converged.
    """
    return (b - a) * abs(h) <= xtol


@numba.njit(cache=False, fastmath=FASTMATH)
def _select_initial_step(d0: float, d1: float, min_step: float, max_step: float) -> float:
    """Heuristic initial step selection used by adaptive RK methods.

    Matches the pattern used across RK45/DOP853: small fixed step for tiny norms,
    otherwise 0.01 * d0 / d1, clamped to [min_step, max_step].

    Parameters
    ----------
    d0 : float
        The initial derivative.
    d1 : float
        The second derivative.
    min_step : float
        The minimum step size.
    max_step : float
        The maximum step size.

    Returns
    -------
    float
        The initial step size.
    """
    if d0 < 1.0e-5 or d1 < 1.0e-5:
        h = 1.0e-6
    else:
        h = 0.01 * d0 / d1
    if h > max_step:
        h = max_step
    if h < min_step:
        h = min_step
    return h


@numba.njit(cache=False, fastmath=FASTMATH)
def _clamp_step(h: float, max_step: float, min_step: float) -> float:
    """Clamp step size into [min_step, max_step].

    Parameters
    ----------
    h : float
        The step size.
    max_step : float
        The maximum step size.
    min_step : float
        The minimum step size.

    Returns
    -------
    float
        The clamped step size.
    """
    if h > max_step:
        h = max_step
    if h < min_step:
        h = min_step
    return h


@numba.njit(cache=False, fastmath=FASTMATH)
def _adjust_step_to_endpoint(t: float, h: float, t_end: float) -> float:
    """Adjust step so that t + h does not overshoot t_end for forward time.

    Assumes forward integration (t_end >= t).
    
    Parameters
    ----------
    t : float
        The current time.
    h : float
        The step size.
    t_end : float
        The end time.

    Returns
    -------
    float
        The adjusted step size.
    """
    if t + h > t_end:
        return abs(t_end - t)
    return h


_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0


@numba.njit(cache=False, fastmath=FASTMATH)
def _pi_accept_factor(err_norm: float, err_prev: float, order: float) -> float:
    """Compute PI controller factor after an accepted step.

    Uses Hairer-style PI control with alpha = 0.4/(order+1) and beta = 1/(order+1).
    Returns a factor clamped to [_MIN_FACTOR, _MAX_FACTOR] with finite fallback.

    Parameters
    ----------
    err_norm : float
        The error norm.
    err_prev : float
        The previous error norm.
    order : float
        The order of the method.

    Returns
    -------
    float
        The PI controller factor.
    """
    beta = 1.0 / (order + 1.0)
    alpha = 0.4 * beta
    if err_prev < 0.0:
        if err_norm == 0.0:
            factor = _MAX_FACTOR
        else:
            factor = _SAFETY * (err_norm ** (-beta))
    else:
        if err_norm == 0.0:
            factor = _MAX_FACTOR
        else:
            factor = _SAFETY * (err_norm ** (-beta)) * (err_prev ** alpha)
    # clamp and sanitize
    if not (factor == factor):  # NaN check
        factor = _MAX_FACTOR
    if factor < _MIN_FACTOR:
        factor = _MIN_FACTOR
    if factor > _MAX_FACTOR:
        factor = _MAX_FACTOR
    return factor


@numba.njit(cache=False, fastmath=FASTMATH)
def _pi_reject_factor(err_norm: float, order: float) -> float:
    """Compute PI controller factor after a rejected step.

    Uses exponent err_exp = 1/order; clamps and sanitizes the factor.

    Parameters
    ----------
    err_norm : float
        The error norm.
    order : float
        The order of the method.

    Returns
    -------
    float
        The PI controller factor.
    """
    err_exp = 1.0 / order
    if err_norm <= 0.0:
        factor = _MIN_FACTOR
    else:
        factor = _SAFETY * (err_norm ** (-err_exp))
    if not (factor == factor):  # NaN
        factor = _MIN_FACTOR
    if factor < _MIN_FACTOR:
        factor = _MIN_FACTOR
    if factor > _MAX_FACTOR:
        factor = _MAX_FACTOR
    return factor


@numba.njit(cache=False, fastmath=FASTMATH)
def _error_scale(y, y_high, rtol: float, atol: float):
    """Return elementwise error scale vector used for error norms.

    Computes atol + rtol * max(|y|, |y_high|) elementwise.
    """
    return atol + rtol * np.maximum(np.abs(y), np.abs(y_high))
