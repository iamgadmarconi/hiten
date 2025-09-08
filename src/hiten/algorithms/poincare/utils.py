"""Utility functions for Poincare section computations.

This module provides optimized utility functions for interpolation
and numerical computations used in Poincare section detection and
refinement. The functions are optimized using Numba JIT compilation
for high performance in numerical computations.
"""

import numpy as np
from numba import njit

from hiten.algorithms.utils.config import FASTMATH


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _interp_linear(t0: float, x0: np.ndarray, t1: float, x1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points.

    Parameters
    ----------
    t0 : float
        Start time (nondimensional units).
    x0 : ndarray
        Start state vector (nondimensional units).
    t1 : float
        End time (nondimensional units).
    x1 : ndarray
        End state vector (nondimensional units).
    t : float
        Interpolation time (nondimensional units).

    Returns
    -------
    ndarray
        Interpolated state vector at time t.

    Notes
    -----
    This function performs linear interpolation between two points
    in time and state space. It computes the interpolation parameter
    s = (t - t0) / (t1 - t0) and uses it to linearly interpolate
    between the start and end states.

    The function is optimized with Numba JIT compilation for
    high performance in numerical computations.

    All time units are in nondimensional units.
    """
    s = (t - t0) / (t1 - t0)
    return (1.0 - s) * x0 + s * x1


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _hermite_scalar(s: float, y0: float, y1: float, dy0: float, dy1: float, dt: float) -> float:
    """Cubic Hermite interpolation for scalar values.

    Parameters
    ----------
    s : float
        Interpolation parameter in [0, 1].
    y0 : float
        Function value at s=0 (nondimensional units).
    y1 : float
        Function value at s=1 (nondimensional units).
    dy0 : float
        Derivative at s=0 (nondimensional units per time).
    dy1 : float
        Derivative at s=1 (nondimensional units per time).
    dt : float
        Time step (nondimensional units).

    Returns
    -------
    float
        Interpolated function value at parameter s.

    Notes
    -----
    This function implements cubic Hermite interpolation for scalar
    functions. It uses the standard Hermite basis functions:
    - h00(s) = (1 + 2s)(1 - s)^2
    - h10(s) = s(1 - s)^2
    - h01(s) = s^2(3 - 2s)
    - h11(s) = s^2(s - 1)

    The interpolation formula is:
        H(s) = h00(s)*y0 + h10(s)*dy0*dt + h01(s)*y1 + h11(s)*dy1*dt

    This provides C1 continuity and high accuracy for smooth functions.

    All units are in nondimensional units.
    """
    h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
    h10 = s * (1.0 - s) ** 2
    h01 = s ** 2 * (3.0 - 2.0 * s)
    h11 = s ** 2 * (s - 1.0)
    return h00 * y0 + h10 * dy0 * dt + h01 * y1 + h11 * dy1 * dt


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _hermite_der(s: float, y0: float, y1: float, dy0: float, dy1: float, dt_seg: float) -> float:
    """Derivative of cubic Hermite interpolation for scalar values.

    Parameters
    ----------
    s : float
        Interpolation parameter in [0, 1].
    y0 : float
        Function value at s=0 (nondimensional units).
    y1 : float
        Function value at s=1 (nondimensional units).
    dy0 : float
        Derivative at s=0 (nondimensional units per time).
    dy1 : float
        Derivative at s=1 (nondimensional units per time).
    dt_seg : float
        Time step (nondimensional units).

    Returns
    -------
    float
        Derivative of the interpolated function at parameter s.

    Notes
    -----
    This function computes the analytical derivative of the cubic
    Hermite interpolation polynomial. It uses the derivatives of
    the Hermite basis functions:
    - dh00(s) = 6s(s-1) + (1-s)^2*2 - 2(1-s)(1+2s)
    - dh10(s) = (1-s)^2 + s*2(s-1)
    - dh01(s) = 6s(1-s) - 2s(3-2s)
    - dh11(s) = 2s(s-1) + s^2

    The derivative formula is:
        dH/ds = dh00(s)*y0 + dh10(s)*dy0*dt + dh01(s)*y1 + dh11(s)*dy1*dt

    This is used in Newton's method for root finding in Poincare
    section detection and refinement.

    All units are in nondimensional units.
    """
    # Analytical derivative of the cubic Hermite polynomial
    dh00 = 6.0 * s * (s - 1.0) + (1.0 - s) ** 2 * 2.0 - 2.0 * (1.0 - s) * (1.0 + 2.0 * s)
    dh10 = (1.0 - s) ** 2 + s * (2.0 * (s - 1.0))
    dh01 = 6.0 * s * (1.0 - s) - 2.0 * s * (3.0 - 2.0 * s)
    dh11 = 2.0 * s * (s - 1.0) + s ** 2
    return dh00 * y0 + dh10 * dy0 * dt_seg + dh01 * y1 + dh11 * dy1 * dt_seg
