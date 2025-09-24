"""Root-finding utilities shared across algorithms.

This module centralizes small, reusable numerical helpers for scalar
root-location that are not tied to any specific domain logic. It
implements:

- Bracket expansion around a candidate point
- A Brent-style bracketed root solver (pure Python/NumPy)

Notes
-----
These helpers are intentionally pure Python and NumPy only. Numba-specific
event refinements used by integrators should live with the integrator
implementations to preserve compilation constraints.
"""

from typing import Callable

import numpy as np

from hiten.algorithms.types.exceptions import BackendError


def expand_bracket(
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

    Parameters
    ----------
    f : callable
        Scalar function whose root is being searched for.
    x0 : float
        Reference point around which to start expanding the bracket.
    dx0 : float
        Initial half-width of the trial interval.
    grow : float
        Multiplicative factor applied to dx after every unsuccessful iteration.
    max_expand : int
        Maximum number of expansion attempts before giving up.
    crossing_test : callable
        A 2-argument predicate crossing_test(f_prev, f_curr) that returns True
        when the desired crossing is located inside (prev, curr).
    symmetric : bool, default=True
        If True, probe both the +dx and -dx directions; otherwise only +dx.

    Returns
    -------
    tuple[float, float]
        Bracket (a, b) containing the root, with a < b. If the function is
        already very close to zero at x0, a zero-length bracket (x0, x0) is
        returned for the caller to decide next steps.

    Raises
    ------
    BackendError
        If the root cannot be bracketed within max_expand iterations.
    """

    f0 = float(f(x0))

    # If we are already on (or extremely near) the root, return a zero-length bracket.
    if abs(f0) < 1e-14:
        return (float(x0), float(x0))

    dx = float(dx0)
    for _ in range(int(max_expand)):
        # Probe +dx first (forward propagation)
        xr = float(x0 + dx)
        fr = float(f(xr))
        if crossing_test(f0, fr):
            return (x0, xr) if x0 < xr else (xr, x0)

        if symmetric:
            xl = float(x0 - dx)
            fl = float(f(xl))
            if crossing_test(f0, fl):
                return (xl, x0) if xl < x0 else (x0, xl)

        dx *= float(grow)

    raise BackendError("Failed to bracket root.")


def solve_bracketed_brent(
    f: "Callable[[float], float]",
    a: float,
    b: float,
    *,
    xtol: float = 1e-12,
    max_iter: int = 200,
) -> float | None:
    """Brent-style bracketed scalar root solve in pure Python.

    Returns the root in [a, b] if found; None if the bracket is invalid or
    convergence is not achieved.

    Notes
    -----
    This is a classic Brent variant that blends bisection, secant, and inverse
    quadratic interpolation while preserving the bracket.
    """

    a = float(a)
    b = float(b)
    fa = float(f(a))
    fb = float(f(b))

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return None

    c = a
    fc = fa
    d = b - a
    e = d
    eps = float(np.finfo(np.float64).eps)

    for _ in range(int(max_iter)):
        if fb == 0.0:
            return b

        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = 2.0 * eps * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol:
            return b

        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q_ = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q_ * (q_ - r) - (b - a) * (r - 1.0))
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if (2.0 * p) < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a = b
        fa = fb
        if abs(d) > tol:
            b = b + d
        else:
            b = b + (tol if m > 0.0 else -tol)
        fb = float(f(b))

    tol = 2.0 * eps * abs(b) + 0.5 * xtol
    m = 0.5 * (c - b)
    if abs(m) <= tol or fb == 0.0:
        return b
    return None


