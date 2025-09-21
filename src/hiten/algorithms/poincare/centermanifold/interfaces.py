"""Interface for center manifold Poincare map domain translations.

This module defines a stateless Interface that adapts between domain-level
objects (plane points on a Poincare section; center-manifold coordinates)
and the low-level numerical kernels used by the Backend. It centralises the
logic for:

- Building constraint dictionaries for the energy equation H(q,p) = h0
- Solving for the missing coordinate on a section using root finding
- Lifting plane points to 4D center-manifold states (q2, p2, q3, p3)

The interface exposes pure functions (implemented as @staticmethods) so it is
easy to test and does not carry state. All numerical inputs (Hamiltonian
blocks, CLMO tables, energy) are passed explicitly per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.centermanifold.config import _get_section_config
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.utils.exceptions import BackendError, ConvergenceError


def _solve_bracketed(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-12,
    max_iter: int = 200,
) -> Optional[float]:
    """Brent-style bracketed scalar root solve in pure Python.

    Returns the root in [a, b] if found; None if the bracket is invalid
    or convergence is not achieved.
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


@dataclass(frozen=True)
class _CenterManifoldInterface:
    """Stateless adapter for center manifold section computations.

    Methods accept required numerical inputs explicitly (energy, polynomial
    blocks, CLMO table) and perform domain ↔ backend translations.
    """

    @staticmethod
    def create_constraints(section_coord: str, **kwargs: float) -> dict[str, float]:
        """Create a constraint dict including the section coordinate value."""
        cfg = _get_section_config(section_coord)
        return cfg.build_constraint_dict(**kwargs)

    @staticmethod
    def solve_missing_coord(
        varname: str,
        fixed_vals: dict[str, float],
        *,
        h0: float,
        H_blocks,
        clmo_table,
        initial_guess: float = 1e-3,
        expand_factor: float = 2.0,
        max_expand: int = 40,
        symmetric: bool = False,
        xtol: float = 1e-12,
    ) -> Optional[float]:
        """Solve H(q,p) = h0 for one coordinate given fixed values.

        Returns the coordinate value (root) if a valid bracket is found
        and the root is located; otherwise returns None.
        """
        var_indices = {
            "q1": 0,
            "q2": 1,
            "q3": 2,
            "p1": 3,
            "p2": 4,
            "p3": 5,
        }
        if varname not in var_indices:

            raise BackendError(f"Unknown variable for energy solve: {varname}")

        solve_idx = var_indices[varname]

        def residual(x: float) -> float:
            state = np.zeros(6, dtype=np.complex128)
            for name, val in fixed_vals.items():
                if name in var_indices:
                    state[var_indices[name]] = val
            state[solve_idx] = x
            return _polynomial_evaluate(H_blocks, state, clmo_table).real - h0

        # Require residual(0) <= 0 so a root can lie (0, x]
        if residual(0.0) > 0.0:
            return None

        # Expand a positive-direction bracket first: [0, b]
        a, b = 0.0, float(initial_guess)
        r_b = residual(b)
        n_expand = 0
        while r_b <= 0.0 and n_expand < int(max_expand):
            b *= float(expand_factor)
            r_b = residual(b)
            n_expand += 1

        if r_b > 0.0:
            root = _solve_bracketed(residual, a, b, xtol=xtol, max_iter=200)
            return None if root is None else float(root)

        if symmetric:
            # Try a symmetric negative-direction bracket: [a, 0]
            b_neg = 0.0
            a_neg = -float(initial_guess)
            r_a = residual(a_neg)
            n_expand = 0
            while r_a <= 0.0 and n_expand < int(max_expand):
                a_neg *= float(expand_factor)
                r_a = residual(a_neg)
                n_expand += 1
            if r_a > 0.0:
                root = _solve_bracketed(residual, a_neg, b_neg, xtol=xtol, max_iter=200)
                return None if root is None else float(root)

        return None

    @staticmethod
    def find_turning(
        q_or_p: str,
        *,
        h0: float,
        H_blocks,
        clmo_table,
        initial_guess: float = 1e-3,
        expand_factor: float = 2.0,
        max_expand: int = 40,
        symmetric: bool = False,
        xtol: float = 1e-12,
    ) -> float:
        """Find absolute turning point for a CM coordinate.

        Solves for the maximum absolute value of the coordinate where the
        energy constraint can be satisfied, with all other CM coordinates
        set to zero.
        """
        fixed_vals = {coord: 0.0 for coord in ("q2", "p2", "q3", "p3") if coord != q_or_p}
        root = _CenterManifoldInterface.solve_missing_coord(
            q_or_p,
            fixed_vals,
            h0=h0,
            H_blocks=H_blocks,
            clmo_table=clmo_table,
            initial_guess=initial_guess,
            expand_factor=expand_factor,
            max_expand=max_expand,
            symmetric=symmetric,
            xtol=xtol,
        )
        if root is None:
            raise ConvergenceError(f"Failed to locate turning point for {q_or_p}")
        return float(root)

    @staticmethod
    def lift_plane_point(
        plane: Tuple[float, float],
        *,
        section_coord: str,
        h0: float,
        H_blocks,
        clmo_table,
        initial_guess: float = 1e-3,
        expand_factor: float = 2.0,
        max_expand: int = 40,
        symmetric: bool = False,
        xtol: float = 1e-12,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Lift a 2D plane point to a 4D center-manifold state.

        Returns (q2, p2, q3, p3) on the section if solvable; otherwise None.
        """
        cfg = _get_section_config(section_coord)
        constraints = cfg.build_constraint_dict(**{
            cfg.plane_coords[0]: float(plane[0]),
            cfg.plane_coords[1]: float(plane[1]),
        })

        missing_val = _CenterManifoldInterface.solve_missing_coord(
            cfg.missing_coord,
            constraints,
            h0=h0,
            H_blocks=H_blocks,
            clmo_table=clmo_table,
            initial_guess=initial_guess,
            expand_factor=expand_factor,
            max_expand=max_expand,
            symmetric=symmetric,
            xtol=xtol,
        )

        if missing_val is None:
            return None

        other_vals = [0.0, 0.0]
        idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
        other_vals[idx] = float(missing_val)

        return cfg.build_state((float(plane[0]), float(plane[1])), tuple(other_vals))
