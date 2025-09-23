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
import os
from typing import Callable, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.centermanifold.config import _get_section_config
from hiten.algorithms.poincare.centermanifold.types import _CenterManifoldMapProblem
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.utils.exceptions import BackendError, ConvergenceError
from hiten.algorithms.utils.rootfinding import solve_bracketed_brent


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
            root = solve_bracketed_brent(residual, a, b, xtol=xtol, max_iter=200)
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
                root = solve_bracketed_brent(residual, a_neg, b_neg, xtol=xtol, max_iter=200)
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

    @staticmethod
    def create_problem(section_coord: str, energy: float, *, dt: float, n_iter: int, n_workers: int | None) -> _CenterManifoldMapProblem:
        default_workers = os.cpu_count() or 1
        resolved_workers = (
            default_workers if (n_workers is None or int(n_workers) <= 0) else int(n_workers)
        )
        return _CenterManifoldMapProblem(
            section_coord=section_coord,
            energy=float(energy),
            dt=float(dt),
            n_iter=int(n_iter),
            n_workers=resolved_workers,
        )
