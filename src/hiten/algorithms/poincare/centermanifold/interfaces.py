"""Interface for center manifold Poincare map domain translations.

This module defines a stateless Interface that adapts between domain-level
objects (plane points on a Poincare section; center-manifold coordinates)
and the low-level numerical kernels used by the Backend. It centralises the
logic for:

- Building constraint dictionaries for the energy equation H(q,p) = h0
- Solving for the missing coordinate on a section using root finding
- Lifting plane points to 4D center-manifold states (q2, p2, q3, p3)
"""

import os
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.centermanifold.config import (
    CenterManifoldMapConfig,
)
from hiten.algorithms.poincare.centermanifold.options import (
    CenterManifoldMapOptions,
)
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldBackendRequest,
    CenterManifoldBackendResponse,
    CenterManifoldDomainPayload,
    CenterManifoldMapResults,
    _CenterManifoldMapProblem,
)
from hiten.algorithms.poincare.core.interfaces import (_PoincareBaseInterface,
                                                       _SectionInterface)
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.types.core import _BackendCall
from hiten.algorithms.types.exceptions import BackendError, ConvergenceError
from hiten.algorithms.types.states import RestrictedCenterManifoldState
from hiten.algorithms.utils.rootfinding import solve_bracketed_brent

if TYPE_CHECKING:
    from hiten.system.center import CenterManifold


class _CenterManifoldSectionInterface(_SectionInterface):
    """Stateless section interface for center manifold sections (q2/p2/q3/p3 planes)."""

    @staticmethod
    def get_plane_coords(section_coord: str) -> tuple[str, str]:
        """Get the plane coordinates for a given section coordinate."""
        cfg = _CM_SECTION_TABLE.get(section_coord)
        if cfg is None:
            raise BackendError(f"Unsupported section_coord: {section_coord}")
        return cfg["plane_coords"]

    @staticmethod
    def build_constraint_dict(section_coord: str, **kwargs: float) -> dict[str, float]:
        """Build constraint dictionary for a given section coordinate."""
        out: dict[str, float] = {section_coord: 0.0}
        for k, v in kwargs.items():
            if k in {"q1", "q2", "q3", "p1", "p2", "p3"}:
                out[k] = float(v)
        return out

    @staticmethod
    def build_state(section_coord: str, plane_vals: Tuple[float, float], other_vals: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Build state from plane values and other values for a given section coordinate."""
        plane_coords = _CenterManifoldSectionInterface.get_plane_coords(section_coord)
        
        q2 = p2 = q3 = p3 = 0.0
        if plane_coords == ("q2", "p2"):
            q2, p2 = plane_vals
            q3, p3 = other_vals
        else:
            q3, p3 = plane_vals
            q2, p2 = other_vals
        if section_coord == "q2":
            q2 = 0.0
        elif section_coord == "p2":
            p2 = 0.0
        elif section_coord == "q3":
            q3 = 0.0
        else:  # p3
            p3 = 0.0
        return q2, p2, q3, p3


# Minimal section geometry table for CM planes
_CM_SECTION_TABLE: dict[str, dict[str, object]] = {
    "q3": {"plane_coords": ("q2", "p2")},
    "p3": {"plane_coords": ("q2", "p2")},
    "q2": {"plane_coords": ("q3", "p3")},
    "p2": {"plane_coords": ("q3", "p3")},
}

_STATE_INDEX = {
    "q2": int(RestrictedCenterManifoldState.q2),
    "p2": int(RestrictedCenterManifoldState.p2),
    "q3": int(RestrictedCenterManifoldState.q3),
    "p3": int(RestrictedCenterManifoldState.p3),
}

def _get_plane_coords(section_coord: str) -> tuple[str, str]:
    """Get plane coordinates for a section coordinate."""
    return _CenterManifoldSectionInterface.get_plane_coords(section_coord)


class _CenterManifoldInterface(
    _PoincareBaseInterface[
        CenterManifoldMapConfig,
        _CenterManifoldMapProblem,
        CenterManifoldMapResults,
        CenterManifoldBackendResponse,
    ]
):

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: "CenterManifold",
        config: CenterManifoldMapConfig,
        options: CenterManifoldMapOptions,
    ) -> _CenterManifoldMapProblem:
        # Extract hamiltonian data from the domain object
        hamsys = domain_obj.dynamics.hamsys
        energy = domain_obj.energy
        
        jac_H = hamsys.jac_H
        H_blocks = hamsys.poly_H()
        clmo_table = hamsys.clmo_table
        
        section_coord = config.section_coord
        dt = options.integration.dt
        n_iter = options.iteration.n_iter
        n_workers = options.workers.n_workers
        
        default_workers = os.cpu_count() or 1
        resolved_workers = default_workers if (n_workers is None or int(n_workers) <= 0) else int(n_workers)

        def solve_missing_coord_fn(varname: str, fixed_vals: dict[str, float]) -> Optional[float]:
            return self.solve_missing_coord(
                varname,
                fixed_vals,
                h0=energy,
                H_blocks=H_blocks,
                clmo_table=clmo_table,
            )

        def find_turning_fn(name: str) -> float:
            return self.find_turning(
                name,
                h0=energy,
                H_blocks=H_blocks,
                clmo_table=clmo_table,
            )

        return _CenterManifoldMapProblem(
            section_coord=section_coord,
            energy=float(energy),
            dt=float(dt),
            n_iter=int(n_iter),
            n_workers=resolved_workers,
            jac_H=jac_H,
            H_blocks=H_blocks,
            clmo_table=clmo_table,
            max_steps=options.integration.max_steps,
            method=config.integration.method,
            order=options.integration.order,
            c_omega_heuristic=options.integration.c_omega_heuristic,
            solve_missing_coord_fn=solve_missing_coord_fn,
            find_turning_fn=find_turning_fn,
        )

    def to_backend_inputs(self, problem: _CenterManifoldMapProblem) -> _BackendCall:
        request = CenterManifoldBackendRequest(
            seeds=np.empty((0, 4)),  # seeds are supplied dynamically per iteration
            dt=problem.dt,
            jac_H=problem.jac_H,
            clmo_table=problem.clmo_table,
            section_coord=problem.section_coord,
            max_steps=problem.max_steps,
            method=problem.method,
            order=problem.order,
            c_omega_heuristic=problem.c_omega_heuristic,
        )
        return _BackendCall(request=request)

    def to_domain(self, outputs: CenterManifoldBackendResponse, *, problem: _CenterManifoldMapProblem) -> CenterManifoldDomainPayload:
        points = outputs.states[:, :2] if outputs.states.size else np.empty((0, 2))
        return CenterManifoldDomainPayload._from_mapping(
            {
                "points": points,
                "states": outputs.states,
                "times": outputs.times,
                "labels": self.plane_labels(problem.section_coord),
            }
        )

    def to_results(self, outputs: CenterManifoldBackendResponse, *, problem: _CenterManifoldMapProblem, domain_payload=None) -> CenterManifoldMapResults:
        payload = domain_payload or self.to_domain(outputs, problem=problem)
        return CenterManifoldMapResults(
            payload.points,
            payload.states,
            payload.labels,
            payload.times,
        )

    def create_constraints(self, section_coord: str, **kwargs: float) -> dict[str, float]:
        return _CenterManifoldSectionInterface.build_constraint_dict(section_coord, **kwargs)

    def solve_missing_coord(self, varname: str, fixed_vals: dict[str, float], *, h0: float, H_blocks, clmo_table, initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40, symmetric: bool = False, xtol: float = 1e-12) -> Optional[float]:
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

    def find_turning(self, q_or_p: str, *, h0: float, H_blocks, clmo_table, initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40, symmetric: bool = False, xtol: float = 1e-12) -> float:
        """Find absolute turning point for a CM coordinate.

        Solves for the maximum absolute value of the coordinate where the
        energy constraint can be satisfied, with all other CM coordinates
        set to zero.
        """
        fixed_vals = {coord: 0.0 for coord in ("q2", "p2", "q3", "p3") if coord != q_or_p}
        root = self.solve_missing_coord(
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

    def lift_plane_point(self, plane: Tuple[float, float], *, section_coord: str, h0: float, H_blocks, clmo_table, initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40, symmetric: bool = False, xtol: float = 1e-12) -> Optional[Tuple[float, float, float, float]]:
        """Lift a 2D plane point to a 4D center-manifold state.

        Returns (q2, p2, q3, p3) on the section if solvable; otherwise None.
        """
        plane_coords = _get_plane_coords(section_coord)
        constraints = _CenterManifoldSectionInterface.build_constraint_dict(section_coord, **{
            plane_coords[0]: float(plane[0]),
            plane_coords[1]: float(plane[1]),
        })

        missing_val = self.solve_missing_coord(
            missing_coord := {
                "q3": "p3",
                "p3": "q3",
                "q2": "p2",
                "p2": "q2",
            }[section_coord],
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
        # Infer which other coord was solved based on section_coord
        missing_coord = missing_coord
        other_coords = ("q3", "p3") if plane_coords == ("q2", "p2") else ("q2", "p2")
        idx = 0 if missing_coord == other_coords[0] else 1

        other_vals[idx] = float(missing_val)

        return _CenterManifoldSectionInterface.build_state(section_coord, (float(plane[0]), float(plane[1])), tuple(other_vals))

    def enforce_section_coordinate(self, states: np.ndarray, *, section_coord: str) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float64)
        if arr.size == 0:
            return arr.reshape(0, 4)
        idx = _STATE_INDEX[section_coord]
        out = np.array(arr, copy=True, order="C")
        out[:, idx] = 0.0
        return out

    def plane_points_from_states(self, states: np.ndarray, *, section_coord: str) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        plane_coords = _get_plane_coords(section_coord)
        idx0 = _STATE_INDEX[plane_coords[0]]
        idx1 = _STATE_INDEX[plane_coords[1]]
        return arr[:, (idx0, idx1)]

    def plane_labels(self, section_coord: str) -> tuple[str, str]:
        return _get_plane_coords(section_coord)

