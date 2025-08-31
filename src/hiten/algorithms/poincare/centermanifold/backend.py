from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numba import njit, prange
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.hamiltonian import (_eval_dH_dP, _eval_dH_dQ,
                                                   _hamiltonian_rhs,
                                                   _HamiltonianSystemProtocol)
from hiten.algorithms.integrators.rk import (RK4_A, RK4_B, RK4_C, RK6_A, RK6_B,
                                             RK6_C, RK8_A, RK8_B, RK8_C)
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                                     _integrate_symplectic)
from hiten.algorithms.poincare.centermanifold.config import _get_section_config
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import _SurfaceEvent
from hiten.algorithms.poincare.utils import _hermite_scalar
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _detect_crossing(section_coord: str, state_old: np.ndarray, state_new: np.ndarray, rhs_new: np.ndarray, n_dof: int) -> Tuple[bool, float]:
    if section_coord == "q3":
        f_old = state_old[2]
        f_new = state_new[2]
    elif section_coord == "p3":
        f_old = state_old[n_dof + 2]
        f_new = state_new[n_dof + 2]
    elif section_coord == "q2":
        f_old = state_old[1]
        f_new = state_new[1]
    else:  # "p2"
        f_old = state_old[n_dof + 1]
        f_new = state_new[n_dof + 1]

    # Must have sign change
    if f_old * f_new >= 0.0:
        return False, 0.0

    # Direction check
    if section_coord == "q3":
        good_dir = state_new[n_dof + 2] > 0.0
    elif section_coord == "q2":
        good_dir = state_new[n_dof + 1] > 0.0
    elif section_coord == "p3":
        good_dir = rhs_new[2] > 0.0
    else:  # "p2"
        good_dir = rhs_new[1] > 0.0

    if not good_dir:
        return False, 0.0

    alpha = f_old / (f_old - f_new)
    return True, alpha


@njit(cache=False, fastmath=FASTMATH)
def _get_rk_coefficients(order: int):
    if order == 4:
        return RK4_A, RK4_B, RK4_C
    elif order == 6:
        return RK6_A, RK6_B, RK6_C
    else:
        return RK8_A, RK8_B, RK8_C


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_ham(y0: np.ndarray, t_vals: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, jac_H, clmo_H):
    n_steps = t_vals.shape[0]
    dim = y0.shape[0]
    n_stages = B.shape[0]
    traj = np.empty((n_steps, dim), dtype=np.float64)
    traj[0, :] = y0.copy()

    k = np.empty((n_stages, dim), dtype=np.float64)

    n_dof = dim // 2

    for step in range(n_steps - 1):
        t_n = t_vals[step]
        h = t_vals[step + 1] - t_n

        y_n = traj[step].copy()

        for s in range(n_stages):
            y_stage = y_n.copy()
            for j in range(s):
                a_sj = A[s, j]
                if a_sj != 0.0:
                    y_stage += h * a_sj * k[j]

            Q = y_stage[0:n_dof]
            P = y_stage[n_dof: 2 * n_dof]

            dQ = _eval_dH_dP(Q, P, jac_H, clmo_H)
            dP = -_eval_dH_dQ(Q, P, jac_H, clmo_H)

            k[s, 0:n_dof] = dQ
            k[s, n_dof: 2 * n_dof] = dP

        y_np1 = y_n.copy()
        for s in range(n_stages):
            b_s = B[s]
            if b_s != 0.0:
                y_np1 += h * b_s * k[s]

        traj[step + 1] = y_np1

    return traj


@njit(cache=False, fastmath=FASTMATH)
def _integrate_map(y0: np.ndarray, t_vals: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                   jac_H, clmo_H, order: int, c_omega_heuristic: float = 20.0, use_symplectic: bool = False):

    if use_symplectic:
        traj = _integrate_symplectic(y0, t_vals, jac_H, clmo_H, order, c_omega_heuristic)
    else:
        traj = _integrate_rk_ham(y0, t_vals, A, B, C, jac_H, clmo_H)

    return traj


@njit(cache=False, fastmath=FASTMATH)
def _poincare_step(q2: float, p2: float, q3: float, p3: float, dt: float,
                   jac_H, clmo, order: int, max_steps: int, use_symplectic: bool,
                   n_dof: int, section_coord: str, c_omega_heuristic: float = 20.0):

    state_old = np.zeros(2 * n_dof, dtype=np.float64)
    state_old[1] = q2
    state_old[2] = q3
    state_old[n_dof + 1] = p2
    state_old[n_dof + 2] = p3

    elapsed = 0.0
    for _ in range(max_steps):
        c_A, c_B, c_C = _get_rk_coefficients(order)
        traj = _integrate_map(y0=state_old, t_vals=np.array([0.0, dt]), A=c_A, B=c_B, C=c_C,
                              jac_H=jac_H, clmo_H=clmo, order=order,
                              c_omega_heuristic=c_omega_heuristic, use_symplectic=use_symplectic)
        state_new = traj[1]

        rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
        crossed, alpha = _detect_crossing(section_coord, state_old, state_new, rhs_new, n_dof)

        if crossed:
            rhs_old = _hamiltonian_rhs(state_old, jac_H, clmo, n_dof)

            q2p = _hermite_scalar(alpha, state_old[1],       state_new[1],       rhs_old[1],       rhs_new[1],       dt)
            p2p = _hermite_scalar(alpha, state_old[n_dof+1], state_new[n_dof+1], rhs_old[n_dof+1], rhs_new[n_dof+1], dt)
            q3p = _hermite_scalar(alpha, state_old[2],       state_new[2],       rhs_old[2],       rhs_new[2],       dt)
            p3p = _hermite_scalar(alpha, state_old[n_dof+2], state_new[n_dof+2], rhs_old[n_dof+2], rhs_new[n_dof+2], dt)

            t_cross = elapsed + alpha * dt
            return 1, q2p, p2p, q3p, p3p, t_cross

        state_old = state_new
        elapsed += dt

    return 0, 0.0, 0.0, 0.0, 0.0, 0.0


@njit(parallel=True, cache=False)
def _poincare_map(seeds: np.ndarray, dt: float, jac_H, clmo, order: int, max_steps: int,
                  use_symplectic: bool, n_dof: int, section_coord: str, c_omega_heuristic: float):

    n_seeds = seeds.shape[0]
    success = np.zeros(n_seeds, dtype=np.int64)
    q2p_out = np.empty(n_seeds, dtype=np.float64)
    p2p_out = np.empty(n_seeds, dtype=np.float64)
    q3p_out = np.empty(n_seeds, dtype=np.float64)
    p3p_out = np.empty(n_seeds, dtype=np.float64)
    t_out = np.empty(n_seeds, dtype=np.float64)

    for i in prange(n_seeds):
        q2, p2, q3, p3 = seeds[i, 0], seeds[i, 1], seeds[i, 2], seeds[i, 3]

        flag, q2_new, p2_new, q3_new, p3_new, t_cross = _poincare_step(
            q2, p2, q3, p3, dt, jac_H, clmo, order, max_steps, use_symplectic,
            n_dof, section_coord, c_omega_heuristic
        )

        if flag == 1:
            success[i] = 1
            q2p_out[i] = q2_new
            p2p_out[i] = p2_new
            q3p_out[i] = q3_new
            p3p_out[i] = p3_new
            t_out[i] = t_cross

    return success, q2p_out, p2p_out, q3p_out, p3p_out, t_out


class _CenterManifoldBackend(_ReturnMapBackend):
    """Backend that delegates computation to the fast Centre-Manifold kernels.

    Parameters
    ----------
    generate_map_kwargs
        Arguments forwarded verbatim to
        :func:`hiten.algorithms.poincare.map._generate_map`.
    generate_grid_kwargs, optional
        Arguments forwarded to
        :func:`hiten.algorithms.poincare.map._generate_grid`.  If *None* the
        backend will raise ``NotImplementedError`` when
        :py:meth:`compute_grid` is requested.
    """

    def __init__(
        self,
        *,
        dynsys: "_HamiltonianSystemProtocol",
        surface: "_SurfaceEvent",
        section_coord: str,
        h0: float,
        forward: int = 1,
        max_steps: int = 2000,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
        order: int = 8,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
        c_omega_heuristic: float = 20.0,
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

        self._section_cfg = _get_section_config(section_coord)
        self._h0 = h0
        self._H_blocks = dynsys.poly_H()
        self._clmo_table = dynsys.clmo_table
        self._jac_H = dynsys.jac_H
        self._order = order
        self._max_steps = max_steps
        self._use_symplectic = method == "symplectic"
        self._n_dof = N_SYMPLECTIC_DOF
        self._c_omega_heuristic = c_omega_heuristic

    def _lift_plane_point(self, plane: tuple[float, float]) -> Optional[tuple[float, float, float, float]]:
        """Convert a 2-D plane point into a 4-tuple CM state if inside Hill box."""

        cfg = self._section_cfg

        constraints = cfg.build_constraint_dict(**{
            cfg.plane_coords[0]: plane[0],
            cfg.plane_coords[1]: plane[1],
        })

        # _solve_missing_coord is a bound method; it already knows energy and
        # Hamiltonian data via self.  Passing them again shifts positional
        # parameters and breaks the signature.  Only provide the variable
        # name and the dict of fixed values.
        missing_val = self._solve_missing_coord(cfg.missing_coord, constraints)

        if missing_val is None:
            return None

        other_vals = [0.0, 0.0]
        idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
        other_vals[idx] = missing_val

        return cfg.build_state(plane, tuple(other_vals))

    def _bracket_root(
        self,
        f: Callable[[float], float],
        initial: float = 1e-3,
        factor: float = 2.0,
        max_expand: int = 40,
        *,
        symmetric: bool = False,
        method: str = "brentq",
        xtol: float = 1e-12,
    ) -> Optional[float]:
        """Return a positive root of *f* bracketed between 0 and some *x_hi*."""

        # We require f(0) <= 0 so that a root can lie in (0, x).
        if f(0.0) > 0.0:
            return None

        try:
            a, b = self._expand_bracket(
                f,
                0.0,
                dx0=initial,
                grow=factor,
                max_expand=max_expand,
                crossing_test=lambda prev, curr: prev <= 0.0 < curr,
                symmetric=symmetric,
            )
        except RuntimeError:
            return None

        sol = root_scalar(f, bracket=(a, b), method=method, xtol=xtol)
        return float(sol.root) if sol.converged else None

    def _solve_missing_coord(
        self,
        varname: str,
        fixed_vals: dict[str, float],
        initial_guess: float = 1e-3,
        expand_factor: float = 2.0,
        max_expand: int = 40,
        *,
        symmetric: bool = False,
        method: str = "brentq",
        xtol: float = 1e-12,
    ) -> Optional[float]:
        """Solve for the *turning-point* value of one CM coordinate.
        """

        var_indices = {
            "q1": 0, "q2": 1, "q3": 2,
            "p1": 3, "p2": 4, "p3": 5,
        }

        if varname not in var_indices:
            raise ValueError(f"Unknown variable: {varname}")

        solve_idx = var_indices[varname]

        def _residual(x: float) -> float:
            state = np.zeros(6, dtype=np.complex128)

            # Apply fixed values.
            for name, val in fixed_vals.items():
                if name in var_indices:
                    state[var_indices[name]] = val

            state[solve_idx] = x

            return _polynomial_evaluate(self._H_blocks, state, self._clmo_table).real - self._h0

        root = self._bracket_root(
            _residual,
            initial=initial_guess,
            factor=expand_factor,
            max_expand=max_expand,
            symmetric=symmetric,
            method=method,
            xtol=xtol,
        )

        if root is None:
            logger.warning("Failed to locate %s turning point within search limits", varname)
            return None

        return root

    def _find_turning(
        self,
        q_or_p: str,
        initial_guess: float = 1e-3,
        expand_factor: float = 2.0,
        max_expand: int = 40,
        *,
        symmetric: bool = False,
        method: str = "brentq",
        xtol: float = 1e-12,
    ) -> float:
        """Return the absolute turning-point value for *q_or_p* coordinate."""

        fixed_vals = {coord: 0.0 for coord in ("q2", "p2", "q3", "p3") if coord != q_or_p}

        root = self._solve_missing_coord(
            q_or_p,
            fixed_vals,
            initial_guess,
            expand_factor,
            max_expand,
            symmetric=symmetric,
            method=method,
            xtol=xtol,
        )

        if root is None:
            logger.warning("Failed to locate %s turning point within search limits", q_or_p)
            raise RuntimeError("Root finding for Hill boundary did not converge.")

        return root

    def step_to_section(
        self,
        seeds: np.ndarray,
        *,
        dt: float = 1e-2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagate every CM seed until the next section crossing.

        Parameters
        ----------
        seeds
            Array of shape (m, 4) with (q2, p2, q3, p3) states.

        Returns
        -------
        pts : (k, 2) ndarray
            Plane coordinates of the crossings.
        cm_states : (k, 4) ndarray
            Centre-manifold coordinates of the crossings.
        """

        if seeds.size == 0:
            return (
                np.empty((0, 2)),
                np.empty((0, 4)),
                np.empty((0,)),
                np.empty((0,), dtype=np.int64),
            )

        max_steps = int(np.ceil(20.0 / dt))

        flags, q2p_arr, p2p_arr, q3p_arr, p3p_arr, t_arr = _poincare_map(
            np.ascontiguousarray(seeds, dtype=np.float64),
            dt,
            self._jac_H,
            self._clmo_table,
            self._order,
            max_steps,
            self._use_symplectic,
            N_SYMPLECTIC_DOF,
            self._section_cfg.section_coord,
            self._c_omega_heuristic,
        )

        cfg = self._section_cfg
        pts_list: list[tuple[float, float]] = []
        states_list: list[tuple[float, float, float, float]] = []
        times_list: list[float] = []

        for i in range(flags.shape[0]):
            if flags[i]:
                state = (q2p_arr[i], p2p_arr[i], q3p_arr[i], p3p_arr[i])
                # Ensure the returned seed lies EXACTLY on the section plane so that
                # the following iteration does not detect the same crossing again.
                # We zero the coordinate that defines the section (q2, p2, q3 or p3)
                # to avoid re-registering the same hit in subsequent steps.
                if cfg.section_coord == "q3":
                    state = (state[0], state[1], 0.0, state[3])
                elif cfg.section_coord == "p3":
                    state = (state[0], state[1], state[2], 0.0)
                elif cfg.section_coord == "q2":
                    state = (0.0, state[1], state[2], state[3])
                else:  # "p2"
                    state = (state[0], 0.0, state[2], state[3])
                states_list.append(state)
                if cfg.plane_coords == ("q2", "p2"):
                    pts_list.append((state[0], state[1]))
                else:
                    pts_list.append((state[2], state[3]))
                times_list.append(float(t_arr[i]))

        return (
            np.asarray(pts_list, dtype=np.float64),
            np.asarray(states_list, dtype=np.float64),
            np.asarray(times_list, dtype=np.float64),
            np.asarray(flags, dtype=np.int64),
        )
