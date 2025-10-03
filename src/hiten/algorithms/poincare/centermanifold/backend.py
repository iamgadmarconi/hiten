"""Center manifold backend for efficient Poincare map computations.

This module provides Numba-compiled kernels for computing center manifold
trajectories in the Circular Restricted Three-Body Problem (CR3BP). The
backend implements both Runge-Kutta and symplectic integration methods
with parallel processing capabilities for high-performance computation
of Poincare maps.

The main class :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend` 
provides the interface for center manifold computations, while the Numba-compiled functions
handle the low-level numerical integration and section crossing detection.
"""

from typing import Literal, Tuple

import numpy as np
from numba import njit, prange

from hiten.algorithms.dynamics.hamiltonian import (_eval_dH_dP, _eval_dH_dQ,
                                                   _hamiltonian_rhs)
from hiten.algorithms.integrators.rk import (RK4_A, RK4_B, RK4_C, RK6_A, RK6_B,
                                             RK6_C, RK8_A, RK8_B, RK8_C)
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                                     _integrate_symplectic)
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldBackendRequest,
    CenterManifoldBackendResponse,
)
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.utils import _hermite_scalar
from hiten.algorithms.utils.config import FASTMATH


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _detect_crossing(section_coord: str, state_old: np.ndarray, state_new: np.ndarray, rhs_new: np.ndarray, n_dof: int) -> Tuple[bool, float]:
    """Detect if trajectory crossed the Poincare section.

    Parameters
    ----------
    section_coord : str
        Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').
    state_old : ndarray, shape (2*n_dof,)
        Previous state vector [q1, q2, q3, p1, p2, p3].
    state_new : ndarray, shape (2*n_dof,)
        Current state vector [q1, q2, q3, p1, p2, p3].
    rhs_new : ndarray, shape (2*n_dof,)
        Current right-hand side of Hamiltonian equations.
    n_dof : int
        Number of degrees of freedom (typically 3 for CR3BP).

    Returns
    -------
    bool
        True if section crossing detected.
    float
        Interpolation parameter alpha for crossing time.
    """
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
    """Return Runge-Kutta coefficients for specified order.

    Parameters
    ----------
    order : int
        Integration order (4, 6, or 8).

    Returns
    -------
    A : ndarray
        Runge-Kutta A matrix.
    B : ndarray
        Runge-Kutta B vector.
    C : ndarray
        Runge-Kutta C vector.
    """
    if order == 4:
        return RK4_A, RK4_B, RK4_C
    elif order == 6:
        return RK6_A, RK6_B, RK6_C
    else:
        return RK8_A, RK8_B, RK8_C


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_ham(y0: np.ndarray, t_vals: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, jac_H, clmo_H):
    """Integrate Hamiltonian system using Runge-Kutta method.

    Parameters
    ----------
    y0 : ndarray, shape (2*n_dof,)
        Initial state vector [q1, q2, q3, p1, p2, p3].
    t_vals : ndarray, shape (n_steps,)
        Time values for integration.
    A : ndarray
        Runge-Kutta A matrix.
    B : ndarray
        Runge-Kutta B vector.
    C : ndarray
        Runge-Kutta C vector.
    jac_H : ndarray
        Jacobian of Hamiltonian polynomial.
    clmo_H : ndarray
        CLMO table for polynomial evaluation.

    Returns
    -------
    traj : ndarray, shape (n_steps, 2*n_dof)
        Integrated trajectory.
    """
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
    """Integrate Hamiltonian system using specified method.

    Parameters
    ----------
    y0 : ndarray, shape (2*n_dof,)
        Initial state vector [q1, q2, q3, p1, p2, p3].
    t_vals : ndarray, shape (n_steps,)
        Time values for integration.
    A : ndarray
        Runge-Kutta A matrix.
    B : ndarray
        Runge-Kutta B vector.
    C : ndarray
        Runge-Kutta C vector.
    jac_H : ndarray
        Jacobian of Hamiltonian polynomial.
    clmo_H : ndarray
        CLMO table for polynomial evaluation.
    order : int
        Integration order.
    c_omega_heuristic : float, default=20.0
        Heuristic parameter for symplectic integration.
    use_symplectic : bool, default=False
        If True, use symplectic integration; otherwise use Runge-Kutta.

    Returns
    -------
    traj : ndarray, shape (n_steps, 2*n_dof)
        Integrated trajectory.
    """

    if use_symplectic:
        traj = _integrate_symplectic(y0, t_vals, jac_H, clmo_H, order, c_omega_heuristic)
    else:
        traj = _integrate_rk_ham(y0, t_vals, A, B, C, jac_H, clmo_H)

    return traj


@njit(cache=False, fastmath=FASTMATH)
def _poincare_step(q2: float, p2: float, q3: float, p3: float, dt: float,
                   jac_H, clmo, order: int, max_steps: int, use_symplectic: bool,
                   n_dof: int, section_coord: str, c_omega_heuristic: float = 20.0):
    """Perform one Poincare map step for center manifold integration.

    Parameters
    ----------
    q2 : float
        q2 coordinate (nondimensional units).
    p2 : float
        p2 coordinate (nondimensional units).
    q3 : float
        q3 coordinate (nondimensional units).
    p3 : float
        p3 coordinate (nondimensional units).
    dt : float
        Integration time step (nondimensional units).
    jac_H : ndarray
        Jacobian of Hamiltonian polynomial.
    clmo : ndarray
        CLMO table for polynomial evaluation.
    order : int
        Integration order.
    max_steps : int
        Maximum number of integration steps.
    use_symplectic : bool
        If True, use symplectic integration.
    n_dof : int
        Number of degrees of freedom.
    section_coord : str
        Section coordinate identifier.
    c_omega_heuristic : float, default=20.0
        Heuristic parameter for symplectic integration.

    Returns
    -------
    flag : int
        1 if section crossing found, 0 otherwise.
    q2p : float
        q2 coordinate at crossing.
    p2p : float
        p2 coordinate at crossing.
    q3p : float
        q3 coordinate at crossing.
    p3p : float
        p3 coordinate at crossing.
    t_cross : float
        Time of section crossing.
    """

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
    """Compute Poincare map for multiple center manifold seeds in parallel.

    Parameters
    ----------
    seeds : ndarray, shape (n_seeds, 4)
        Array of initial seeds [q2, p2, q3, p3] (nondimensional units).
    dt : float
        Integration time step (nondimensional units).
    jac_H : ndarray
        Jacobian of Hamiltonian polynomial.
    clmo : ndarray
        CLMO table for polynomial evaluation.
    order : int
        Integration order.
    max_steps : int
        Maximum number of integration steps.
    use_symplectic : bool
        If True, use symplectic integration.
    n_dof : int
        Number of degrees of freedom.
    section_coord : str
        Section coordinate identifier.
    c_omega_heuristic : float
        Heuristic parameter for symplectic integration.

    Returns
    -------
    success : ndarray, shape (n_seeds,)
        Success flags (1 if crossing found, 0 otherwise).
    q2p_out : ndarray, shape (n_seeds,)
        q2 coordinates at crossings (or zeros for failed seeds).
    p2p_out : ndarray, shape (n_seeds,)
        p2 coordinates at crossings (or zeros for failed seeds).
    q3p_out : ndarray, shape (n_seeds,)
        q3 coordinates at crossings (or zeros for failed seeds).
    p3p_out : ndarray, shape (n_seeds,)
        p3 coordinates at crossings (or zeros for failed seeds).
    t_out : ndarray, shape (n_seeds,)
        Times of section crossings (or zeros for failed seeds).
    """

    n_seeds = seeds.shape[0]
    success = np.zeros(n_seeds, dtype=np.int64)
    q2p_out = np.zeros(n_seeds, dtype=np.float64)
    p2p_out = np.zeros(n_seeds, dtype=np.float64)
    q3p_out = np.zeros(n_seeds, dtype=np.float64)
    p3p_out = np.zeros(n_seeds, dtype=np.float64)
    t_out = np.zeros(n_seeds, dtype=np.float64)

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
    """Backend for center manifold computations in the CR3BP.

    This backend provides efficient computation of center manifold trajectories
    using Numba-compiled kernels for Hamiltonian integration and Poincare map
    evaluation. It supports both Runge-Kutta and symplectic integration methods.

    Notes
    -----
    State vectors are ordered as [q1, q2, q3, p1, p2, p3].
    The backend is stateless - all dynamic system data must be passed to
    the step_to_section method as arguments.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def run(
        self,
        request: CenterManifoldBackendRequest,
    ) -> CenterManifoldBackendResponse:
        """Propagate center manifold seeds until the next Poincare section crossing.

        Parameters
        ----------
        request : :class:`~hiten.algorithms.poincare.centermanifold.types.CenterManifoldBackendRequest`
            Structured request containing seeds and integration parameters.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.types.CenterManifoldBackendResponse`
            Structured response containing center manifold states, times, flags, and metadata.

        Notes
        -----
        This method uses parallel Numba compilation for efficient computation
        of multiple trajectories. The integration continues until either a
        section crossing is detected or the maximum number of steps is reached.
        """

        if request.seeds.size == 0:
            return CenterManifoldBackendResponse(
                states=np.empty((0, 4)),
                times=np.empty((0,)),
                flags=np.empty((0,), dtype=np.int64),
                metadata={},
            )

        # Guard against unsupported adaptive label
        if request.method == "adaptive":
            raise NotImplementedError("Adaptive integrator is not implemented in CM backend; use 'fixed' (RK) or 'symplectic'.")

        flags, q2p_arr, p2p_arr, q3p_arr, p3p_arr, t_arr = _poincare_map(
            np.ascontiguousarray(request.seeds, dtype=np.float64),
            request.dt,
            request.jac_H,
            request.clmo_table,
            request.order,
            request.max_steps,
            request.method == "symplectic",
            N_SYMPLECTIC_DOF,
            request.section_coord,
            request.c_omega_heuristic,
        )

        states_list: list[tuple[float, float, float, float]] = []
        times_list: list[float] = []

        for i in range(flags.shape[0]):
            if flags[i]:
                state = (q2p_arr[i], p2p_arr[i], q3p_arr[i], p3p_arr[i])
                states_list.append(state)
                times_list.append(float(t_arr[i]))

        return CenterManifoldBackendResponse(
            states=np.asarray(states_list, dtype=np.float64),
            times=np.asarray(times_list, dtype=np.float64),
            flags=np.asarray(flags, dtype=np.int64),
            metadata={},
        )
