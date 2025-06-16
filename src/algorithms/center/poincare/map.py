import math
from typing import Callable, List, Optional, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.optimize import root_scalar

from algorithms.center.polynomial.operations import (polynomial_evaluate,
                                                     polynomial_jacobian)
from algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                               integrate_symplectic)
from config import FASTMATH
from utils.log_config import logger


def _bracketed_root(
    f: Callable[[float], float],
    initial: float = 1e-3,
    factor: float = 2.0,
    max_expand: int = 40,
    xtol: float = 1e-12,
) -> Optional[float]:
    """Return a positive root of *f* if a sign change can be bracketed.

    The routine starts from ``x=0`` and expands the upper bracket until the
    function changes sign.  If no sign change occurs within
    ``initial * factor**max_expand`` it returns ``None``.
    """
    # Early exit if already above root at x=0 ⇒ no positive solution.
    if f(0.0) > 0.0:
        return None

    x_hi = initial
    for _ in range(max_expand):
        if f(x_hi) > 0.0:
            sol = root_scalar(f, bracket=(0.0, x_hi), method="brentq", xtol=xtol)
            return float(sol.root) if sol.converged else None
        x_hi *= factor

    # No sign change detected within the expansion range
    return None

def _find_turning(
    q_or_p: str,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo: List[np.ndarray],
    initial_guess: float = 1e-3,
    expand_factor: float = 2.0,
    max_expand: int = 40,
) -> float:
    """
    Return the positive intercept q2_max or p2_max of the Hill boundary.

    Parameters
    ----------
    q_or_p : str
        "q2" or "p2" specifying which variable to solve for
    h0 : float
        Energy level (centre-manifold value)
    H_blocks : List[np.ndarray]
        Polynomial coefficients of H restricted to the CM
    clmo : List[np.ndarray]
        CLMO index table matching *H_blocks*
    initial_guess : float, optional
        Initial guess for bracketing procedure, by default 1e-3
    expand_factor : float, optional
        Factor for expanding the bracket, by default 2.0
    max_expand : int, optional
        Maximum number of expansions to try, by default 40
        
    Returns
    -------
    float
        The positive intercept value
        
    Raises
    ------
    ValueError
        If q_or_p is not 'q2' or 'p2'
    RuntimeError
        If root finding fails or doesn't converge
    """
    logger.info(f"Finding {q_or_p} turning point at energy h0={h0:.6e}")
    
    if q_or_p not in {"q2", "p2"}:
        raise ValueError("q_or_p must be 'q2' or 'p2'.")

    def f(x: float) -> float:
        state = np.zeros(6, dtype=np.complex128)
        if q_or_p == "q2":
            state[1] = x  # q2
        else:
            state[4] = x  # p2 (index 4 in (q1,q2,q3,p1,p2,p3))
        return polynomial_evaluate(H_blocks, state, clmo).real - h0

    root = _bracketed_root(
        f,
        initial=initial_guess,
        factor=expand_factor,
        max_expand=max_expand,
    )

    if root is None:
        logger.warning("Failed to locate %s turning point within search limits", q_or_p)
        raise RuntimeError("Root finding for Hill boundary did not converge.")

    logger.info("Found %s turning point: %.6e", q_or_p, root)
    return root

def _solve_p3(
    q2: float,
    p2: float,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo: List[np.ndarray],
    initial_guess: float = 1e-3,
    expand_factor: float = 2.0,
    max_expand: int = 40,
) -> Optional[float]:
    """
    Solve H(q2,p2,0,p3)=h0 for p3>0.
    
    Parameters
    ----------
    q2 : float
        Second position coordinate
    p2 : float
        Second momentum coordinate
    h0 : float
        Energy level
    H_blocks : List[np.ndarray]
        Polynomial coefficients
    clmo : List[np.ndarray]
        CLMO index table
    initial_guess : float, optional
        Initial guess for bracketing procedure, by default 1e-3
    expand_factor : float, optional
        Factor for expanding the bracket, by default 2.0
    max_expand : int, optional
        Maximum number of expansions to try, by default 40
        
    Returns
    -------
    Optional[float]
        Positive p3 solution if exists, None otherwise
    """
    logger.info(f"Solving for p3 at (q2,p2)=({q2:.4e},{p2:.4e}), h0={h0:.6e}")
    
    def f(p3: float) -> float:
        state = np.zeros(6, dtype=np.complex128)
        state[1] = q2
        state[4] = p2
        state[5] = p3
        return polynomial_evaluate(H_blocks, state, clmo).real - h0

    root = _bracketed_root(f, initial=initial_guess, factor=expand_factor, max_expand=max_expand)

    if root is None:
        logger.warning("Failed to locate p3 turning point within search limits")
        raise RuntimeError("Root finding for Hill boundary did not converge.")

    if root < 0.0:
        logger.warning("Found negative p3 solution: %.6e", root)
        return None

    logger.info("Found p3 turning point: %.6e", root)
    return root

@njit(cache=True, fastmath=FASTMATH)
def _hamiltonian_rhs(
    state6: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    n_dof: int,
) -> np.ndarray:
    """Compute time derivative (Qdot, Pdot) for the 2*n_dof Hamiltonian system."""

    dH_dQ = np.empty(n_dof)
    dH_dP = np.empty(n_dof)

    for i in range(n_dof):
        dH_dQ[i] = polynomial_evaluate(jac_H[i], state6.astype(np.complex128), clmo).real
        dH_dP[i] = polynomial_evaluate(jac_H[n_dof + i], state6.astype(np.complex128), clmo).real

    rhs = np.empty_like(state6)
    rhs[:n_dof] = dH_dP  # dq/dt
    rhs[n_dof : 2 * n_dof] = -dH_dQ  # dp/dt
    return rhs

@njit(fastmath=FASTMATH, cache=True)
def _rk4_step(
    state6: np.ndarray,
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    n_dof: int,
) -> np.ndarray:
    """Single RK4 step for the Hamiltonian ODE."""
    k1 = _hamiltonian_rhs(state6, jac_H, clmo, n_dof)
    k2 = _hamiltonian_rhs(state6 + 0.5 * dt * k1, jac_H, clmo, n_dof)
    k3 = _hamiltonian_rhs(state6 + 0.5 * dt * k2, jac_H, clmo, n_dof)
    k4 = _hamiltonian_rhs(state6 + dt * k3, jac_H, clmo, n_dof)
    return state6 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=True, fastmath=FASTMATH)
def _poincare_step_jit(
    q2: float,
    p2: float,
    p3: float,
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
    c_omega_heuristic: float=20.0,
) -> Tuple[int, float, float, float]:
    """Return (flag, q2', p2', p3').  flag=1 if success, 0 otherwise."""
    # Build the centre-manifold state directly (replaces _embed_cm_state_jit)
    state_old = np.zeros(2 * n_dof, dtype=np.float64)
    state_old[1] = q2
    state_old[2] = 0.0  # q3 always zero on the section
    state_old[n_dof + 1] = p2
    state_old[n_dof + 2] = p3

    for _ in range(max_steps):
        if use_symplectic:
            traj = integrate_symplectic(
                initial_state_6d=state_old,
                t_values=np.array([0.0, dt]),
                jac_H_rn_typed=jac_H,
                clmo_H_typed=clmo,
                order=order,
                c_omega_heuristic=c_omega_heuristic,
            )
            state_new = traj[1]
        else:
            state_new = _rk4_step(state_old, dt, jac_H, clmo, n_dof)

        q3_old = state_old[2]
        q3_new = state_new[2]
        p3_old = state_old[n_dof + 2]
        p3_new = state_new[n_dof + 2]

        if (q3_old * q3_new < 0.0) and (p3_new > 0.0):

            # 1) linear first guess
            alpha = q3_old / (q3_old - q3_new)

            # 2) endpoint derivatives for Hermite poly (need dt-scaled slopes)
            rhs_old = _hamiltonian_rhs(state_old, jac_H, clmo, n_dof)
            rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
            m0 = rhs_old[2] * dt          # dq3/dt at t=0  → slope * dt
            m1 = rhs_new[2] * dt          # dq3/dt at t=dt

            # 3) cubic Hermite coefficients  H(t) = a t³ + b t² + c t + d   ( 0 ≤ t ≤ 1 )
            d  = q3_old
            c  = m0
            b  = 3.0*(q3_new - q3_old) - (2.0*m0 +   m1)
            a  = 2.0*(q3_old - q3_new) + (   m0 +   m1)

            # 4) one Newton iteration on H(t)=0  (enough because linear guess is very close)
            f  = ((a*alpha + b)*alpha + c)*alpha + d
            fp = (3.0*a*alpha + 2.0*b)*alpha + c        # derivative
            alpha -= f / fp
            # clamp in case numerical noise pushed it slightly outside
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0

            # 5) use *the same* cubic basis to interpolate q₂, p₂, p₃
            h00 = (1.0 + 2.0*alpha) * (1.0 - alpha)**2
            h10 = alpha * (1.0 - alpha)**2
            h01 = alpha**2 * (3.0 - 2.0*alpha)
            h11 = alpha**2 * (alpha - 1.0)

            def hermite(y0, y1, dy0, dy1):
                return (
                    h00 * y0 +
                    h10 * dy0 * dt +
                    h01 * y1 +
                    h11 * dy1 * dt
                )

            q2p = hermite(state_old[1], state_new[1], rhs_old[1], rhs_new[1])
            p2p = hermite(state_old[n_dof+1], state_new[n_dof+1],
                          rhs_old[n_dof+1],    rhs_new[n_dof+1])
            p3p = hermite(p3_old, p3_new, rhs_old[n_dof+2], rhs_new[n_dof+2])

            return 1, q2p, p2p, p3p

        state_old = state_new

    return 0, 0.0, 0.0, 0.0

@njit(parallel=True, cache=True)
def _poincare_map_parallel(
    seeds: np.ndarray,  # (N,4) float64
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (success flags, q2p array, p2p array, p3p array) processed in parallel."""
    n_seeds = seeds.shape[0]
    success = np.zeros(n_seeds, dtype=np.int64)
    q2p_out = np.empty(n_seeds, dtype=np.float64)
    p2p_out = np.empty(n_seeds, dtype=np.float64)
    p3p_out = np.empty(n_seeds, dtype=np.float64)

    for i in prange(n_seeds):
        q2 = seeds[i, 0]
        p2 = seeds[i, 1]
        p3 = seeds[i, 3]  # q3 column is seeds[:,2] which is zero

        flag, q2_new, p2_new, p3_new = _poincare_step_jit(
            q2,
            p2,
            p3,
            dt,
            jac_H,
            clmo,
            order,
            max_steps,
            use_symplectic,
            n_dof,
        )

        if flag == 1:
            success[i] = 1
            q2p_out[i] = q2_new
            p2p_out[i] = p2_new
            p3p_out[i] = p3_new

    return success, q2p_out, p2p_out, p3p_out

def _generate_map(
    h0: float,
    H_blocks: List[np.ndarray],
    max_degree: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    n_seeds: int = 20,
    n_iter: int = 1500,
    dt: float = 1e-2,
    use_symplectic: bool = True,
    integrator_order: int = 6,
    c_omega_heuristic: float=20.0,
    seed_axis: str = "q2",  # "q2" or "p2"
) -> np.ndarray:
    """Generate a Poincaré map by iterating each seed many times.

    Parameters
    ----------
    h0 : float
        Energy level.
    H_blocks, max_degree, psi_table, clmo_table, encode_dict_list
        Same polynomial data as `_generate_grid`.
    n_seeds : int, optional
        Number of initial seeds to distribute along the chosen axis.
    n_iter : int, optional
        How many Poincaré iterates to compute for each seed.
    dt : float, optional
        Timestep for the integrator.
    use_symplectic : bool, optional
        True → Symplectic (recommended); False → RK4.
    seed_axis : {"q2", "p2"}
        Place seeds on this axis with the other momentum/position set to zero.

    Returns
    -------
    np.ndarray, shape (n_success * n_iter, 2)
        Collected (q2, p2) points of all iterates.
    """
    # 1. Build Jacobian once.
    jac_H = polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2. Turning points for seed placement.
    q2_max = _find_turning("q2", h0, H_blocks, clmo_table)
    p2_max = _find_turning("p2", h0, H_blocks, clmo_table)

    seeds: list[Tuple[float, float, float, float]] = []

    if seed_axis == "q2":
        q2_vals = np.linspace(-0.9 * q2_max, 0.9 * q2_max, n_seeds)
        for q2 in q2_vals:
            p2 = 0.0
            p3 = _solve_p3(q2, p2, h0, H_blocks, clmo_table)
            if p3 is not None:
                seeds.append((q2, p2, 0.0, p3))
    elif seed_axis == "p2":
        p2_vals = np.linspace(-0.9 * p2_max, 0.9 * p2_max, n_seeds)
        for p2 in p2_vals:
            q2 = 0.0
            p3 = _solve_p3(q2, p2, h0, H_blocks, clmo_table)
            if p3 is not None:
                seeds.append((q2, p2, 0.0, p3))
    else:
        raise ValueError("seed_axis must be 'q2' or 'p2'.")

    logger.info("Iterating %d seeds (%s-axis) for %d crossings each", len(seeds), seed_axis, n_iter)

    # 3. Iterate.
    pts_accum: list[Tuple[float, float]] = []

    # Dynamically adjust max_steps based on dt to allow a consistent total integration time for finding a crossing.
    # The original implicit max integration time (when dt=1e-3 and max_steps=20000) was 20.0.
    target_max_integration_time_per_crossing = 20.0
    calculated_max_steps = int(math.ceil(target_max_integration_time_per_crossing / dt))
    logger.info(f"Using dt={dt:.1e}, calculated max_steps per crossing: {calculated_max_steps}")

    for seed in seeds:
        state = seed
        for i in range(n_iter): # Use a different loop variable, e.g., i
            try:
                flag, q2p, p2p, p3p = _poincare_step_jit(
                    state[0],  # q2
                    state[1],  # p2
                    state[3],  # p3 (q3 is always 0)
                    dt,
                    jac_H,
                    clmo_table,
                    integrator_order,
                    calculated_max_steps,
                    use_symplectic,
                    N_SYMPLECTIC_DOF,
                    c_omega_heuristic,
                )

                if flag == 1:
                    pts_accum.append((q2p, p2p))
                    # fixed: restart with the p3 belonging to this crossing
                    state = (q2p, p2p, 0.0, p3p)
                else:
                    logger.warning(
                        "Failed to find Poincaré crossing for seed %s at iteration %d/%d",
                        seed,
                        i + 1,
                        n_iter,
                    )
                    break
            except RuntimeError as e:
                logger.warning(f"Failed to find Poincaré crossing for seed {seed} at iteration {i+1}/{n_iter}: {e}")
                break # Stop iterating this seed if a crossing is not found

    return np.asarray(pts_accum, dtype=np.float64)

def _generate_grid(
    h0: float,
    H_blocks: List[np.ndarray],
    max_degree: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    dt: float = 1e-3,
    max_steps: int = 20_000,
    Nq: int = 201,
    Np: int = 201,
    integrator_order: int = 6,
    use_symplectic: bool = False,
) -> np.ndarray:
    """
    Compute Poincaré map points at a given energy level.

    Parameters
    ----------
    h0 : float
        Energy level
    H_blocks : List[np.ndarray]
        Polynomial coefficients of Hamiltonian
    max_degree : int
        Maximum degree of polynomial
    psi_table : np.ndarray
        PSI table for polynomial operations
    clmo_table : List[np.ndarray]
        CLMO index table
    encode_dict_list : List
        Encoding dictionary for polynomial operations
    dt : float, optional
        Small integration timestep, by default 1e-3
    max_steps : int, optional
        Safety cap on the number of sub-steps, by default 20_000
    Nq : int, optional
        Number of q2 values, by default 201
    Np : int, optional
        Number of p2 values, by default 201
    integrator_order : int, optional
        Order of symplectic integrator, by default 6
    use_symplectic : bool, optional
        If True, use the extended-phase symplectic integrator; otherwise use
        an explicit RK4 step.  Default is False.

    Returns
    -------
    np.ndarray
        An (M,2) array of (q2', p2') Poincaré-map points at energy h0
    """
    logger.info(f"Computing Poincaré map for energy h0={h0:.6e}, grid size: {Nq}x{Np}")
    
    # 1.  Jacobian (once per energy level).
    logger.info("Computing Hamiltonian Jacobian")
    jac_H = polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2.  Hill-boundary turning points.
    q2_max = _find_turning("q2", h0, H_blocks, clmo_table)
    p2_max = _find_turning("p2", h0, H_blocks, clmo_table)
    logger.info(f"Hill boundary turning points: q2_max={q2_max:.6e}, p2_max={p2_max:.6e}")

    q2_vals = np.linspace(-q2_max, q2_max, Nq)
    p2_vals = np.linspace(-p2_max, p2_max, Np)

    # Find valid seeds
    logger.info("Finding valid seeds within Hill boundary")
    seeds: list[Tuple[float, float, float, float]] = []
    total_points = Nq * Np
    points_checked = 0
    valid_seeds_found = 0
    
    for q2 in q2_vals:
        for p2 in p2_vals:
            points_checked += 1
            if points_checked % (total_points // 10) == 0:
                percentage = int(100 * points_checked / total_points)
                logger.info(f"Seed search progress: {percentage}%, found {valid_seeds_found} valid seeds")
                
            p3 = _solve_p3(q2, p2, h0, H_blocks, clmo_table)
            if p3 is not None:
                seeds.append((q2, p2, 0.0, p3))
                valid_seeds_found += 1
    
    logger.info(f"Found {len(seeds)} valid seeds out of {total_points} grid points")

    # 3.  Iterate all seeds in a parallel JIT kernel.
    logger.info("Computing Poincaré map points in parallel")

    if len(seeds) == 0:
        return np.empty((0, 2), dtype=np.float64)

    seeds_arr = np.asarray(seeds, dtype=np.float64)

    success_flags, q2p_arr, p2p_arr, p3p_arr = _poincare_map_parallel(
        seeds_arr,
        dt,
        jac_H,
        clmo_table,
        integrator_order,
        max_steps,
        use_symplectic,
        N_SYMPLECTIC_DOF,
    )

    n_success = int(np.sum(success_flags))
    logger.info(f"Completed Poincaré map: {n_success} successful seeds out of {len(seeds)}")

    map_pts = np.empty((n_success, 2), dtype=np.float64)
    idx = 0
    for i in range(success_flags.shape[0]):
        if success_flags[i]:
            map_pts[idx, 0] = q2p_arr[i]
            map_pts[idx, 1] = p2p_arr[i]
            idx += 1

    return map_pts
