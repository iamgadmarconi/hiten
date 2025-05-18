from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import root_scalar

from algorithms.center.polynomial.operations import (polynomial_evaluate,
                                                     polynomial_jacobian)
from algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                               integrate_symplectic)
from log_config import logger


def find_turning(
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

    f0 = f(0.0)
    if f0 > 0:
        logger.warning(f"H(0)-h0 = {f0:.6e} > 0 along Hill boundary")
        raise RuntimeError(
            "Expected H(0)-h0 <= 0 along Hill boundary, got positive value."
        )

    # Bracket the root
    x_hi = initial_guess
    expansion_count = 0
    for expansion_count in range(max_expand):
        if f(x_hi) > 0.0:
            break
        x_hi *= expand_factor
    else:
        logger.warning(f"Failed to bracket root after {max_expand} expansions up to {x_hi:.6e}")
        raise RuntimeError(
            "Failed to bracket root for turning point after expansions up to "
            f"{x_hi}."
        )
    
    logger.debug(f"Bracketed {q_or_p} root in [0, {x_hi:.6e}] after {expansion_count+1} expansions")

    # Find the root
    sol = root_scalar(f, bracket=(0.0, x_hi), method="brentq", xtol=1e-12)
    if not sol.converged:
        logger.warning("Root finding for Hill boundary did not converge")
        raise RuntimeError("Root finding for Hill boundary did not converge.")
    
    logger.info(f"Found {q_or_p} turning point: {sol.root:.6e}")
    return float(sol.root)


def solve_p3(
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
    logger.debug(f"Solving for p3 at (q2,p2)=({q2:.4e},{p2:.4e}), h0={h0:.6e}")
    
    def f(p3: float) -> float:
        state = np.zeros(6, dtype=np.complex128)
        state[1] = q2
        state[4] = p2
        state[5] = p3
        return polynomial_evaluate(H_blocks, state, clmo).real - h0

    f0 = f(0.0)
    if f0 > 0:
        # Already above energy level at p3=0 ⇒ no positive root.
        logger.debug(f"No p3 solution at (q2,p2)=({q2:.4e},{p2:.4e}): f(0)={f0:.4e} > 0")
        return None

    p3_hi = initial_guess
    for _ in range(max_expand):
        if f(p3_hi) > 0.0:
            break
        p3_hi *= expand_factor
    else:
        logger.debug(f"No sign change for p3 up to {p3_hi:.4e} at (q2,p2)=({q2:.4e},{p2:.4e})")
        # Could not find sign change – no solution.
        return None

    sol = root_scalar(f, bracket=(0.0, p3_hi), method="brentq", xtol=1e-12)
    if sol.converged and sol.root > 0.0:
        return float(sol.root)
    return None


@njit(cache=True)
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


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
def _embed_cm_state_jit(q2: float, q3: float, p2: float, p3: float, n_dof: int) -> np.ndarray:
    vec = np.zeros(2 * n_dof, dtype=np.float64)
    vec[1] = q2
    vec[2] = q3
    vec[n_dof + 1] = p2
    vec[n_dof + 2] = p3
    return vec


@njit(cache=True)
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


@njit(cache=True)
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
) -> Tuple[int, float, float]:
    """Return (flag, q2', p2').  flag=1 if success, 0 otherwise."""
    state_old = _embed_cm_state_jit(q2, 0.0, p2, p3, n_dof)

    for _ in range(max_steps):
        if use_symplectic:
            traj = integrate_symplectic(
                initial_state_6d=state_old,
                t_values=np.array([0.0, dt]),
                jac_H_rn_typed=jac_H,
                clmo_H_typed=clmo,
                order=order,
            )
            state_new = traj[1]
        else:
            state_new = _rk4_step(state_old, dt, jac_H, clmo, n_dof)

        q3_old = state_old[2]
        q3_new = state_new[2]
        p3_new = state_new[n_dof + 2]

        if (q3_old * q3_new < 0.0) and (p3_new > 0.0):
            alpha = q3_old / (q3_old - q3_new)
            q2p = state_old[1] + alpha * (state_new[1] - state_old[1])
            p2p = state_old[n_dof + 1] + alpha * (state_new[n_dof + 1] - state_old[n_dof + 1])
            return 1, q2p, p2p

        state_old = state_new

    return 0, 0.0, 0.0


def poincare_step(
    seed4: Tuple[float, float, float, float],
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    order: int = 6,
    max_steps: int = 20_000,
    use_symplectic: bool = False,
) -> Tuple[float, float]:
    """Python wrapper around fully JIT-compiled step loop."""
    q2, p2, q3, p3 = seed4

    if not np.isclose(q3, 0.0):
        raise ValueError("Seed must lie on q3=0 section.")

    flag, q2p, p2p = _poincare_step_jit(
        q2,
        p2,
        p3,
        dt,
        jac_H,
        clmo,
        order,
        max_steps,
        use_symplectic,
        N_SYMPLECTIC_DOF,
    )

    if flag == 1:
        return float(q2p), float(p2p)

    raise RuntimeError("Poincaré return not detected within max_steps.")


def compute_poincare_map_for_energy(
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
    q2_max = find_turning("q2", h0, H_blocks, clmo_table)
    p2_max = find_turning("p2", h0, H_blocks, clmo_table)
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
                
            p3 = solve_p3(q2, p2, h0, H_blocks, clmo_table)
            if p3 is not None:
                seeds.append((q2, p2, 0.0, p3))
                valid_seeds_found += 1
    
    logger.info(f"Found {len(seeds)} valid seeds out of {total_points} grid points")

    # 3.  Iterate each seed once through the section.
    logger.info("Computing Poincaré map points")
    map_pts: list[Tuple[float, float]] = []
    successful_iterations = 0
    failed_iterations = 0
    
    for i, seed in enumerate(seeds):
        if i % max(1, len(seeds) // 10) == 0:
            percentage = int(100 * i / len(seeds))
            logger.info(f"Poincaré iteration progress: {percentage}%, mapped {successful_iterations}/{i} seeds")
            
        try:
            map_pt = poincare_step(
                seed4=seed,
                dt=dt,
                jac_H=jac_H,
                clmo=clmo_table,
                order=integrator_order,
                max_steps=max_steps,
                use_symplectic=use_symplectic,
            )
            map_pts.append(map_pt)
            successful_iterations += 1
        except RuntimeError:
            # Skip seeds that fail to return within max_steps.
            failed_iterations += 1
            continue

    logger.info(f"Completed Poincaré map with {successful_iterations} points, {failed_iterations} failed iterations")
    return np.asarray(map_pts, dtype=np.float64)
