r"""
hiten.algorithms.poincare.map
=======================

Fast generation of Poincaré sections on the centre manifold of the spatial
circular restricted three body problem (CRTBP).

References
----------
Jorba, À. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".

Zhang, H. Q., Li, S. (2001). "Improved semi-analytical computation of center
manifolds near collinear libration points".
"""

import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.optimize import root_scalar

from hiten.algorithms.polynomial.operations import (polynomial_evaluate,
                                                     polynomial_jacobian)
from hiten.algorithms.dynamics.hamiltonian import (_eval_dH_dP, _eval_dH_dQ,
                                             _hamiltonian_rhs)
from hiten.algorithms.integrators.rk import (RK4_A, RK4_B, RK4_C, RK6_A, RK6_B,
                                       RK6_C, RK8_A, RK8_B, RK8_C)
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                               integrate_symplectic)
from hiten.utils.config import FASTMATH
from hiten.utils.log_config import logger


class PoincareSection(NamedTuple):
    r"""
    Named tuple holding Poincaré section points and coordinate labels.
    """
    points: np.ndarray  # shape (n, 2) 
    labels: tuple[str, str]  # coordinate labels for the two columns


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_ham(
    y0: np.ndarray,
    t_vals: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    jac_H,
    clmo_H,
) -> np.ndarray:
    r"""
    Explicit RK integrator specialised for :pyclass:`HamiltonianSystem`.
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

def _bracketed_root(
    f: Callable[[float], float],
    initial: float = 1e-3,
    factor: float = 2.0,
    max_expand: int = 40,
    xtol: float = 1e-12,
) -> Optional[float]:
    r"""
    Return a positive root of *f* if a sign change can be bracketed.

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
    r"""
    Return the positive intercept q2_max, p2_max, q3_max or p3_max of the Hill boundary.

    Parameters
    ----------
    q_or_p : str
        One of "q2", "p2", "q3" or "p3" specifying which variable to solve for
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
        If q_or_p is not 'q2', 'p2', 'q3' or 'p3'
    RuntimeError
        If root finding fails or doesn't converge
    """
    logger.info(f"Finding {q_or_p} turning point at energy h0={h0:.6e}")
    
    if q_or_p not in {"q2", "p2", "q3", "p3"}:
        raise ValueError("q_or_p must be 'q2', 'p2', 'q3' or 'p3'.")

    def f(x: float) -> float:
        state = np.zeros(6, dtype=np.complex128)
        if q_or_p == "q2":
            state[1] = x  # q2
        elif q_or_p == "p2":
            state[4] = x  # p2 (index 4 in (q1,q2,q3,p1,p2,p3))
        elif q_or_p == "q3":
            state[2] = x  # q3
        else:  # "p3"
            state[5] = x  # p3
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

def section_closure(section_coord: str) -> Tuple[int, int, Tuple[str, str]]:
    r"""
    Create closure information for a section defined by section_coord=0.
    
    Parameters
    ----------
    section_coord : str
        The coordinate that defines the section (e.g., "q3" for q3=0 section)
        
    Returns
    -------
    tuple
        (section_index, direction_sign, labels)
        section_index: index in 6D state vector
        direction_sign: +1 or -1 for momentum crossing direction
        labels: tuple of two coordinate names that vary on the section
    """
    coord_map = {
        "q3": (2, 1, ("q2", "p2")),     # q3=0 section, p3>0 direction
        "p3": (5, -1, ("q2", "p2")),    # p3=0 section, q3<0 direction  
        "q2": (1, 1, ("q3", "p3")),     # q2=0 section, p2>0 direction
        "p2": (4, -1, ("q3", "p3")),    # p2=0 section, q2<0 direction
    }
    
    if section_coord not in coord_map:
        raise ValueError(f"Unsupported section_coord: {section_coord}")
        
    return coord_map[section_coord]


def _solve_missing_coord(
    varname: str,
    fixed_vals: dict[str, float],
    h0: float,
    H_blocks: List[np.ndarray],
    clmo: List[np.ndarray],
    initial_guess: float = 1e-3,
    expand_factor: float = 2.0,
    max_expand: int = 40,
) -> Optional[float]:
    r"""
    Solve H(...) = h0 for a missing coordinate.
    
    Parameters
    ----------
    varname : str
        Name of the variable to solve for ("q2", "p2", "q3", "p3")
    fixed_vals : dict
        Dictionary of fixed coordinate values
    h0 : float
        Energy level
    H_blocks : List[np.ndarray]
        Polynomial coefficients
    clmo : List[np.ndarray]
        CLMO index table
    initial_guess : float, optional
        Initial guess for bracketing procedure
    expand_factor : float, optional
        Factor for expanding the bracket
    max_expand : int, optional
        Maximum number of expansions to try
        
    Returns
    -------
    Optional[float]
        Solution if found, None otherwise
    """
    logger.info(f"Solving for {varname} with fixed values {fixed_vals}, h0={h0:.6e}")
    
    # Map variable names to indices in 6D state vector
    var_indices = {
        "q1": 0, "q2": 1, "q3": 2,
        "p1": 3, "p2": 4, "p3": 5
    }
    
    if varname not in var_indices:
        raise ValueError(f"Unknown variable: {varname}")
    
    solve_idx = var_indices[varname]
    
    def f(x: float) -> float:
        state = np.zeros(6, dtype=np.complex128)
        
        # Set fixed values
        for name, val in fixed_vals.items():
            if name in var_indices:
                state[var_indices[name]] = val
                
        # Set the variable we're solving for
        state[solve_idx] = x
        
        return polynomial_evaluate(H_blocks, state, clmo).real - h0

    root = _bracketed_root(f, initial=initial_guess, factor=expand_factor, max_expand=max_expand)

    if root is None:
        logger.warning("Failed to locate %s turning point within search limits", varname)
        return None

    logger.info("Found %s turning point: %.6e", varname, root)
    return root


@njit(cache=False, fastmath=FASTMATH)
def _get_section_value(state: np.ndarray, section_coord: str) -> float:
    r"""
    Return the section coordinate value.
    """
    if section_coord == "q3":
        return state[2]
    elif section_coord == "p3":
        return state[5]
    elif section_coord == "q2":
        return state[1]
    elif section_coord == "p2":
        return state[4]
    else:
        return state[2]  # Default to q3

@njit(cache=False, fastmath=FASTMATH)
def _get_direction_sign(section_coord: str) -> float:
    r"""
    Return the direction sign for crossing detection.
    """
    if section_coord == "q3":
        return 1.0  # p3 > 0
    elif section_coord == "p3":
        return 1.0  # p3 crosses from - to +, so we want dp3/dt > 0 
    elif section_coord == "q2":
        return 1.0  # p2 > 0
    elif section_coord == "p2":
        return 1.0  # p2 crosses from - to +, so we want dp2/dt > 0
    else:
        return 1.0

@njit(cache=False, fastmath=FASTMATH)
def _get_rk_coefficients(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order == 4:
        return RK4_A, RK4_B, RK4_C
    elif order == 6:
        return RK6_A, RK6_B, RK6_C
    elif order == 8:
        return RK8_A, RK8_B, RK8_C


@njit(cache=False, fastmath=FASTMATH)
def _poincare_step(
    q2: float,
    p2: float,
    q3: float,
    p3: float,
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
    section_coord: str,
    c_omega_heuristic: float=20.0,
) -> Tuple[int, float, float, float]:
    r"""
    Return (flag, q2', p2', p3').  flag=1 if success, 0 otherwise.
    """

    state_old = np.zeros(2 * n_dof, dtype=np.float64)
    state_old[1] = q2
    state_old[2] = q3
    state_old[n_dof + 1] = p2
    state_old[n_dof + 2] = p3

    for _ in range(max_steps):
        if use_symplectic:
            traj = integrate_symplectic(
                initial_state_6d=state_old,
                t_values=np.array([0.0, dt]),
                jac_H=jac_H,
                clmo_H=clmo,
                order=order,
                c_omega_heuristic=c_omega_heuristic,
            )
            state_new = traj[1]
        else:
            c_A, c_B, c_C = _get_rk_coefficients(order)
            traj = _integrate_rk_ham(
                y0=state_old,
                t_vals=np.array([0.0, dt]),
                A=c_A,
                B=c_B,
                C=c_C,
                jac_H=jac_H,
                clmo_H=clmo,
            )
            state_new = traj[1]

        f_old = _get_section_value(state_old, section_coord)
        f_new = _get_section_value(state_new, section_coord)
        
        # Direction-dependent momentum check
        # For coordinate sections (q3=0, q2=0), check associated momentum > 0
        # For momentum sections (p3=0, p2=0), check associated coordinate derivative > 0
        if section_coord == "q3":
            momentum_check = state_new[n_dof + 2] > 0.0  # p3 > 0
        elif section_coord == "p3":
            # For p3=0 section, check that dq3/dt > 0 (trajectory moving in +q3 direction)
            rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
            momentum_check = rhs_new[2] > 0.0  # dq3/dt > 0
        elif section_coord == "q2":
            momentum_check = state_new[n_dof + 1] > 0.0  # p2 > 0
        elif section_coord == "p2":
            # For p2=0 section, check that dq2/dt > 0 (trajectory moving in +q2 direction)
            rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
            momentum_check = rhs_new[1] > 0.0  # dq2/dt > 0
        else:
            momentum_check = True  # Default

        if (f_old * f_new < 0.0) and momentum_check:

            # 1) linear first guess
            alpha = f_old / (f_old - f_new)

            # 2) endpoint derivatives for Hermite poly (need dt-scaled slopes)
            rhs_old = _hamiltonian_rhs(state_old, jac_H, clmo, n_dof)
            rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
            
            # Get the derivative index based on section coordinate
            if section_coord == "q3":
                deriv_idx = 2  # dq3/dt
            elif section_coord == "p3":
                deriv_idx = n_dof + 2  # dp3/dt
            elif section_coord == "q2":
                deriv_idx = 1  # dq2/dt
            elif section_coord == "p2":
                deriv_idx = n_dof + 1  # dp2/dt
            else:
                deriv_idx = 2  # Default to q3
                
            m0 = rhs_old[deriv_idx] * dt    # section derivative at t=0
            m1 = rhs_new[deriv_idx] * dt    # section derivative at t=dt

            # 3) cubic Hermite coefficients H(t) = a t³ + b t² + c t + d   ( 0 ≤ t ≤ 1 )
            d  = f_old
            c  = m0
            b  = 3.0*(f_new - f_old) - (2.0*m0 +   m1)
            a  = 2.0*(f_old - f_new) + (   m0 +   m1)

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
            p3p = hermite(state_old[n_dof+2], state_new[n_dof+2],
                          rhs_old[n_dof+2], rhs_new[n_dof+2])

            return 1, q2p, p2p, p3p

        state_old = state_new

    return 0, 0.0, 0.0, 0.0

@njit(parallel=True, cache=False)
def _poincare_map(
    seeds: np.ndarray,  # (N,4) float64
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
    section_coord: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return (success flags, q2p array, p2p array, p3p array) processed in parallel.
    """
    n_seeds = seeds.shape[0]
    success = np.zeros(n_seeds, dtype=np.int64)
    q2p_out = np.empty(n_seeds, dtype=np.float64)
    p2p_out = np.empty(n_seeds, dtype=np.float64)
    p3p_out = np.empty(n_seeds, dtype=np.float64)

    for i in prange(n_seeds):
        q2 = seeds[i, 0]
        p2 = seeds[i, 1]
        q3 = seeds[i, 2]
        p3 = seeds[i, 3]

        flag, q2_new, p2_new, p3_new = _poincare_step(
            q2,
            p2,
            q3,
            p3,
            dt,
            jac_H,
            clmo,
            order,
            max_steps,
            use_symplectic,
            n_dof,
            section_coord,
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
    section_coord: str = "q3",  # "q2", "p2", "q3", or "p3"
) -> PoincareSection:
    r"""
    Generate a Poincaré return map by forward integration of a small set of seeds.

    Parameters
    ----------
    h0 : float
        Target energy :math:`h_0` on the centre manifold.
    H_blocks : list[numpy.ndarray]
        Packed polynomial coefficients of the reduced Hamiltonian.
    max_degree : int
        Maximum homogeneous degree :math:`N` kept in the truncation.
    psi_table : numpy.ndarray
        PSI lookup table used by polynomial routines.
    clmo_table : list[numpy.ndarray]
        CLMO index tables corresponding to *H_blocks*.
    encode_dict_list : list[dict]
        Helper encoders for multi-index compression.
    n_seeds : int, default 20
        Number of seeds laid uniformly along *seed_axis*.
    n_iter : int, default 1500
        Successive Poincaré crossings computed for each seed.
    dt : float, default 1e-2
        Fixed integration step size.
    use_symplectic : bool, default True
        If ``True`` use the extended-phase symplectic integrator, otherwise an explicit Runge-Kutta scheme.
    integrator_order : {4, 6, 8}, default 6
        Tableau order for the numerical integrator.
    c_omega_heuristic : float, default 20.0
        Heuristic scaling constant for the extended-phase method.
    seed_axis : {'q2', 'p2'}, default 'q2'
        Coordinate along which seeds are distributed.
    section_coord : {'q2', 'p2', 'q3', 'p3'}, default 'q3'
        Coordinate that defines the Poincaré section :math:`\Sigma`.

    Returns
    -------
    PoincareSection
        Map points with matching axis labels.

    Raises
    ------
    ValueError
        If *section_coord* is not supported.

    Notes
    -----
    The routine is performance-critical and therefore JIT-compiled with :pyfunc:`numba.njit`.
    """
    # Get section information
    section_idx, direction_sign, labels = section_closure(section_coord)
    
    # 1. Build Jacobian once.
    jac_H = polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2. Generate seeds based on section type
    seeds: list[Tuple[float, float, float, float]] = []
    
    if section_coord == "q3":
        # Traditional q3=0 section: vary along seed_axis, solve for p3
        q2_max = _find_turning("q2", h0, H_blocks, clmo_table)
        p2_max = _find_turning("p2", h0, H_blocks, clmo_table)
        
        if seed_axis == "q2":
            q2_vals = np.linspace(-0.9 * q2_max, 0.9 * q2_max, n_seeds)
            for q2 in q2_vals:
                p2 = 0.0
                p3 = _solve_missing_coord("p3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if p3 is not None:
                    seeds.append((q2, p2, 0.0, p3))
        elif seed_axis == "p2":
            p2_vals = np.linspace(-0.9 * p2_max, 0.9 * p2_max, n_seeds)
            for p2 in p2_vals:
                q2 = 0.0
                p3 = _solve_missing_coord("p3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if p3 is not None:
                    seeds.append((q2, p2, 0.0, p3))
    
    elif section_coord == "p3":
        # p3=0 section: vary along seed_axis, solve for q3
        q2_max = _find_turning("q2", h0, H_blocks, clmo_table)
        p2_max = _find_turning("p2", h0, H_blocks, clmo_table)
        
        if seed_axis == "q2":
            q2_vals = np.linspace(-0.9 * q2_max, 0.9 * q2_max, n_seeds)
            for q2 in q2_vals:
                p2 = 0.0
                q3 = _solve_missing_coord("q3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if q3 is not None:
                    seeds.append((q2, p2, q3, 0.0))
        elif seed_axis == "p2":
            p2_vals = np.linspace(-0.9 * p2_max, 0.9 * p2_max, n_seeds)
            for p2 in p2_vals:
                q2 = 0.0
                q3 = _solve_missing_coord("q3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if q3 is not None:
                    seeds.append((q2, p2, q3, 0.0))
                    
    elif section_coord == "q2":
        # q2=0 section: vary along seed_axis (q3 or p3), solve for the other
        q3_max = _find_turning("q3", h0, H_blocks, clmo_table)
        p3_max = _find_turning("p3", h0, H_blocks, clmo_table)
        
        if seed_axis == "q3":
            q3_vals = np.linspace(-0.9 * q3_max, 0.9 * q3_max, n_seeds)
            for q3 in q3_vals:
                p2 = 0.0
                p3 = _solve_missing_coord("p3", {"q2": 0.0, "q3": q3, "p2": p2}, h0, H_blocks, clmo_table)
                if p3 is not None:
                    seeds.append((0.0, p2, q3, p3))
        elif seed_axis == "p3":
            p3_vals = np.linspace(-0.9 * p3_max, 0.9 * p3_max, n_seeds)
            for p3 in p3_vals:
                p2 = 0.0
                q3 = _solve_missing_coord("q3", {"q2": 0.0, "p2": p2, "p3": p3}, h0, H_blocks, clmo_table)
                if q3 is not None:
                    seeds.append((0.0, p2, q3, p3))
                    
    elif section_coord == "p2":
        # p2=0 section: vary along seed_axis (q3 or p3), solve for the other
        q3_max = _find_turning("q3", h0, H_blocks, clmo_table)
        p3_max = _find_turning("p3", h0, H_blocks, clmo_table)
        
        if seed_axis == "q3":
            q3_vals = np.linspace(-0.9 * q3_max, 0.9 * q3_max, n_seeds)
            for q3 in q3_vals:
                q2 = 0.0
                p3 = _solve_missing_coord("p3", {"q2": q2, "q3": q3, "p2": 0.0}, h0, H_blocks, clmo_table)
                if p3 is not None:
                    seeds.append((q2, 0.0, q3, p3))
        elif seed_axis == "p3":
            p3_vals = np.linspace(-0.9 * p3_max, 0.9 * p3_max, n_seeds)
            for p3 in p3_vals:
                q2 = 0.0
                q3 = _solve_missing_coord("q3", {"q2": q2, "p2": 0.0, "p3": p3}, h0, H_blocks, clmo_table)
                if q3 is not None:
                    seeds.append((q2, 0.0, q3, p3))
    else:
        raise ValueError(f"Unsupported section_coord: {section_coord}")

    # 3. Iterate each seed to generate map points
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
                flag, q2p, p2p, p3p = _poincare_step(
                    state[0],  # q2
                    state[1],  # p2
                    state[2],  # q3
                    state[3],  # p3
                    dt,
                    jac_H,
                    clmo_table,
                    integrator_order,
                    calculated_max_steps,
                    use_symplectic,
                    N_SYMPLECTIC_DOF,
                    section_coord,
                    c_omega_heuristic,
                )

                if flag == 1:
                    # Extract the appropriate coordinates for the section
                    if section_coord == "q3":
                        pts_accum.append((q2p, p2p))
                        state = (q2p, p2p, 0.0, p3p)
                    elif section_coord == "p3":
                        pts_accum.append((q2p, p2p))
                        # For p3=0 section, need to keep q3 from the crossing
                        state = (q2p, p2p, state[2], 0.0)  # p3 is fixed at 0
                    elif section_coord == "q2":
                        pts_accum.append((state[2], p3p))  # (q3, p3)
                        state = (0.0, p2p, state[2], p3p)  # q2 is fixed at 0
                    elif section_coord == "p2":
                        pts_accum.append((state[2], p3p))  # (q3, p3)
                        state = (q2p, 0.0, state[2], p3p)  # p2 is fixed at 0
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

    if len(pts_accum) == 0:
        # Return empty array with correct shape
        points_array = np.empty((0, 2), dtype=np.float64)
    else:
        points_array = np.asarray(pts_accum, dtype=np.float64)
    return PoincareSection(points_array, labels)

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
    section_coord: str = "q3",
) -> PoincareSection:
    r"""
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
    :class:`PoincareSection`
        Section points with coordinate labels

    Raises
    ------
    ValueError
        If *section_coord* is not one of the supported identifiers.

    Notes
    -----
    Designed for exhaustive scans. For faster qualitative exploration use
    :pyfunc:`_generate_map` with a handful of seeds.
    """
    logger.info(f"Computing Poincaré map for energy h0={h0:.6e}, grid size: {Nq}x{Np}")
    
    # Get section information
    section_idx, direction_sign, labels = section_closure(section_coord)
    
    # 1.  Jacobian (once per energy level).
    logger.info("Computing Hamiltonian Jacobian")
    jac_H = polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2.  Hill-boundary turning points - get the right ones based on section
    if section_coord in ("q3", "p3"):
        # For q3 or p3 sections, vary q2 and p2
        q2_max = _find_turning("q2", h0, H_blocks, clmo_table)
        p2_max = _find_turning("p2", h0, H_blocks, clmo_table)
        logger.info(f"Hill boundary turning points: q2_max={q2_max:.6e}, p2_max={p2_max:.6e}")
        coord1_vals = np.linspace(-q2_max, q2_max, Nq)
        coord2_vals = np.linspace(-p2_max, p2_max, Np)
    elif section_coord in ("q2", "p2"):
        # For q2 or p2 sections, vary q3 and p3
        q3_max = _find_turning("q3", h0, H_blocks, clmo_table)
        p3_max = _find_turning("p3", h0, H_blocks, clmo_table)
        logger.info(f"Hill boundary turning points: q3_max={q3_max:.6e}, p3_max={p3_max:.6e}")
        coord1_vals = np.linspace(-q3_max, q3_max, Nq)
        coord2_vals = np.linspace(-p3_max, p3_max, Np)
    else:
        raise ValueError(f"Unsupported section_coord: {section_coord}")

    # Find valid seeds
    logger.info("Finding valid seeds within Hill boundary")
    seeds: list[Tuple[float, float, float, float]] = []
    total_points = Nq * Np
    points_checked = 0
    valid_seeds_found = 0
    
    for coord1 in coord1_vals:
        for coord2 in coord2_vals:
            points_checked += 1
            if points_checked % (total_points // 10) == 0:
                percentage = int(100 * points_checked / total_points)
                logger.info(f"Seed search progress: {percentage}%, found {valid_seeds_found} valid seeds")
                
            # Dispatch seed solving based on section type
            if section_coord == "q3":
                # q3=0 section: coord1=q2, coord2=p2, solve for p3
                missing_coord = _solve_missing_coord("p3", {"q2": coord1, "p2": coord2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if missing_coord is not None:
                    seeds.append((coord1, coord2, 0.0, missing_coord))
                    valid_seeds_found += 1
            elif section_coord == "p3":
                # p3=0 section: coord1=q2, coord2=p2, solve for q3
                missing_coord = _solve_missing_coord("q3", {"q2": coord1, "p2": coord2, "p3": 0.0}, h0, H_blocks, clmo_table)
                if missing_coord is not None:
                    seeds.append((coord1, coord2, missing_coord, 0.0))
                    valid_seeds_found += 1
            elif section_coord == "q2":
                # q2=0 section: coord1=q3, coord2=p3, solve for p2
                missing_coord = _solve_missing_coord("p2", {"q2": 0.0, "q3": coord1, "p3": coord2}, h0, H_blocks, clmo_table)
                if missing_coord is not None:
                    seeds.append((0.0, missing_coord, coord1, coord2))
                    valid_seeds_found += 1
            elif section_coord == "p2":
                # p2=0 section: coord1=q3, coord2=p3, solve for q2
                missing_coord = _solve_missing_coord("q2", {"p2": 0.0, "q3": coord1, "p3": coord2}, h0, H_blocks, clmo_table)
                if missing_coord is not None:
                    seeds.append((missing_coord, 0.0, coord1, coord2))
                    valid_seeds_found += 1
    
    logger.info(f"Found {len(seeds)} valid seeds out of {total_points} grid points")

    # 3.  Iterate all seeds in a parallel JIT kernel.
    logger.info("Computing Poincaré map points in parallel")

    if len(seeds) == 0:
        return PoincareSection(np.empty((0, 2), dtype=np.float64), labels)

    seeds_arr = np.asarray(seeds, dtype=np.float64)

    success_flags, q2p_arr, p2p_arr, p3p_arr = _poincare_map(
        seeds_arr,
        dt,
        jac_H,
        clmo_table,
        integrator_order,
        max_steps,
        use_symplectic,
        N_SYMPLECTIC_DOF,
        section_coord,
    )

    n_success = int(np.sum(success_flags))
    logger.info(f"Completed Poincaré map: {n_success} successful seeds out of {len(seeds)}")

    map_pts = np.empty((n_success, 2), dtype=np.float64)
    idx = 0
    for i in range(success_flags.shape[0]):
        if success_flags[i]:
            # Extract the appropriate coordinates based on section type
            if section_coord == "q3":
                map_pts[idx, 0] = q2p_arr[i]
                map_pts[idx, 1] = p2p_arr[i]
            elif section_coord == "p3":
                map_pts[idx, 0] = q2p_arr[i]
                map_pts[idx, 1] = p2p_arr[i]
            elif section_coord == "q2":
                # For q2=0 section, we store (q3, p3)
                map_pts[idx, 0] = seeds_arr[i, 2]  # q3 coordinate
                map_pts[idx, 1] = p3p_arr[i]  # p3 coordinate from crossing
            elif section_coord == "p2":
                # For p2=0 section, we store (q3, p3) 
                map_pts[idx, 0] = seeds_arr[i, 2]  # q3 coordinate
                map_pts[idx, 1] = p3p_arr[i]  # p3 coordinate from crossing
            idx += 1

    return PoincareSection(map_pts, labels)
