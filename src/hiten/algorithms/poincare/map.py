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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.hamiltonian import (_eval_dH_dP, _eval_dH_dQ,
                                                   _hamiltonian_rhs)
from hiten.algorithms.integrators.rk import (RK4_A, RK4_B, RK4_C, RK6_A, RK6_B,
                                             RK6_C, RK8_A, RK8_B, RK8_C)
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                                     _integrate_symplectic)
from hiten.algorithms.polynomial.operations import (_polynomial_evaluate,
                                                    _polynomial_jacobian)
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


class _PoincareSection(NamedTuple):
    r"""
    Named tuple holding Poincaré section points and coordinate labels.
    """
    points: np.ndarray  # shape (n, 2) 
    labels: tuple[str, str]  # coordinate labels for the two columns


class _PoincareSectionConfig:

    def __init__(self, section_coord: str):
        self.section_coord = section_coord
        
        # Define coordinate mappings for each section type
        if section_coord == "q3":
            self.section_index = 2  # q3 is at index 2 in state vector
            self.section_value = 0.0  # q3 = 0
            self.plane_coords = ("q2", "p2")  # coordinates that vary on the section
            self.plane_indices = (1, 4)  # indices in state vector (q2, p2)
            self.missing_coord = "p3"  # coordinate to solve for given energy constraint
            self.missing_index = 5  # p3 is at index 5
            self.momentum_check_index = 5  # check p3 > 0 for crossing direction
            self.momentum_check_sign = 1.0
            self.deriv_index = 2  # dq3/dt for interpolation
            self.other_coords = ("q3", "p3")  # the other two coordinates
            self.other_indices = (2, 5)
        elif section_coord == "p3":
            self.section_index = 5  # p3 is at index 5
            self.section_value = 0.0  # p3 = 0
            self.plane_coords = ("q2", "p2")
            self.plane_indices = (1, 4)
            self.missing_coord = "q3"
            self.missing_index = 2
            self.momentum_check_index = 2  # check dq3/dt > 0
            self.momentum_check_sign = 1.0
            self.deriv_index = 5  # dp3/dt for interpolation
            self.other_coords = ("q3", "p3")
            self.other_indices = (2, 5)
        elif section_coord == "q2":
            self.section_index = 1  # q2 is at index 1
            self.section_value = 0.0  # q2 = 0
            self.plane_coords = ("q3", "p3")
            self.plane_indices = (2, 5)
            self.missing_coord = "p2"
            self.missing_index = 4
            self.momentum_check_index = 4  # check p2 > 0
            self.momentum_check_sign = 1.0
            self.deriv_index = 1  # dq2/dt for interpolation
            self.other_coords = ("q2", "p2")
            self.other_indices = (1, 4)
        elif section_coord == "p2":
            self.section_index = 4  # p2 is at index 4
            self.section_value = 0.0  # p2 = 0
            self.plane_coords = ("q3", "p3")
            self.plane_indices = (2, 5)
            self.missing_coord = "q2"
            self.missing_index = 1
            self.momentum_check_index = 1  # check dq2/dt > 0
            self.momentum_check_sign = 1.0
            self.deriv_index = 4  # dp2/dt for interpolation
            self.other_coords = ("q2", "p2")
            self.other_indices = (1, 4)
        else:
            raise ValueError(f"Unsupported section_coord: {section_coord}")
    
    def get_section_value(self, state: np.ndarray) -> float:
        """Get the value of the section coordinate from a state vector."""
        return state[self.section_index]
    
    def check_crossing_direction(self, state: np.ndarray, rhs: Optional[np.ndarray] = None, 
                               jac_H=None, clmo=None, n_dof: int = 3) -> bool:
        """Check if the trajectory is crossing in the correct direction."""
        if self.section_coord in ("q3", "q2"):
            # For coordinate sections, check associated momentum > 0
            return state[self.momentum_check_index] > 0.0
        else:
            # For momentum sections, check coordinate derivative > 0
            if rhs is None:
                rhs = _hamiltonian_rhs(state, jac_H, clmo, n_dof)
            return rhs[self.momentum_check_index] > 0.0
    
    def extract_plane_coords(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract the two coordinates that define the section plane."""
        return state[self.plane_indices[0]], state[self.plane_indices[1]]
    
    def extract_other_coords(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract the two coordinates not in the section plane."""
        return state[self.other_indices[0]], state[self.other_indices[1]]
    
    def build_state(self, plane_vals: Tuple[float, float], other_vals: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Build a (q2, p2, q3, p3) state tuple from plane and other coordinates."""
        # Initialize with zeros
        q2 = p2 = q3 = p3 = 0.0
        
        # Set plane coordinates
        if self.plane_coords == ("q2", "p2"):
            q2, p2 = plane_vals
            q3, p3 = other_vals
        else:  # ("q3", "p3")
            q3, p3 = plane_vals
            q2, p2 = other_vals
            
        # Override section coordinate with its fixed value
        if self.section_coord == "q2":
            q2 = self.section_value
        elif self.section_coord == "p2":
            p2 = self.section_value
        elif self.section_coord == "q3":
            q3 = self.section_value
        elif self.section_coord == "p3":
            p3 = self.section_value
            
        return q2, p2, q3, p3
    
    def build_constraint_dict(self, **kwargs) -> dict[str, float]:
        """Build a constraint dictionary for _solve_missing_coord."""
        constraints = {}
        
        # Add section constraint
        constraints[self.section_coord] = self.section_value
        
        # Add any additional constraints
        for key, value in kwargs.items():
            if key in ("q1", "q2", "q3", "p1", "p2", "p3"):
                constraints[key] = value
                
        return constraints


# Cache section configs to avoid repeated initialization
_SECTION_CONFIGS = {
    coord: _PoincareSectionConfig(coord) 
    for coord in ("q2", "p2", "q3", "p3")
}


def _get_section_config(section_coord: str) -> _PoincareSectionConfig:
    """Get the configuration for a given section coordinate."""
    if section_coord not in _SECTION_CONFIGS:
        raise ValueError(f"Unsupported section_coord: {section_coord}")
    return _SECTION_CONFIGS[section_coord]


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_ham(y0: np.ndarray, t_vals: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, jac_H, clmo_H) -> np.ndarray:
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

def _bracketed_root(f: Callable[[float], float], initial: float = 1e-3, factor: float = 2.0, max_expand: int = 40, xtol: float = 1e-12) -> Optional[float]:
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

def _find_turning(q_or_p: str, h0: float, H_blocks: List[np.ndarray], clmo: List[np.ndarray], initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40) -> float:
    fixed_vals = {coord: 0.0 for coord in ("q2", "p2", "q3", "p3") if coord != q_or_p}
    
    root = _solve_missing_coord(
        q_or_p, fixed_vals, h0, H_blocks, clmo, 
        initial_guess, expand_factor, max_expand
    )
    
    if root is None:
        logger.warning("Failed to locate %s turning point within search limits", q_or_p)
        raise RuntimeError("Root finding for Hill boundary did not converge.")

    return root

def _section_closure(section_coord: str) -> Tuple[int, int, Tuple[str, str]]:
    config = _get_section_config(section_coord)
    return config.section_index, config.momentum_check_sign, config.plane_coords


def _solve_missing_coord(varname: str, fixed_vals: dict[str, float], h0: float, H_blocks: List[np.ndarray], clmo: List[np.ndarray], initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40) -> Optional[float]:
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
        
        return _polynomial_evaluate(H_blocks, state, clmo).real - h0

    root = _bracketed_root(f, initial=initial_guess, factor=expand_factor, max_expand=max_expand)

    if root is None:
        logger.warning("Failed to locate %s turning point within search limits", varname)
        return None

    logger.debug("Found %s turning point: %.6e", varname, root)
    return root


@njit(cache=False, fastmath=FASTMATH)
def _get_section_value(state: np.ndarray, section_coord: str) -> float:
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
) -> Tuple[int, float, float, float, float]:
    state_old = np.zeros(2 * n_dof, dtype=np.float64)
    state_old[1] = q2
    state_old[2] = q3
    state_old[n_dof + 1] = p2
    state_old[n_dof + 2] = p3

    for _ in range(max_steps):
        if use_symplectic:
            traj = _integrate_symplectic(
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
        
        # Direction-dependent momentum check using simplified logic
        # For coordinate sections (q3=0, q2=0), check associated momentum > 0
        # For momentum sections (p3=0, p2=0), check associated coordinate derivative > 0
        if section_coord in ("q3", "q2"):
            # For coordinate sections, check momentum
            if section_coord == "q3":
                momentum_check = state_new[n_dof + 2] > 0.0  # p3 > 0
            else:  # q2
                momentum_check = state_new[n_dof + 1] > 0.0  # p2 > 0
        else:
            # For momentum sections, check coordinate derivative
            rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
            if section_coord == "p3":
                momentum_check = rhs_new[2] > 0.0  # dq3/dt > 0
            else:  # p2
                momentum_check = rhs_new[1] > 0.0  # dq2/dt > 0

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
            else:  # p2
                deriv_idx = n_dof + 1  # dp2/dt
                
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
            q3p = hermite(state_old[2], state_new[2], rhs_old[2], rhs_new[2])
            p3p = hermite(state_old[n_dof+2], state_new[n_dof+2],
                          rhs_old[n_dof+2], rhs_new[n_dof+2])

            return 1, q2p, p2p, q3p, p3p

        state_old = state_new

    return 0, 0.0, 0.0, 0.0, 0.0

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_seeds = seeds.shape[0]
    success = np.zeros(n_seeds, dtype=np.int64)
    q2p_out = np.empty(n_seeds, dtype=np.float64)
    p2p_out = np.empty(n_seeds, dtype=np.float64)
    q3p_out = np.empty(n_seeds, dtype=np.float64)
    p3p_out = np.empty(n_seeds, dtype=np.float64)

    for i in prange(n_seeds):
        q2 = seeds[i, 0]
        p2 = seeds[i, 1]
        q3 = seeds[i, 2]
        p3 = seeds[i, 3]

        flag, q2_new, p2_new, q3_new, p3_new = _poincare_step(q2, p2, q3,
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
            q3p_out[i] = q3_new
            p3p_out[i] = p3_new

    return success, q2p_out, p2p_out, q3p_out, p3p_out

def _iterate_seed(
    seed: Tuple[float, float, float, float],
    n_iter: int,
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo_table: List[np.ndarray],
    integrator_order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
    section_coord: str,
    c_omega_heuristic: float,
) -> List[Tuple[float, float]]:
    config = _get_section_config(section_coord)
    pts_accum: List[Tuple[float, float]] = []
    state = seed
    
    for i in range(n_iter):
        try:
            flag, q2p, p2p, q3p, p3p = _poincare_step(
                state[0],  # q2
                state[1],  # p2
                state[2],  # q3
                state[3],  # p3
                dt,
                jac_H,
                clmo_table,
                integrator_order,
                max_steps,
                use_symplectic,
                n_dof,
                section_coord,
                c_omega_heuristic,
            )

            if flag == 1:
                # Extract coordinates using configuration
                new_state_6d = np.array([0.0, q2p, q3p, 0.0, p2p, p3p])  # (q1, q2, q3, p1, p2, p3)
                plane_coords = config.extract_plane_coords(new_state_6d)
                pts_accum.append(plane_coords)
                
                # Build next state with section constraint
                other_coords = config.extract_other_coords(new_state_6d)
                state = config.build_state(plane_coords, other_coords)
            else:
                logger.warning(
                    "Failed to find Poincaré crossing for seed %s at iteration %d/%d",
                    seed, i + 1, n_iter
                )
                break
        except RuntimeError as e:
            logger.warning(f"Failed to find Poincaré crossing for seed {seed} at iteration {i+1}/{n_iter}: {e}")
            break

    return pts_accum


def _process_seed_chunk(
    seed_chunk: List[Tuple[float, float, float, float]],
    n_iter: int,
    dt: float,
    jac_H: List[List[np.ndarray]],
    clmo_table: List[np.ndarray],
    integrator_order: int,
    max_steps: int,
    use_symplectic: bool,
    n_dof: int,
    section_coord: str,
    c_omega_heuristic: float,
) -> List[Tuple[float, float]]:
    pts_accum: List[Tuple[float, float]] = []
    
    for seed in seed_chunk:
        seed_points = _iterate_seed(
            seed, n_iter, dt, jac_H, clmo_table, integrator_order,
            max_steps, use_symplectic, n_dof, section_coord, c_omega_heuristic
        )
        pts_accum.extend(seed_points)

    return pts_accum

def _process_grid_chunk(
    coord_pairs: List[Tuple[float, float]],
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray],
    section_coord: str,
) -> List[Tuple[float, float, float, float]]:
    config = _get_section_config(section_coord)
    seeds: List[Tuple[float, float, float, float]] = []
    
    for coord1, coord2 in coord_pairs:
        # Build constraints using the configuration
        constraints = config.build_constraint_dict(**{
            config.plane_coords[0]: coord1,
            config.plane_coords[1]: coord2
        })
        
        missing_coord = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
        if missing_coord is not None:
            # Build the full state
            other_vals = [0.0, 0.0]
            missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
            other_vals[missing_idx] = missing_coord
            seed = config.build_state((coord1, coord2), (other_vals[0], other_vals[1]))
            seeds.append(seed)
    
    return seeds

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
    section_coord: str = "q3",  # "q2", "p2", "q3", or "p3"
    parallel: bool = False,  # Enable CPU parallelization
    n_processes: Optional[int] = None,  # Number of processes (default: CPU count)
    seed_strategy: str = "axis_aligned",  # "single", "axis_aligned", "level_sets", "radial", "random"
    seed_axis: Optional[str] = None,  # Which axis to seed on for "single" strategy
) -> _PoincareSection:
    # Get section information
    config = _get_section_config(section_coord)
    
    # 1. Build Jacobian once.
    jac_H = _polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2. Generate seeds using the specified strategy
    seeds = _generate_seeds(section_coord=section_coord, h0=h0, H_blocks=H_blocks,
                            clmo_table=clmo_table, n_seeds=n_seeds, seed_strategy=seed_strategy,
                            seed_axis=seed_axis,
                        )

    # 3. Process seeds to generate map points
    # Dynamically adjust max_steps based on dt to allow a consistent total integration time for finding a crossing.
    # The original implicit max integration time (when dt=1e-3 and max_steps=20000) was 20.0.
    target_max_integration_time_per_crossing = 20.0
    calculated_max_steps = int(math.ceil(target_max_integration_time_per_crossing / dt))
    logger.info(f"Using dt={dt:.1e}, calculated max_steps per crossing: {calculated_max_steps}")

    pts_accum: list[Tuple[float, float]] = []

    if parallel and len(seeds) > 1:
        # Parallel processing
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        logger.info(f"Using parallel processing with {n_processes} threads for {len(seeds)} seeds")
        
        # Split seeds into chunks for parallel processing
        chunk_size = max(1, len(seeds) // n_processes)
        seed_chunks = [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]
        
        # Process chunks in parallel using threads (avoids Numba pickling issues)
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            for chunk in seed_chunks:
                if chunk:  # Skip empty chunks
                    future = executor.submit(
                        _process_seed_chunk, chunk, n_iter, dt, jac_H, clmo_table,
                            integrator_order, calculated_max_steps, use_symplectic,
                            N_SYMPLECTIC_DOF, section_coord, c_omega_heuristic
                        )
                    futures.append(future)
            
            # Collect results from all processes
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    pts_accum.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    
        logger.info(f"Parallel processing completed. Generated {len(pts_accum)} map points.")
        
    else:
        # Sequential processing using unified iteration function
        for seed in seeds:
            seed_points = _iterate_seed(
                seed, n_iter, dt, jac_H, clmo_table, integrator_order,
                calculated_max_steps, use_symplectic, N_SYMPLECTIC_DOF, 
                section_coord, c_omega_heuristic
            )
            pts_accum.extend(seed_points)

    if len(pts_accum) == 0:
        # Return empty array with correct shape
        points_array = np.empty((0, 2), dtype=np.float64)
    else:
        points_array = np.asarray(pts_accum, dtype=np.float64)
    return _PoincareSection(points_array, config.plane_coords)

def _generate_seeds(
    section_coord: str,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray],
    n_seeds: int,
    seed_strategy: str = "axis_aligned",  # "single", "axis_aligned", "level_sets", "radial", "random"
    seed_axis: Optional[str] = None,  # Which axis to seed on for "single" strategy
) -> List[Tuple[float, float, float, float]]:
    logger.info(f"Generating {n_seeds} seeds with strategy '{seed_strategy}' for {section_coord} section")
    
    config = _get_section_config(section_coord)
    
    # Get boundary turning points for plane coordinates
    try:
        plane_maxes = []
        for coord in config.plane_coords:
            turning_point = _find_turning(coord, h0, H_blocks, clmo_table)
            plane_maxes.append(turning_point)
    except RuntimeError as e:
        logger.warning(f"Failed to find some turning points: {e}")
        plane_maxes = [0.1, 0.1]  # Fallback values
    
    # Dispatch to appropriate strategy function
    if seed_strategy == "single":
        seeds = _generate_seeds_single(config, plane_maxes, n_seeds, seed_axis, h0, H_blocks, clmo_table)
    elif seed_strategy == "axis_aligned":
        seeds = _generate_seeds_axis_aligned(config, plane_maxes, n_seeds, h0, H_blocks, clmo_table)
    elif seed_strategy == "level_sets":
        seeds = _generate_seeds_level_sets(config, plane_maxes, n_seeds, h0, H_blocks, clmo_table)
    elif seed_strategy == "radial":
        seeds = _generate_seeds_radial(config, plane_maxes, n_seeds, h0, H_blocks, clmo_table)
    elif seed_strategy == "random":
        seeds = _generate_seeds_random(config, plane_maxes, n_seeds, h0, H_blocks, clmo_table)
    else:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy}")
    
    logger.info(f"Generated {len(seeds)} seeds using '{seed_strategy}' strategy")
    return seeds


def _generate_seeds_single(
    config: _PoincareSectionConfig,
    plane_maxes: List[float],
    n_seeds: int,
    seed_axis: Optional[str],
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray]
) -> List[Tuple[float, float, float, float]]:
    """Generate seeds along a single axis with the other coordinate set to zero."""
    # Determine which plane coordinate to vary
    if seed_axis is None:
        logger.warning("seed_axis not specified for 'single' strategy, using first plane coordinate")
        axis_idx = 0
    else:
        try:
            axis_idx = config.plane_coords.index(seed_axis)
        except ValueError:
            logger.warning(f"seed_axis '{seed_axis}' not found in plane coordinates {config.plane_coords}. Using first coordinate.")
            axis_idx = 0
    
    seeds = []
    coord_vals = np.linspace(-0.9 * plane_maxes[axis_idx], 0.9 * plane_maxes[axis_idx], n_seeds)
    
    for coord_val in coord_vals:
        plane_vals = [0.0, 0.0]
        plane_vals[axis_idx] = coord_val
        constraints = config.build_constraint_dict(**{
            config.plane_coords[0]: plane_vals[0],
            config.plane_coords[1]: plane_vals[1]
        })
        
        missing_val = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
        if missing_val is not None:
            other_vals = [0.0, 0.0]
            missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
            other_vals[missing_idx] = missing_val
            seed = config.build_state((plane_vals[0], plane_vals[1]), (other_vals[0], other_vals[1]))
            seeds.append(seed)
    
    return seeds


def _generate_seeds_axis_aligned(
    config: _PoincareSectionConfig,
    plane_maxes: List[float],
    n_seeds: int,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray]
) -> List[Tuple[float, float, float, float]]:
    """Generate seeds along both coordinate axes (e.g., q2-axis and p2-axis)."""
    seeds = []
    seeds_per_axis = n_seeds // 2
    
    for i, coord in enumerate(config.plane_coords):
        coord_vals = np.linspace(-0.9 * plane_maxes[i], 0.9 * plane_maxes[i], seeds_per_axis)
        
        for coord_val in coord_vals:
            plane_vals = [0.0, 0.0]
            plane_vals[i] = coord_val
            constraints = config.build_constraint_dict(**{
                config.plane_coords[0]: plane_vals[0],
                config.plane_coords[1]: plane_vals[1]
            })
            
            missing_val = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
            if missing_val is not None:
                other_vals = [0.0, 0.0]
                missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
                other_vals[missing_idx] = missing_val
                seed = config.build_state((plane_vals[0], plane_vals[1]), (other_vals[0], other_vals[1]))
                seeds.append(seed)
    
    return seeds


def _generate_seeds_level_sets(
    config: _PoincareSectionConfig,
    plane_maxes: List[float],
    n_seeds: int,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray]
) -> List[Tuple[float, float, float, float]]:
    """Generate seeds along level sets (non-zero levels of one coordinate)."""
    seeds = []
    n_levels = max(2, int(np.sqrt(n_seeds)))  # Number of level curves
    seeds_per_level = n_seeds // (2 * n_levels)  # Split between both coordinates
    
    for i, varying_coord in enumerate(config.plane_coords):
        other_coord_idx = 1 - i
        # Create level curves at non-zero values
        level_vals = np.linspace(-0.7 * plane_maxes[other_coord_idx], 0.7 * plane_maxes[other_coord_idx], n_levels + 2)[1:-1]  # Exclude endpoints
        
        for level_val in level_vals:
            if abs(level_val) > 0.05 * plane_maxes[other_coord_idx]:  # Skip near-zero
                varying_vals = np.linspace(-0.8 * plane_maxes[i], 0.8 * plane_maxes[i], seeds_per_level)
                
                for varying_val in varying_vals:
                    plane_vals = [0.0, 0.0]
                    plane_vals[i] = varying_val
                    plane_vals[other_coord_idx] = level_val
                    constraints = config.build_constraint_dict(**{
                        config.plane_coords[0]: plane_vals[0],
                        config.plane_coords[1]: plane_vals[1]
                    })
                    
                    missing_val = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
                    if missing_val is not None:
                        other_vals = [0.0, 0.0]
                        missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
                        other_vals[missing_idx] = missing_val
                        seed = config.build_state((plane_vals[0], plane_vals[1]), (other_vals[0], other_vals[1]))
                        seeds.append(seed)
    
    return seeds


def _generate_seeds_radial(
    config: _PoincareSectionConfig,
    plane_maxes: List[float],
    n_seeds: int,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray]
) -> List[Tuple[float, float, float, float]]:
    """Generate seeds in concentric circles (radial distribution)."""
    seeds = []
    
    # Calculate optimal radial and angular distribution
    max_radius = 0.8 * min(plane_maxes[0], plane_maxes[1])
    n_radial = max(1, int(np.sqrt(n_seeds / (2 * np.pi))))  # Approximate optimal radial levels
    n_angular = max(4, n_seeds // n_radial)  # Angular points per radial level
    
    for i in range(n_radial):
        r = (i + 1) / n_radial * max_radius
        for j in range(n_angular):
            theta = 2 * np.pi * j / n_angular
            plane_val1 = r * np.cos(theta)
            plane_val2 = r * np.sin(theta)
            
            # Check if point is within boundary
            if abs(plane_val1) < plane_maxes[0] and abs(plane_val2) < plane_maxes[1]:
                constraints = config.build_constraint_dict(**{
                    config.plane_coords[0]: plane_val1,
                    config.plane_coords[1]: plane_val2
                })
                
                missing_val = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
                if missing_val is not None:
                    other_vals = [0.0, 0.0]
                    missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
                    other_vals[missing_idx] = missing_val
                    seed = config.build_state((plane_val1, plane_val2), (other_vals[0], other_vals[1]))
                    seeds.append(seed)
                    
                    # Stop if we've generated enough seeds
                    if len(seeds) >= n_seeds:
                        return seeds
    
    return seeds


def _generate_seeds_random(
    config: _PoincareSectionConfig,
    plane_maxes: List[float],
    n_seeds: int,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray]
) -> List[Tuple[float, float, float, float]]:
    """Generate seeds randomly distributed within the Hill boundary."""
    seeds = []
    max_attempts = n_seeds * 10  # Prevent infinite loops
    attempts = 0
    
    while len(seeds) < n_seeds and attempts < max_attempts:
        attempts += 1
        
        # Random point within rectangular boundary
        plane_val1 = np.random.uniform(-0.9 * plane_maxes[0], 0.9 * plane_maxes[0])
        plane_val2 = np.random.uniform(-0.9 * plane_maxes[1], 0.9 * plane_maxes[1])
        
        constraints = config.build_constraint_dict(**{
            config.plane_coords[0]: plane_val1,
            config.plane_coords[1]: plane_val2
        })
        
        missing_val = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
        if missing_val is not None:
            other_vals = [0.0, 0.0]
            missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
            other_vals[missing_idx] = missing_val
            seed = config.build_state((plane_val1, plane_val2), (other_vals[0], other_vals[1]))
            seeds.append(seed)
    
    if len(seeds) < n_seeds:
        logger.warning(f"Only generated {len(seeds)} out of {n_seeds} requested random seeds")
    
    return seeds


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
    parallel: bool = False,  # Enable CPU parallelization
    parallel_seed_finding: bool = True,  # Parallelize seed finding phase
    n_processes: Optional[int] = None,  # Number of processes (default: CPU count)
) -> _PoincareSection:
    logger.info(f"Computing Poincaré map for energy h0={h0:.6e}, grid size: {Nq}x{Np}")
    
    # Get section information
    config = _get_section_config(section_coord)
    
    jac_H = _polynomial_jacobian(poly_p=H_blocks, max_deg=max_degree, psi_table=psi_table,
                                clmo_table=clmo_table, encode_dict_list=encode_dict_list,
            )

    # Find turning points for the plane coordinates
    plane_maxes = []
    for coord in config.plane_coords:
        turning_point = _find_turning(coord, h0, H_blocks, clmo_table)
        plane_maxes.append(turning_point)
    
    logger.info(f"Hill boundary turning points: {config.plane_coords[0]}_max={plane_maxes[0]:.6e}, "
                f"{config.plane_coords[1]}_max={plane_maxes[1]:.6e}")
    
    coord1_vals = np.linspace(-plane_maxes[0], plane_maxes[0], Nq)
    coord2_vals = np.linspace(-plane_maxes[1], plane_maxes[1], Np)

    # Find valid seeds
    logger.info("Finding valid seeds within Hill boundary")
    total_points = Nq * Np
    
    if parallel and parallel_seed_finding and total_points > 100:
        # Parallel seed finding
        if n_processes is None:
            n_processes = mp.cpu_count()
            
        logger.info(f"Using parallel seed finding with {n_processes} threads for {total_points} coordinate pairs")
        
        # Create all coordinate pairs
        coord_pairs = [(coord1, coord2) for coord1 in coord1_vals for coord2 in coord2_vals]
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(coord_pairs) // n_processes)
        coord_chunks = [coord_pairs[i:i + chunk_size] for i in range(0, len(coord_pairs), chunk_size)]
        
        # Process chunks in parallel using threads (avoids Numba pickling issues)
        seeds: list[Tuple[float, float, float, float]] = []
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            for chunk in coord_chunks:
                if chunk:  # Skip empty chunks
                    future = executor.submit(
                        _process_grid_chunk,
                        chunk,
                        h0,
                        H_blocks,
                        clmo_table,
                        section_coord,
                    )
                    futures.append(future)
            
            # Collect results from all processes
            for future in as_completed(futures):
                try:
                    chunk_seeds = future.result()
                    seeds.extend(chunk_seeds)
                except Exception as e:
                    logger.error(f"Error in parallel seed finding: {e}")
                    
        logger.info(f"Parallel seed finding completed. Found {len(seeds)} valid seeds out of {total_points} grid points")
        
    else:
        # Sequential seed finding (original implementation)
        seeds: list[Tuple[float, float, float, float]] = []
        points_checked = 0
        valid_seeds_found = 0
        
        for coord1 in coord1_vals:
            for coord2 in coord2_vals:
                points_checked += 1
                if points_checked % (total_points // 10) == 0:
                    percentage = int(100 * points_checked / total_points)
                    logger.info(f"Seed search progress: {percentage}%, found {valid_seeds_found} valid seeds")
                    
                # Use configuration to build constraints and solve
                constraints = config.build_constraint_dict(**{
                    config.plane_coords[0]: coord1,
                    config.plane_coords[1]: coord2
                })
                
                missing_coord = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
                if missing_coord is not None:
                    # Build the full state
                    other_vals = [0.0, 0.0]
                    missing_idx = 0 if config.missing_coord == config.other_coords[0] else 1
                    other_vals[missing_idx] = missing_coord
                    seed = config.build_state((coord1, coord2), (other_vals[0], other_vals[1]))
                    seeds.append(seed)
                    valid_seeds_found += 1
                        
        logger.info(f"Sequential seed finding completed. Found {len(seeds)} valid seeds out of {total_points} grid points")

    if len(seeds) == 0:
        return _PoincareSection(np.empty((0, 2), dtype=np.float64), config.plane_coords)

    seeds_arr = np.asarray(seeds, dtype=np.float64)

    success_flags, q2p_arr, p2p_arr, q3p_arr, p3p_arr = _poincare_map(seeds_arr, dt, jac_H, clmo_table, integrator_order,
                                                            max_steps, use_symplectic, N_SYMPLECTIC_DOF, section_coord,
                                                        )

    n_success = int(np.sum(success_flags))
    logger.info(f"Completed Poincaré map: {n_success} successful seeds out of {len(seeds)}")

    map_pts = np.empty((n_success, 2), dtype=np.float64)
    idx = 0
    for i in range(success_flags.shape[0]):
        if success_flags[i]:
            # Extract the appropriate coordinates using configuration
            state_6d = np.array([0.0, q2p_arr[i], q3p_arr[i], 0.0, p2p_arr[i], p3p_arr[i]])
            plane_coords = config.extract_plane_coords(state_6d)
            map_pts[idx, 0] = plane_coords[0]
            map_pts[idx, 1] = plane_coords[1]
            idx += 1

    return _PoincareSection(map_pts, config.plane_coords)
