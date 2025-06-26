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
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

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
from hiten.algorithms.poincare.config import _get_section_config
from hiten.algorithms.poincare.seeding import _make_strategy
from hiten.algorithms.polynomial.operations import (_polynomial_evaluate,
                                                    _polynomial_jacobian)
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


class _PoincareSection(NamedTuple):
    points: np.ndarray  # shape (n, 2) 
    labels: tuple[str, str]  # coordinate labels for the two columns


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
        c_A, c_B, c_C = _get_rk_coefficients(order)
        traj = _integrate_map(y0=state_old, t_vals=np.array([0.0, dt]), A=c_A, B=c_B, C=c_C, jac_H=jac_H, 
                            clmo_H=clmo, order=order, c_omega_heuristic=c_omega_heuristic, use_symplectic=use_symplectic)
        state_new = traj[1]

        rhs_new = _hamiltonian_rhs(state_new, jac_H, clmo, n_dof)
        crossed, alpha = _detect_crossing(section_coord, state_old, state_new, rhs_new, n_dof)

        if crossed:
            rhs_old = _hamiltonian_rhs(state_old, jac_H, clmo, n_dof)

            q2p = _hermite_scalar(alpha, state_old[1],       state_new[1],       rhs_old[1],       rhs_new[1],       dt)
            p2p = _hermite_scalar(alpha, state_old[n_dof+1], state_new[n_dof+1], rhs_old[n_dof+1], rhs_new[n_dof+1], dt)
            q3p = _hermite_scalar(alpha, state_old[2],       state_new[2],       rhs_old[2],       rhs_new[2],       dt)
            p3p = _hermite_scalar(alpha, state_old[n_dof+2], state_new[n_dof+2], rhs_old[n_dof+2], rhs_new[n_dof+2], dt)

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

def _generate_seeds(
    section_coord: str,
    h0: float,
    H_blocks: List[np.ndarray],
    clmo_table: List[np.ndarray],
    n_seeds: int,
    seed_strategy: str = "axis_aligned",  # "single", "axis_aligned", "level_sets", "radial", "random"
    seed_axis: Optional[str] = None,  # Which axis to seed on for "single" strategy
) -> List[Tuple[float, float, float, float]]:
    logger.info(
        "Generating %d seeds with strategy '%s' for %s section",
        n_seeds,
        seed_strategy,
        section_coord,
    )
    
    config = _get_section_config(section_coord)

    # Extra keyword arguments specific to some strategies
    extra_kwargs: dict[str, Any] = {}
    if seed_strategy == "single":
        extra_kwargs["seed_axis"] = seed_axis

    strategy = _make_strategy(seed_strategy, config, n_seeds=n_seeds, **extra_kwargs)

    seeds = strategy.generate(
        h0=h0,
        H_blocks=H_blocks,
        clmo_table=clmo_table,
        solve_missing_coord_fn=_solve_missing_coord,
    )

    logger.info("Generated %d seeds using '%s' strategy", len(seeds), seed_strategy)
    return seeds

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
                new_state_6d = np.array([0.0, q2p, q3p, 0.0, p2p, p3p])  # (q1, q2, q3, p1, p2, p3)
                plane_coords = config.extract_plane_coords(new_state_6d)
                pts_accum.append(plane_coords)
                
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
        constraints = config.build_constraint_dict(**{
            config.plane_coords[0]: coord1,
            config.plane_coords[1]: coord2
        })
        
        missing_coord = _solve_missing_coord(config.missing_coord, constraints, h0, H_blocks, clmo_table)
        if missing_coord is not None:
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
    c_omega_heuristic: float = 20.0,
    section_coord: str = "q3",  # "q2", "p2", "q3", or "p3"
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

    # Always use parallel processing
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

    if len(pts_accum) == 0:
        # Return empty array with correct shape
        points_array = np.empty((0, 2), dtype=np.float64)
    else:
        points_array = np.asarray(pts_accum, dtype=np.float64)
    return _PoincareSection(points_array, config.plane_coords)

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
    
    # Always use parallel seed finding
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

    if len(seeds) == 0:
        return _PoincareSection(np.empty((0, 2), dtype=np.float64), config.plane_coords)

    seeds_arr = np.asarray(seeds, dtype=np.float64)

    success_flags, q2p_arr, p2p_arr, q3p_arr, p3p_arr = _poincare_map(
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
            state_6d = np.array([0.0, q2p_arr[i], q3p_arr[i], 0.0, p2p_arr[i], p3p_arr[i]])
            plane_coords = config.extract_plane_coords(state_6d)
            map_pts[idx, 0] = plane_coords[0]
            map_pts[idx, 1] = plane_coords[1]
            idx += 1

    return _PoincareSection(map_pts, config.plane_coords)

@njit(cache=False, fastmath=FASTMATH)
def _integrate_map(y0: np.ndarray, t_vals: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                jac_H: List[List[np.ndarray]], clmo_H: List[np.ndarray], order: int, c_omega_heuristic: float=20.0, use_symplectic: bool=False) -> np.ndarray:
    
    if use_symplectic:
        traj =  _integrate_symplectic(y0, t_vals, jac_H, clmo_H, order, c_omega_heuristic)
    else:
        traj = _integrate_rk_ham(y0, t_vals, A, B, C, jac_H, clmo_H)
    
    return traj

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

@njit(cache=False, fastmath=FASTMATH, inline="always")
def _hermite_scalar(alpha: float, y0: float, y1: float, dy0: float, dy1: float, dt: float) -> float:
    h00 = (1.0 + 2.0 * alpha) * (1.0 - alpha) ** 2
    h10 = alpha * (1.0 - alpha) ** 2
    h01 = alpha ** 2 * (3.0 - 2.0 * alpha)
    h11 = alpha ** 2 * (alpha - 1.0)
    return h00 * y0 + h10 * dy0 * dt + h01 * y1 + h11 * dy1 * dt
