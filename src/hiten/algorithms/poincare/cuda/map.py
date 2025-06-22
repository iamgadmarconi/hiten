import math
from typing import List, Tuple

import numpy as np

from hiten.algorithms.poincare.cuda.step import PoincareMapCUDA
from hiten.algorithms.poincare.map import _find_turning, _solve_missing_coord, section_closure, PoincareSection
from hiten.algorithms.polynomial.operations import polynomial_jacobian
from hiten.utils.log_config import logger


def _generate_map_gpu(
    h0: float,
    H_blocks: List[np.ndarray],
    max_degree: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    n_seeds: int = 20,
    n_iter: int = 1500,
    dt: float = 1e-2,
    use_symplectic: bool = False,  # Currently only RK4 is implemented
    integrator_order: int = 6,
    c_omega_heuristic: float = 20.0,
    seed_axis: str = "q2",  # "q2" or "p2"
    section_coord: str = "q3",  # "q2", "p2", "q3", or "p3"
) -> PoincareSection:
    """
    GPU-accelerated version of _generate_map.
    
    Parameters
    ----------
    h0 : float
        Energy level.
    H_blocks, max_degree, psi_table, clmo_table, encode_dict_list
        Same polynomial data as original function.
    n_seeds : int, optional
        Number of initial seeds to distribute along the chosen axis.
    n_iter : int, optional
        How many Poincaré iterates to compute for each seed.
    dt : float, optional
        Timestep for the integrator.
    use_symplectic : bool, optional
        Currently only False (RK4) is supported. Symplectic integrator 
        would need to be ported to GPU.
    seed_axis : {"q2", "p2"}
        Place seeds on this axis with the other momentum/position set to zero.
    section_coord : {"q2", "p2", "q3", "p3"}
        The coordinate that defines the section.

    Returns
    -------
    PoincareSection
        Collected section points with appropriate labels.
    """
    if use_symplectic:
        logger.warning("Symplectic integrator not yet implemented on GPU, using RK4")
    
    # Get section information
    section_idx, direction_sign, labels = section_closure(section_coord)
    
    # 1. Build Jacobian once (CPU)
    logger.info("Building polynomial Jacobian")
    jac_H = polynomial_jacobian(
        poly_p=H_blocks,
        max_deg=max_degree,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list,
    )

    # 2. Generate seeds based on section type (CPU)
    logger.info("Finding Hill boundary turning points")
    seeds: List[Tuple[float, float, float, float]] = []
    
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

    logger.info("Generated %d valid seeds (%s-axis) for %d crossings each", 
                len(seeds), seed_axis, n_iter)

    # 3. Calculate max_steps based on dt
    target_max_integration_time_per_crossing = 20.0
    calculated_max_steps = int(math.ceil(target_max_integration_time_per_crossing / dt))
    logger.info(f"Using dt={dt:.1e}, calculated max_steps per crossing: {calculated_max_steps}")

    # 4. Convert seeds to numpy array
    if len(seeds) == 0:
        logger.warning("No valid seeds found")
        return PoincareSection(np.empty((0, 2), dtype=np.float64), labels)
    
    seeds_array = np.array(seeds, dtype=np.float64)
    
    # 5. Initialize GPU Poincaré map calculator
    logger.info("Initializing GPU computation")
    poincare_map = PoincareMapCUDA(jac_H, clmo_table)
    
    # 6. Run GPU computation
    logger.info("Starting GPU Poincaré map iteration")
    points = poincare_map.iterate_map(
        seeds_array, 
        n_iterations=n_iter,
        dt=dt,
        max_steps=calculated_max_steps,
        section_coord=section_coord
    )
    
    logger.info("GPU computation complete: generated %d points from %d seeds", 
                points.shape[0], len(seeds))
    
    return PoincareSection(points, labels)
