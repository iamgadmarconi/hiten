import numpy as np
from numba.typed import List

from algorithms.center.lie import (evaluate_inverse_transform,
                                   inverse_lie_transform)
from algorithms.center.poincare.map import solve_p3
from utils.log_config import logger


def _clean_coordinates(coords: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Clean tiny numerical artifacts from complex coordinates."""
    before = np.asarray(coords, dtype=np.complex128)
    after = np.where(np.abs(before) <= tol, 0.0, before)

    if np.any(before != after):
        logger.warning(f"Cleaned {np.sum(before != after)} coordinates. \nBefore: {before}\nAfter: {after}")

    return after

def _realify_coordinates(complex_coords: np.ndarray) -> np.ndarray:
    c = np.asarray(complex_coords, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]

    q1c, q2c, q3c, p1c, p2c, p3c = c # [q1, q2, q3, p1, p2, p3]

    q1r = q1c
    q2r = (q2c - 1j * p2c) / np.sqrt(2)
    q3r = (q3c - 1j * p3c) / np.sqrt(2)
    p1r = p1c
    p2r = (-1j * q2c + p2c) / np.sqrt(2)
    p3r = (-1j * q3c + p3c) / np.sqrt(2)

    return _clean_coordinates(np.array([q1r, q2r, q3r, p1r, p2r, p3r], dtype=np.complex128)) # [q1, q2, q3, p1, p2, p3]

def _complexify_coordinates(real_coords: np.ndarray) -> np.ndarray:
    r = np.asarray(real_coords, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]

    q1r, q2r, q3r, p1r, p2r, p3r = r # [q1, q2, q3, p1, p2, p3]

    q1c = q1r
    q2c = (q2r + 1j * p2r) / np.sqrt(2)
    q3c = (q3r + 1j * p3r) / np.sqrt(2)
    p1c = p1r
    p2c = (1j * q2r + p2r) / np.sqrt(2)
    p3c = (1j * q3r + p3r) / np.sqrt(2)

    return _clean_coordinates(np.array([q1c, q2c, q3c, p1c, p2c, p3c], dtype=np.complex128)) # [q1, q2, q3, p1, p2, p3]

def _realmodal2local_coordinates(point, modal_coords: np.ndarray) -> np.ndarray:
    # modal_coords: [q1, q2, q3, p1, p2, p3]
    _, Cinv = point.normal_form_transform()
    return _clean_coordinates(modal_coords @ Cinv.T) # [x1, x2, x3, px1, px2, px3]

def _local2synodic(point, coords: np.ndarray) -> np.ndarray:
    # coords: [x1, x2, x3, px1, px2, px3] - local coordinates
    gamma, mu, sgn, a = point.gamma, point.mu, point.sign, point.a

    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 1 or c.size != 6:
        raise ValueError(
            f"coords must be a flat array of 6 elements, got shape {c.shape}"
        )

    syn = np.empty(6, dtype=np.float64) # [X, Y, Z, Vx, Vy, Vz]

    # Positions
    syn[0] =  sgn * gamma * c[0] + mu + a      # X
    syn[1] =  sgn * gamma * c[1]               # Y
    syn[2] =        gamma * c[2]               # Z

    # Local momenta to synodic velocities (see standard relations)
    vx = c[3] + c[1]
    vy = c[4] - c[0]
    vz = c[5]

    syn[3] = sgn * gamma * vx  # Vx
    syn[4] = sgn * gamma * vy  # Vy
    syn[5] =       gamma * vz  # Vz

    # Flip X and Vx according to NASA/Szebehely convention
    syn[[0, 3]] *= -1.0

    return syn # [X, Y, Z, Vx, Vy, Vz]

def _complete_cm_coordinates(
    poly_cm: List[np.ndarray],
    cm_coords: np.ndarray, # [q2, p2]
    energy: float,
    clmo: np.ndarray,
) -> np.ndarray:
    q2, p2 = cm_coords # [q2, p2]

    p3 = solve_p3(
        q2=float(q2), 
        p2=float(p2), 
        h0=energy, 
        H_blocks=poly_cm, 
        clmo=clmo
    )
    
    if p3 is None or p3 < 0:
        err = f"solve_p3 failed for q2={q2}, p2={p2}, energy={energy}"
        logger.error(err)
        raise ValueError(err)

    return np.array([q2, p2, 0.0, p3], dtype=np.complex128) # [q2, p2, q3, p3]

def _cmreal2synodic_coordinates(
    point,
    poly_cm: List[np.ndarray],  # RN Hamiltonian for solve_p3
    cm_coords: np.ndarray,      # RN center manifold coordinates [q2, p2]
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    energy: float = 0.0,
    tol: float = 1e-30
) -> np.ndarray:
    logger.info(f"Converting points {cm_coords} to synodic coordinates")

    real_4d_cm = _complete_cm_coordinates(poly_cm, cm_coords, energy, clmo) # [q2, p2, q3, p3]

    logger.info(f"real_4d_cm: \n{real_4d_cm}")

    real_6d_cm = np.zeros(6, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]
    real_6d_cm[1] = real_4d_cm[0] # q2 = q2
    real_6d_cm[2] = real_4d_cm[2] # q3 = 0.0 (q3 is zero on center manifold)
    real_6d_cm[4] = real_4d_cm[1] # p2 = p2
    real_6d_cm[5] = real_4d_cm[3] # p3 = p3

    logger.info(f"real_6d_cm: \n{real_6d_cm}")

    complex_6d_cm = _complexify_coordinates(real_6d_cm) # [q1, q2, q3, p1, p2, p3]

    logger.info(f"complex_6d_cm: \n{complex_6d_cm}")

    expansions = inverse_lie_transform(
        poly_G_total, max_degree, psi, clmo, tol
    )
    # logger.info(f"expansions: \n\n{expansions}\n\n") # Verbose

    complex_6d = evaluate_inverse_transform(expansions, complex_6d_cm, clmo) # [q1, q2, q3, p1, p2, p3]

    logger.info(f"complex_6d: \n{complex_6d}")

    real_6d = _realify_coordinates(complex_6d) # [q1, q2, q3, p1, p2, p3]

    logger.info(f"real_6d: \n{real_6d}")

    local_6d = _realmodal2local_coordinates(point, real_6d) # [x1, x2, x3, px1, px2, px3]

    logger.info(f"local_6d: \n{local_6d}")

    synodic_6d = _local2synodic(point, local_6d) # [X, Y, Z, Vx, Vy, Vz]

    logger.info(f"synodic_6d: \n{synodic_6d}")
    return synodic_6d


def poincare2ic(
    poincare_points: np.ndarray,
    point,
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    energy: float,
) -> np.ndarray:
    """Convert Poincaré section points to initial conditions in physical coordinates."""
    logger.info(f"Converting {len(poincare_points)} Poincaré points to initial conditions\n\nPoint: {point}\nEnergy: {energy}\nSystem mu: {point.mu}\nMax degree: {max_degree}\n")
    
    # Get both RN and CN Hamiltonians using existing cache access
    poly_cm_real = point.cache_get(('hamiltonian', max_degree, 'center_manifold_real'))
    poly_G_total = point.cache_get(('generating_functions', max_degree))
    
    if poly_cm_real is None:
        err = f"Center manifold real Hamiltonian not cached for max_degree={max_degree}. Call center_manifold_real() first."
        logger.error(err)
        raise ValueError(err)
        
    if poly_G_total is None:
        err = f"Generating functions not cached for max_degree={max_degree}. Call center_manifold computation first."
        logger.error(err)
        raise ValueError(err)
    
    poly_cm_real = [h.copy() for h in poly_cm_real]
    poly_G_total = [g.copy() for g in poly_G_total]
    
    initial_conditions = np.zeros((len(poincare_points), 6))
    
    for i, poincare_point in enumerate(poincare_points):
        try:
            ic = _cmreal2synodic_coordinates(
                point=point,
                poly_cm=poly_cm_real,     # For solve_p3
                cm_coords=poincare_point,  # RN coordinates from Poincaré map
                poly_G_total=poly_G_total,
                psi=psi,
                clmo=clmo,
                max_degree=max_degree,
                energy=energy
            )
            initial_conditions[i] = ic
            
        except Exception as e:
            err = f"Failed to transform point {i}: {poincare_point}, error: {e}"
            logger.error(err)
            raise RuntimeError(err)
    
    logger.info("Completed transformation to initial conditions")
    return initial_conditions
