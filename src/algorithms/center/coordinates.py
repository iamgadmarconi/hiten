import numpy as np
from numba.typed import List

from algorithms.center.lie import inverse_lie_transform
from algorithms.center.poincare.map import solve_p3
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                                encode_multiindex)
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import realify, complexify, rn2phys
from utils.log_config import logger


def _realify_coordinates(
    complex_coords: np.ndarray,
    max_degree: int,
    psi: np.ndarray,
    clmo: np.ndarray
) -> np.ndarray:
    complex_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    if len(complex_polys) > 1:
        for i in range(6):
            if abs(complex_coords[i]) > 1e-15:
                # Find position of x_i monomial in degree-1 polynomial
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < complex_polys[1].shape[0]:
                    complex_polys[1][pos] = complex_coords[i]
    
    # Use existing transformation function
    real_polys = realify(complex_polys, max_degree, psi, clmo)
    
    # Extract coordinate values (degree-1 terms)
    # Note: real normal coordinates can have complex coefficients!
    real_coords = np.zeros(6, dtype=np.complex128)
    if len(real_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < real_polys[1].shape[0]:
                real_coords[i] = real_polys[1][pos]
                
    return real_coords


def _complexify_coordinates(
    real_coords: np.ndarray,
    max_degree: int,
    psi: np.ndarray,
    clmo: np.ndarray
) -> np.ndarray:
    # Create polynomial representation of RN coordinates
    real_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    if len(real_polys) > 1:
        for i in range(6):
            if abs(real_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < real_polys[1].shape[0]:
                    real_polys[1][pos] = real_coords[i]
    
    # Use existing transformation function
    complex_polys = complexify(real_polys, max_degree, psi, clmo)
    
    # Extract coordinate values (degree-1 terms)
    complex_coords = np.zeros(6, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]
    if len(complex_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < complex_polys[1].shape[0]:
                complex_coords[i] = complex_polys[1][pos]
    
    return complex_coords


def _rn2phys_coordinates(
    rn_coords: np.ndarray,
    point,
    max_degree: int,
    psi: np.ndarray,
    clmo: np.ndarray
) -> np.ndarray:
    """Convert real normal coordinates to physical coordinates."""
    # Create polynomial representation
    rn_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Set degree-1 terms
    if len(rn_polys) > 1:
        for i in range(6):
            if abs(rn_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < rn_polys[1].shape[0]:
                    rn_polys[1][pos] = rn_coords[i]
    
    phys_polys = rn2phys(point, rn_polys, max_degree, psi, clmo)
    
    # Extract physical coordinates
    phys_coords = np.zeros(6, dtype=np.complex128)
    if len(phys_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < phys_polys[1].shape[0]:
                phys_coords[i] = phys_polys[1][pos]
                
    return phys_coords

def _complete_cm_coordinates(
    poly_cm: List[np.ndarray],
    cm_coords: np.ndarray,
    energy: float,
    clmo: np.ndarray,
) -> np.ndarray:
    q2, p2 = cm_coords

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

    return np.array([q2, p2, 0.0, p3], dtype=np.complex128)

def _cmreal2phys_coordinates(
    point,
    poly_cm: List[np.ndarray],  # RN Hamiltonian for solve_p3
    cm_coords: np.ndarray,      # RN center manifold coordinates [q2, p2]
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    energy: float = 0.0,
    tol: float = 1e-15
) -> np.ndarray:
    if len(cm_coords) == 2:
        real_4d_cm = _complete_cm_coordinates(
            poly_cm, cm_coords, energy, clmo
        ) # [q2, p2, q3, p3]
    else:
        real_4d_cm = cm_coords

    real_6d_cm = np.zeros(6, dtype=np.complex128) # [0, q2, q3, 0, p2, p3]
    real_6d_cm[1] = real_4d_cm[0]  # q2
    real_6d_cm[2] = real_4d_cm[2]  # q3
    real_6d_cm[4] = real_4d_cm[1]  # p2
    real_6d_cm[5] = real_4d_cm[3]  # p3
    
    complex_6d_cm = _complexify_coordinates(real_6d_cm, max_degree, psi, clmo)
    complex_6d_cm = np.array([
        complex_6d_cm[1],  # q2
        complex_6d_cm[4],  # p2
        complex_6d_cm[2],  # q3
        complex_6d_cm[5]   # p3
    ], dtype=np.complex128)
    
    complex_6d = inverse_lie_transform(
        complex_6d_cm, poly_G_total, psi, clmo, max_degree, tol
    )
    
    real_6d = _realify_coordinates(complex_6d, max_degree, psi, clmo)
    local_6d = _rn2phys_coordinates(real_6d, point, max_degree, psi, clmo)
    
    return local_6d


def _local2synodic(point, coords):
    gamma, mu, sgn, a = point.gamma, point.mu, point.sign, point.a

    c = np.asarray(coords, dtype=np.float64)
    single = c.ndim == 1
    if single:
        c = c.reshape(1, -1)

    synodic_6d = np.empty_like(c)

    synodic_6d[:, 0] =  sgn * gamma * c[:, 0] + mu + a      # X
    synodic_6d[:, 1] =  sgn * gamma * c[:, 1]               # Y
    synodic_6d[:, 2] =         gamma * c[:, 2]              # Z

    vx = c[:, 3] + c[:, 1] 
    vy = c[:, 4] - c[:, 0]
    vz = c[:, 5]

    synodic_6d[:, 3] =  sgn * gamma * vx # Vx
    synodic_6d[:, 4] =  sgn * gamma * vy # Vy
    synodic_6d[:, 5] =        gamma * vz # Vz

    synodic_6d[:, (0, 3)] *= -1.0 # flip X and Vx only (NASA/Szebehely convention)

    return synodic_6d.squeeze() if single else synodic_6d


def poincare2ic(
    poincare_points: np.ndarray,
    point,
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int = 8,
    energy: float = 0.0,
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
    
    # Make copies to avoid modifying cached data
    poly_cm_real = [h.copy() for h in poly_cm_real]
    poly_G_total = [g.copy() for g in poly_G_total]
    
    initial_conditions = np.zeros((len(poincare_points), 6))
    
    for i, poincare_point in enumerate(poincare_points):
        try:
            ic = _cmreal2phys_coordinates(
                point=point,
                poly_cm=poly_cm_real,     # For solve_p3
                cm_coords=poincare_point,  # RN coordinates from Poincaré map
                poly_G_total=poly_G_total,
                psi=psi,
                clmo=clmo,
                max_degree=max_degree,
                energy=energy
            )
            initial_conditions[i] = _local2synodic(point, ic)
            
        except Exception as e:
            err = f"Failed to transform point {i}: {poincare_point}, error: {e}"
            logger.error(err)
            raise RuntimeError(err)
    
    logger.info("Completed transformation to initial conditions")
    return initial_conditions
