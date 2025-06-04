import numpy as np
from numba.typed import List

from algorithms.center.lie import inverse_lie_transform
from algorithms.center.poincare.map import solve_p3
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                                encode_multiindex)
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import cn2rn, rn2cn, rn2phys
from utils.log_config import logger


def _cn2rn_coordinates(
    cn_coords: np.ndarray,
    max_degree: int,
    psi: np.ndarray,
    clmo: np.ndarray
) -> np.ndarray:
    """Convert complex normal coordinates to real normal coordinates."""
    # Create a polynomial representation of the CN coordinates
    cn_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Set degree-1 terms to the coordinate values
    if len(cn_polys) > 1:
        for i in range(6):
            if abs(cn_coords[i]) > 1e-15:
                # Find position of x_i monomial in degree-1 polynomial
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < cn_polys[1].shape[0]:
                    cn_polys[1][pos] = cn_coords[i]
    
    # Use existing transformation function
    rn_polys = cn2rn(cn_polys, max_degree, psi, clmo)
    
    # Extract coordinate values (degree-1 terms)
    rn_coords = np.zeros(6, dtype=np.float64)
    if len(rn_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < rn_polys[1].shape[0]:
                rn_coords[i] = rn_polys[1][pos].real
                
    return rn_coords


def _rn2cn_coordinates(
    rn_coords: np.ndarray,
    max_degree: int,
    psi: np.ndarray,
    clmo: np.ndarray
) -> np.ndarray:
    """Convert real normal coordinates to complex normal coordinates."""
    # Create polynomial representation of RN coordinates
    rn_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Set degree-1 terms to the coordinate values
    if len(rn_polys) > 1:
        for i in range(6):
            if abs(rn_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < rn_polys[1].shape[0]:
                    rn_polys[1][pos] = rn_coords[i]
    
    # Use existing transformation function
    cn_polys = rn2cn(rn_polys, max_degree, psi, clmo)
    
    # Extract coordinate values (degree-1 terms)
    cn_coords = np.zeros(6, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]
    if len(cn_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < cn_polys[1].shape[0]:
                cn_coords[i] = cn_polys[1][pos]
    
    return cn_coords


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
    phys_coords = np.zeros(6, dtype=np.float64)
    if len(phys_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < phys_polys[1].shape[0]:
                phys_coords[i] = phys_polys[1][pos].real
                
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


def _cm_cn2phys_coordinates(
    point,
    poly_cm: List[np.ndarray],
    cm_coords: np.ndarray,
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    energy: float = 0.0,
    tol: float = 1e-15
) -> np.ndarray:
    if len(cm_coords) == 2:
        full_cm_coords = _complete_cm_coordinates(
            poly_cm, cm_coords, energy, clmo,
        ) # [q2, p2, q3, p3]
    else:
        full_cm_coords = cm_coords
    
    cn_coords = inverse_lie_transform(
        full_cm_coords, poly_G_total, psi, clmo, max_degree, tol
    )
    
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    physical_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    
    return physical_coords


def _cm_rn2phys_coordinates(
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
        full_cm_coords_rn = _complete_cm_coordinates(
            poly_cm, cm_coords, energy, clmo
        ) # [q2, p2, q3, p3]
    else:
        full_cm_coords_rn = cm_coords

    rn_coords_6d = np.zeros(6, dtype=np.complex128) # [0, q2, q3, 0, p2, p3]
    rn_coords_6d[1] = full_cm_coords_rn[0]  # q2
    rn_coords_6d[2] = full_cm_coords_rn[2]  # q3
    rn_coords_6d[4] = full_cm_coords_rn[1]  # p2
    rn_coords_6d[5] = full_cm_coords_rn[3]  # p3
    
    cn_coords_6d = _rn2cn_coordinates(rn_coords_6d, max_degree, psi, clmo)
    
    full_cm_coords_cn = np.array([
        cn_coords_6d[1],  # q2
        cn_coords_6d[4],  # p2
        cn_coords_6d[2],  # q3
        cn_coords_6d[5]   # p3
    ], dtype=np.complex128)
    
    cn_coords = inverse_lie_transform(
        full_cm_coords_cn, poly_G_total, psi, clmo, max_degree, tol
    )
    
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    physical_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    
    return physical_coords


def _local2synodic(point, coords):
    gamma, mu, sgn, a = point.gamma, point.mu, point.sign, point.a

    c = np.asarray(coords, dtype=np.float64)
    single = c.ndim == 1
    if single:
        c = c.reshape(1, -1)

    out = np.empty_like(c)

    out[:, 0] =  sgn * gamma * c[:, 0] + mu + a      # X
    out[:, 1] =  sgn * gamma * c[:, 1]               # Y
    out[:, 2] =         gamma * c[:, 2]              # Z

    vx = c[:, 3] + c[:, 1] 
    vy = c[:, 4] - c[:, 0]
    vz = c[:, 5]

    out[:, 3] =  sgn * gamma * vx  # Vx
    out[:, 4] =  sgn * gamma * vy  # Vy
    out[:, 5] =        gamma * vz  # Vz

    out[:, (0, 3)] *= -1.0     # flip X and Vx only (NASA/Szebehely convention)

    return out.squeeze() if single else out


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
    
    # Get both RN and CN Hamiltonians
    poly_cm_rn = point.get_cached_hamiltonian(max_degree, "center_manifold_rn")
    poly_G_total = point.get_cached_generating_functions(max_degree)
    
    initial_conditions = np.zeros((len(poincare_points), 6))
    
    for i, poincare_point in enumerate(poincare_points):
        try:
            ic = _cm_rn2phys_coordinates(
                point=point,
                poly_cm=poly_cm_rn,     # For solve_p3
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
