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
    cn_coords = np.zeros(6, dtype=np.complex128)
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
    """Complete center manifold coordinates using energy constraint."""
    if len(cm_coords) == 4:
        return cm_coords.astype(np.complex128)
    elif len(cm_coords) == 2:
        # Poincaré section case: solve for p3 using existing infrastructure
        q2, p2 = cm_coords
        q3 = 0.0
        
        # Use existing solve_p3 function from map.py
        p3 = solve_p3(
            q2=float(q2), 
            p2=float(p2), 
            h0=energy, 
            H_blocks=poly_cm, 
            clmo=clmo
        )
        
        if p3 is None:
            p3 = 0.0
            
        return np.array([q2, p2, q3, p3], dtype=np.complex128)
    else:
        err = f"Invalid CM coordinates length: {len(cm_coords)}, expected 2 or 4. Shape: {cm_coords.shape}, Contents: {cm_coords}"
        logger.error(err)
        raise ValueError(err)


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
    """
    Transform coordinates from center manifold back to physical (rotating) frame.
    
    This reverses the transformation pipeline used in center_manifold_cn:
    Physical → RN → CN → Normalized CN → Center Manifold
    
    We apply the inverse: Center Manifold → CN → RN → Physical
    
    Parameters
    ----------
    point : object
        Point object containing equilibrium point information and transformations
    cm_coords : numpy.ndarray
        Center manifold coordinates, either [q2, p2] for Poincaré section 
        or [q2, p2, q3, p3] for full 4D coordinates
    poly_G_total : List[numpy.ndarray]
        Generating functions from lie_transform normalization
    psi, clmo : arrays
        Polynomial index tables
    max_degree : int
        Maximum degree used in normalization
    energy : float, optional
        Energy level for completing missing coordinates
    tol : float, optional
        Tolerance for cleaning small coefficients
        
    Returns
    -------
    numpy.ndarray
        Physical coordinates [X, Y, Z, PX, PY, PZ]
    """
    # Step 1: Complete center manifold coordinates if needed
    full_cm_coords = _complete_cm_coordinates(
        poly_cm, cm_coords, energy, clmo,
    )
    
    # Step 2: Apply inverse Lie transforms to get complex normal form
    cn_coords = inverse_lie_transform(
        full_cm_coords, poly_G_total, psi, clmo, max_degree, tol
    )
    
    # Step 3: Transform CN → RN → Physical using existing infrastructure
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    physical_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    
    return physical_coords


def _cm_rn2phys_coordinates(
    point,
    poly_cm_rn: List[np.ndarray],  # RN Hamiltonian for solve_p3
    poly_cm_cn: List[np.ndarray],  # CN Hamiltonian for Lie transforms
    cm_coords_rn: np.ndarray,      # RN center manifold coordinates [q2, p2]
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    energy: float = 0.0,
    tol: float = 1e-15
) -> np.ndarray:
    """
    Transform from RN center manifold coordinates to physical coordinates.
    
    Pipeline: CM(RN) → CM(CN) → CN → RN → Physical
    """
    # Step 1: Complete RN center manifold coordinates using RN Hamiltonian
    if len(cm_coords_rn) == 2:
        q2, p2 = cm_coords_rn
        q3 = 0.0
        
        p3 = solve_p3(
            q2=float(q2),
            p2=float(p2), 
            h0=energy,
            H_blocks=poly_cm_rn,  # Use RN Hamiltonian
            clmo=clmo
        )
        
        if p3 is None:
            raise ValueError(f"solve_p3 failed for q2={q2}, p2={p2}, energy={energy}")
            
        full_cm_coords_rn = np.array([q2, p2, q3, p3], dtype=np.float64)
    else:
        full_cm_coords_rn = cm_coords_rn.astype(np.float64)
    
    # Step 2: Convert RN center manifold coordinates to CN center manifold coordinates
    # Map 4D CM coords to 6D RN coords: [0, q2, q3, 0, p2, p3]
    rn_coords_6d = np.zeros(6, dtype=np.float64)
    rn_coords_6d[1] = full_cm_coords_rn[0]  # q2
    rn_coords_6d[2] = full_cm_coords_rn[2]  # q3
    rn_coords_6d[4] = full_cm_coords_rn[1]  # p2
    rn_coords_6d[5] = full_cm_coords_rn[3]  # p3
    
    # Convert to CN coordinates
    cn_coords_6d = _rn2cn_coordinates(rn_coords_6d, max_degree, psi, clmo)
    
    # Extract CN center manifold coordinates: [q2, p2, q3, p3]
    full_cm_coords_cn = np.array([
        cn_coords_6d[1],  # q2
        cn_coords_6d[4],  # p2
        cn_coords_6d[2],  # q3
        cn_coords_6d[5]   # p3
    ], dtype=np.complex128)
    
    # Step 3: Apply inverse Lie transforms (existing function)
    cn_coords = inverse_lie_transform(
        full_cm_coords_cn, poly_G_total, psi, clmo, max_degree, tol
    )
    
    # Step 4: CN → RN → Physical (existing functions)
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    physical_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    
    return physical_coords


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
    poly_cm_cn = point.get_cached_hamiltonian(max_degree, "center_manifold_cn") 
    poly_G_total = point.get_cached_generating_functions(max_degree)
    
    initial_conditions = np.zeros((len(poincare_points), 6))
    
    for i, poincare_point in enumerate(poincare_points):
        try:
            ic = _cm_rn2phys_coordinates(
                point=point,
                poly_cm_rn=poly_cm_rn,     # For solve_p3
                poly_cm_cn=poly_cm_cn,     # For Lie transforms
                cm_coords_rn=poincare_point,  # RN coordinates from Poincaré map
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
