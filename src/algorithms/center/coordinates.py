import numpy as np
from numba.typed import List

from algorithms.center.lie import _apply_inverse_lie_transforms
from algorithms.center.manifold import center_manifold_cn
from algorithms.center.poincare.map import solve_p3
from algorithms.center.polynomial.base import _create_encode_dict_from_clmo
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import cn2rn, rn2phys
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
                from algorithms.center.polynomial.base import encode_multiindex
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
                from algorithms.center.polynomial.base import encode_multiindex
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
    point,
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int
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
        raise


def _cm2phys_coordinates(
    point,
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

    poly_G_total = point.get_cached_generating_functions(max_degree)
    
    if poly_G_total is None:
        logger.info("Generating functions not cached, computing...")
        # Trigger computation which will cache everything
        _ = center_manifold_cn(point, psi, clmo, max_degree)
        poly_G_total = point.get_cached_generating_functions(max_degree)
        
        if poly_G_total is None:
            raise RuntimeError("Failed to compute generating functions")
    else:
        logger.debug("Using cached generating functions")

    # Step 1: Complete center manifold coordinates if needed
    full_cm_coords = _complete_cm_coordinates(
        cm_coords, energy, point, psi, clmo, max_degree
    )
    
    # Step 2: Apply inverse Lie transforms to get complex normal form
    cn_coords = _apply_inverse_lie_transforms(
        full_cm_coords, poly_G_total, psi, clmo, max_degree, tol
    )
    
    # Step 3: Transform CN → RN → Physical using existing infrastructure
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    physical_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    
    return physical_coords