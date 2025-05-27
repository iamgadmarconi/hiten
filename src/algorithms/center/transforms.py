import math

import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.poincare.map import solve_p3
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex)
from algorithms.center.polynomial.operations import (polynomial_add_inplace,
                                                     polynomial_clean,
                                                     polynomial_multiply,
                                                     polynomial_power,
                                                     polynomial_variable,
                                                     polynomial_zero_list)
from config import FASTMATH
from utils.log_config import logger


@njit(fastmath=FASTMATH, cache=True)
def _linear_variable_polys(C: np.ndarray, max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
    """
    Create polynomials for new variables after a linear transformation.
    
    Parameters
    ----------
    C : numpy.ndarray
        Transformation matrix (6x6) that defines the linear change of variables
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
    encode_dict_list : numba.typed.List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    List[List[numpy.ndarray]]
        List of length 6 where each element is a polynomial representing 
        a transformed variable
        
    Notes
    -----
    This function computes the linear transformation of variables:
    L_i = ∑_j C[i,j] * var_j
    where var_j are the original variables and L_i are the transformed variables.
    """
    new_basis = [polynomial_variable(j, max_deg, psi, clmo, encode_dict_list) for j in range(6)]
    L: List[np.ndarray] = []
    for i in range(6):
        poly_result = polynomial_zero_list(max_deg, psi)
        for j in range(6):
            if C[i, j] == 0:
                continue
            polynomial_add_inplace(poly_result, new_basis[j], C[i, j], max_deg)
        L.append(poly_result)
    return L


@njit(fastmath=FASTMATH)
def substitute_linear(poly_old: List[np.ndarray], C: np.ndarray, max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
    """
    Perform variable substitution in a polynomial using a linear transformation.
    
    Parameters
    ----------
    poly_old : List[numpy.ndarray]
        Polynomial in the original variables
    C : numpy.ndarray
        Transformation matrix (6x6) that defines the linear change of variables
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
    encode_dict_list : numba.typed.List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in the transformed variables
        
    Notes
    -----
    This function substitutes each original variable with its corresponding
    transformation defined by the matrix C. For each term in the original
    polynomial, it computes the product of the transformed variables raised
    to the appropriate power.
    """
    var_polys = _linear_variable_polys(C, max_deg, psi, clmo, encode_dict_list)
    poly_new = polynomial_zero_list(max_deg, psi)

    for deg in range(max_deg + 1):
        p = poly_old[deg]
        if not p.any():
            continue
        for pos, coeff in enumerate(p):
            if coeff == 0:
                continue
            k = decode_multiindex(pos, deg, clmo)
            
            # build product  Π_i  (var_polys[i] ** k_i)
            term = polynomial_zero_list(max_deg, psi)
            
            # Fix: Preserve the full complex value instead of just the real part
            if len(term) > 0 and term[0].size > 0:
                term[0][0] = coeff
            elif coeff !=0:
                pass
                
            for i_var in range(6):
                if k[i_var] == 0:
                    continue
                pwr = polynomial_power(var_polys[i_var], k[i_var], max_deg, psi, clmo, encode_dict_list)
                term = polynomial_multiply(term, pwr, max_deg, psi, clmo, encode_dict_list)
                
            polynomial_add_inplace(poly_new, term, 1.0, max_deg)

    return polynomial_clean(poly_new, 1e-14)


def phys2rn(point, poly_phys: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Transform a polynomial from physical coordinates to real normal form.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    poly_phys : List[numpy.ndarray]
        Polynomial in physical coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in resonant normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from physical space coordinates to
    resonant normal form coordinates using the transformation matrix obtained
    from the point object.
    """
    C, _ = point.normal_form_transform()
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return substitute_linear(poly_phys, C, max_deg, psi, clmo, encode_dict_list)


@njit(fastmath=FASTMATH)
def rn2cn(poly_rn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Transform a polynomial from real normal form to complex normal form.
    
    Parameters
    ----------
    poly_rn : List[numpy.ndarray]
        Polynomial in real normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in complex normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from real normal form coordinates
    (x, y, z, px, py, pz) to complex normal form coordinates (q1, q2, q3, p1, p2, p3)
    using a specific transformation matrix.
    
    The transformation is:
    y_rn = (q2 + i*p2)/√2
    z_rn = (q3 + i*p3)/√2
    py_rn = (i*q2 + p2)/√2
    pz_rn = (i*q3 + p3)/√2
    with x_rn = q1 and px_rn = p1 unchanged.
    """
    sqrt2 = math.sqrt(2.0)
    C = np.zeros((6, 6), dtype=np.complex128)

    # columns: [q1 q2 q3 p1 p2 p3]
    # rows old: [x  y  z  px py pz]_rn
    C[0, 0] = 1.0                              # x_rn = q1
    C[1, 1] = 1/sqrt2                          # y_rn
    C[1, 4] = 1j/sqrt2
    C[2, 2] = 1/sqrt2                          # z_rn
    C[2, 5] = 1j/sqrt2
    C[3, 3] = 1.0                              # px_rn = p1
    C[4, 1] = 1j/sqrt2                         # py_rn
    C[4, 4] = 1/sqrt2
    C[5, 2] = 1j/sqrt2                         # pz_rn
    C[5, 5] = 1/sqrt2

    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return substitute_linear(poly_rn, C, max_deg, psi, clmo, encode_dict_list)


@njit(fastmath=FASTMATH)
def cn2rn(poly_cn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Transform a polynomial from complex normal form to real normal form.
    
    Parameters
    ----------
    poly_cn : List[numpy.ndarray]
        Polynomial in complex normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in real normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from complex normal form coordinates
    (q1, q2, q3, p1, p2, p3) to real normal form coordinates (x, y, z, px, py, pz)
    using the inverse of the transformation used in rn2cn.
    
    The transformation is:
    q2 = (y_rn + i*py_rn)/√2
    q3 = (z_rn + i*pz_rn)/√2
    p2 = (py_rn - i*y_rn)/√2  
    p3 = (pz_rn - i*z_rn)/√2
    with q1 = x_rn and p1 = px_rn unchanged.
    """
    sqrt2 = math.sqrt(2.0)
    Cinv = np.zeros((6, 6), dtype=np.complex128)

    # columns: [q1 q2 q3 p1 p2 p3]
    # rows   : [x y z px py pz]_rn   expressed in complex vars
    Cinv[0, 0] = 1.0                             # x = q1
    Cinv[1, 1] = 1/sqrt2                         # y   = (q2 - i p2)/√2
    Cinv[1, 4] = -1j/sqrt2
    Cinv[2, 2] = 1/sqrt2                         # z
    Cinv[2, 5] = -1j/sqrt2
    Cinv[3, 3] = 1.0                             # px
    Cinv[4, 1] = -1j/sqrt2                       # py
    Cinv[4, 4] = 1/sqrt2
    Cinv[5, 2] = -1j/sqrt2                       # pz
    Cinv[5, 5] = 1/sqrt2

    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return substitute_linear(poly_cn, Cinv, max_deg, psi, clmo, encode_dict_list)


def rn2phys(point, poly_rn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Transform a polynomial from real normal form to physical coordinates.
    """
    _, Cinv = point.normal_form_transform()
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return substitute_linear(poly_rn, Cinv, max_deg, psi, clmo, encode_dict_list)


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



@njit(fastmath=FASTMATH, cache=True)
def _evaluate_reduced_monomial(
    k: np.ndarray,
    coords: np.ndarray, 
    var_idx: int,
    exp_change: int
) -> np.complex128:
    """Evaluate monomial x^k at coords with exponent of var_idx changed by exp_change."""
    result = 1.0 + 0.0j
    
    for i in range(6):
        exp = k[i]
        if i == var_idx:
            exp += exp_change
            
        if exp > 0:
            if abs(coords[i]) > 1e-15:
                result *= coords[i] ** exp
            else:
                return 0.0 + 0.0j  # Zero coordinate with positive exponent
        # exp == 0: multiply by 1 (no-op)
                
    return result


@njit(fastmath=FASTMATH, cache=True)
def _compute_pb_coord(
    G_coeffs: np.ndarray,
    degree: int,
    coord_idx: int,
    coords: np.ndarray,
    clmo: np.ndarray,
    encode_dict_list: List
) -> np.complex128:
    """
    Compute {G, x_coord_idx} evaluated at the given coordinates.
    
    For a monomial G_term = coeff * q^a * p^b:
    {G_term, q_i} = -coeff * b_i * q^a * p^(b with p_i reduced by 1)  
    {G_term, p_i} = +coeff * a_i * q^(a with q_i reduced by 1) * p^b
    """

    result = 0.0 + 0.0j
    
    for pos in range(G_coeffs.shape[0]):
        coeff = G_coeffs[pos]
        if abs(coeff) < 1e-15:
            continue
            
        # Decode the multi-index for this monomial
        k = decode_multiindex(pos, degree, clmo)
        
        if coord_idx < 3:  # Position coordinate q_i
            # {G_term, q_i} = -k[i+3] * G_term with p_i exponent reduced by 1
            p_idx = coord_idx + 3
            if k[p_idx] > 0:
                # Evaluate monomial with p_i exponent reduced by 1
                monomial_val = _evaluate_reduced_monomial(k, coords, p_idx, -1)
                result -= coeff * k[p_idx] * monomial_val
        else:  # Momentum coordinate p_i  
            # {G_term, p_i} = +k[i] * G_term with q_i exponent reduced by 1
            q_idx = coord_idx - 3
            if k[q_idx] > 0:
                # Evaluate monomial with q_i exponent reduced by 1
                monomial_val = _evaluate_reduced_monomial(k, coords, q_idx, -1)
                result += coeff * k[q_idx] * monomial_val
                
    return result


@njit(fastmath=FASTMATH, cache=True)
def _apply_single_inverse_generator(
    coords: np.ndarray,
    G_n: np.ndarray,
    degree: int,
    psi: np.ndarray,
    clmo: np.ndarray,
    encode_dict_list: List
) -> np.ndarray:
    """
    Apply inverse of a single generating function using first-order approximation.
    
    For the inverse transformation, we use:
    x_new = x_old - {G, x_old} + O(G^2)
    
    This is the first-order approximation of exp(-L_G).
    """
    new_coords = coords.copy()
    
    # Compute Poisson brackets {G, x_i} for each coordinate
    for i in range(6):
        pb_term = _compute_pb_coord(
            G_n, degree, i, coords, clmo, encode_dict_list
        )
        new_coords[i] -= pb_term  # Negative sign for inverse transform
    
    return new_coords


@njit(fastmath=FASTMATH, cache=True)
def _apply_inverse_lie_transforms(
    cm_coords: np.ndarray,
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    tol: float
) -> np.ndarray:
    """Apply inverse Lie transforms to go from center manifold to complex normal coordinates."""
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Start with 6D coordinates: [q1=0, q2, q3, p1=0, p2, p3]
    coords = np.zeros(6, dtype=np.complex128)
    coords[1] = cm_coords[0]  # q2
    coords[2] = cm_coords[2]  # q3
    coords[4] = cm_coords[1]  # p2  
    coords[5] = cm_coords[3]  # p3
    # q1=0, p1=0 remain zero (center manifold constraint)
    
    # Apply inverse generating functions in reverse order
    # The generating functions were applied in order G3, G4, ..., GN
    # So we apply them in reverse: -GN, -G(N-1), ..., -G3
    for degree in range(max_degree, 2, -1):
        if degree < len(poly_G_total) and np.any(poly_G_total[degree]):
            coords = _apply_single_inverse_generator(
                coords, poly_G_total[degree], degree, psi, clmo, encode_dict_list
            )
    
    return coords


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
