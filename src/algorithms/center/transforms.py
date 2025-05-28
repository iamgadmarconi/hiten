import math

import numpy as np
from numba import njit
from numba.typed import List

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
