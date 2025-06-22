import numpy as np
from numba import njit
from numba.typed import List
from numba import types
from typing import Tuple

from algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.polynomial.operations import (polynomial_add_inplace,
                                                     polynomial_multiply,
                                                     polynomial_variable,
                                                     polynomial_zero_list)
from utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _build_T_polynomials(poly_x, poly_y, poly_z, max_deg: int, psi_table, clmo_table, encode_dict_list) -> types.ListType:
    """
    Build Chebyshev polynomials of the first kind up to the specified maximum degree.
    
    Parameters
    ----------
    poly_x : List[numpy.ndarray]
        Polynomial representation of x coordinate
    poly_y : List[numpy.ndarray]
        Polynomial representation of y coordinate
    poly_z : List[numpy.ndarray]
        Polynomial representation of z coordinate
    max_deg : int
        Maximum degree of Chebyshev polynomials to generate
    psi_table : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo_table : List[numpy.ndarray]
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    list[List[numpy.ndarray]]
        Numba typed list of Chebyshev polynomials T_0 through T_{max_deg}
        
    Notes
    -----
    This function implements the recurrence relation for Chebyshev polynomials:
    T_0(r) = 1
    T_1(r) = r
    T_n(r) = 2r*T_{n-1}(r) - T_{n-2}(r)
    
    For our 3D case, r is replaced with x/(x² + y² + z²)^(1/2)
    This leads to the modified recurrence relation:
    T_n = (2n-1)/n * x * T_{n-1} - (n-1)/n * (x² + y² + z²) * T_{n-2}
    """
    poly_T_list_of_polys = List()
    for _ in range(max_deg + 1):
        poly_T_list_of_polys.append(polynomial_zero_list(max_deg, psi_table))

    if max_deg >= 0 and len(poly_T_list_of_polys[0]) > 0 and len(poly_T_list_of_polys[0][0]) > 0:
        poly_T_list_of_polys[0][0][0] = 1.0
    if max_deg >= 1:
        poly_T_list_of_polys[1] = poly_x # type: ignore 

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        term1_mult = polynomial_multiply(poly_x, poly_T_list_of_polys[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term1, term1_mult, a)

        poly_x_sq = polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_y_sq = polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_z_sq = polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)

        poly_sum_sq = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_sum_sq, poly_x_sq, 1.0)
        polynomial_add_inplace(poly_sum_sq, poly_y_sq, 1.0)
        polynomial_add_inplace(poly_sum_sq, poly_z_sq, 1.0)

        term2_mult = polynomial_multiply(poly_sum_sq, poly_T_list_of_polys[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term2 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term2, term2_mult, -b)

        poly_Tn = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_Tn, term1, 1.0)
        polynomial_add_inplace(poly_Tn, term2, 1.0)
        poly_T_list_of_polys[n] = poly_Tn
    return poly_T_list_of_polys


@njit(fastmath=FASTMATH, cache=False)
def _build_R_polynomials(poly_x, poly_y, poly_z, poly_T: types.ListType, max_deg: int, psi_table, clmo_table, encode_dict_list) -> types.ListType:
    """
    Build R_n polynomials up to the specified maximum degree.
    
    Parameters
    ----------
    poly_x : List[numpy.ndarray]
        Polynomial representation of x coordinate
    poly_y : List[numpy.ndarray]
        Polynomial representation of y coordinate
    poly_z : List[numpy.ndarray]
        Polynomial representation of z coordinate
    poly_T : types.ListType
        List of T_n Chebyshev polynomials (output from _build_T_polynomials)
    max_deg : int
        Maximum degree of R_n polynomials to generate
    psi_table : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo_table : List[numpy.ndarray]
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    list[List[numpy.ndarray]]
        Numba typed list of R_n polynomials R_0 through R_{max_deg}
        
    Notes
    -----
    This function implements the recurrence relation:
    R_0 = -1
    R_1 = -3x
    R_n = ((2n+3)/(n+2)) * x * R_{n-1} - ((2n+2)/(n+2)) * T_n - ((n+1)/(n+2)) * (x^2 + y^2 + z^2) * R_{n-2}
    """
    poly_R_list_of_polys = List()
    for _ in range(max_deg + 1):
        poly_R_list_of_polys.append(polynomial_zero_list(max_deg, psi_table))

    if max_deg >= 0:
        # R_0 = -1
        if len(poly_R_list_of_polys[0]) > 0 and len(poly_R_list_of_polys[0][0]) > 0:
            poly_R_list_of_polys[0][0][0] = -1.0
    
    if max_deg >= 1:
        # R_1 = -3x
        r1_poly = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(r1_poly, poly_x, -3.0)
        poly_R_list_of_polys[1] = r1_poly

    # Pre-calculate x^2, y^2, z^2, and x^2 + y^2 + z^2 as they are used in the loop
    poly_x_sq = None # Represents x^2
    poly_y_sq = None # Represents y^2
    poly_z_sq = None # Represents z^2
    poly_rho_sq = None # Represents x^2 + y^2 + z^2

    if max_deg >=2: # Only needed if the loop runs
        poly_x_sq = polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_y_sq = polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_z_sq = polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
        
        poly_rho_sq = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_rho_sq, poly_x_sq, 1.0)
        polynomial_add_inplace(poly_rho_sq, poly_y_sq, 1.0)
        polynomial_add_inplace(poly_rho_sq, poly_z_sq, 1.0)

    for n in range(2, max_deg + 1):
        n_ = float(n)
        
        coeff1 = (2.0 * n_ + 3.0) / (n_ + 2.0)
        coeff2 = (2.0 * n_ + 2.0) / (n_ + 2.0)
        coeff3 = (n_ + 1.0) / (n_ + 2.0)

        # Term 1: coeff1 * x * R_{n-1}
        term1_mult_x_Rnm1 = polynomial_multiply(poly_x, poly_R_list_of_polys[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1_poly = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term1_poly, term1_mult_x_Rnm1, coeff1)

        # Term 2: -coeff2 * T_n
        term2_poly = polynomial_zero_list(max_deg, psi_table)
        # poly_T[n] is T_n
        polynomial_add_inplace(term2_poly, poly_T[n], -coeff2)
        
        # Term 3: -coeff3 * (x^2 + y^2 + z^2) * R_{n-2}
        # poly_rho_sq is already computed if needed
        term3_mult_rhosq_Rnm2 = polynomial_multiply(poly_rho_sq, poly_R_list_of_polys[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term3_poly = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term3_poly, term3_mult_rhosq_Rnm2, -coeff3)
        
        # Combine terms for R_n
        poly_Rn = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_Rn, term1_poly, 1.0)
        polynomial_add_inplace(poly_Rn, term2_poly, 1.0)
        polynomial_add_inplace(poly_Rn, term3_poly, 1.0)
        poly_R_list_of_polys[n] = poly_Rn
        
    return poly_R_list_of_polys


def _build_potential_U(poly_T, point, max_deg: int, psi_table) -> List[np.ndarray]:
    """
    Build the potential energy part of the Hamiltonian.
    
    Parameters
    ----------
    poly_T : list[List[numpy.ndarray]]
        List of Chebyshev polynomials T_0 through T_{max_deg}
    point : object
        Object representing a collinear point, with a _cn method that returns
        the nth coefficient in the potential expansion
    max_deg : int
        Maximum degree for the polynomial representation
    psi_table : numpy.ndarray
        Combinatorial table from init_index_tables
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the potential energy
        
    Notes
    -----
    The potential is expanded as a series of Chebyshev polynomials:
    U = -∑_{n≥2} c_n T_n(x/(x² + y² + z²)^(1/2))
    where c_n are coefficients obtained from the point object.
    """
    poly_U = polynomial_zero_list(max_deg, psi_table)
    for n in range(2, max_deg + 1):
        polynomial_add_inplace(poly_U, poly_T[n], -point._cn(n))
    return poly_U


def _build_kinetic_energy_terms(poly_px, poly_py, poly_pz, max_deg: int, psi_table, clmo_table, encode_dict_list) -> List[np.ndarray]:
    """
    Build the kinetic energy part of the Hamiltonian.
    
    Parameters
    ----------
    poly_px : List[numpy.ndarray]
        Polynomial representation of px (momentum in x direction)
    poly_py : List[numpy.ndarray]
        Polynomial representation of py (momentum in y direction)
    poly_pz : List[numpy.ndarray]
        Polynomial representation of pz (momentum in z direction)
    max_deg : int
        Maximum degree for the polynomial representation
    psi_table : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo_table : List[numpy.ndarray]
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the kinetic energy
        
    Notes
    -----
    The kinetic energy in the rotating frame is given by:
    T = 1/2(px² + py² + pz²)
    """
    poly_kinetic = polynomial_zero_list(max_deg, psi_table)
    for poly_momentum in (poly_px, poly_py, poly_pz):
        term = polynomial_multiply(poly_momentum, poly_momentum, max_deg, psi_table, clmo_table, encode_dict_list)
        polynomial_add_inplace(poly_kinetic, term, 0.5)
    return poly_kinetic


def _build_rotational_terms(poly_x, poly_y, poly_px, poly_py, max_deg: int, psi_table, clmo_table, encode_dict_list) -> List[np.ndarray]:
    """
    Build the Coriolis force terms of the Hamiltonian in the rotating frame.
    
    Parameters
    ----------
    poly_x : List[numpy.ndarray]
        Polynomial representation of x coordinate
    poly_y : List[numpy.ndarray]
        Polynomial representation of y coordinate
    poly_px : List[numpy.ndarray]
        Polynomial representation of px (momentum in x direction)
    poly_py : List[numpy.ndarray]
        Polynomial representation of py (momentum in y direction)
    max_deg : int
        Maximum degree for the polynomial representation
    psi_table : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo_table : List[numpy.ndarray]
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the rotational terms
        
    Notes
    -----
    The rotational terms in the rotating frame Hamiltonian come from
    the Coriolis force and are given by:
    y·px - x·py
    """
    poly_rot = polynomial_zero_list(max_deg, psi_table)
    
    term_ypx = polynomial_multiply(poly_y, poly_px, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(poly_rot, term_ypx, 1.0)

    term_xpy = polynomial_multiply(poly_x, poly_py, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(poly_rot, term_xpy, -1.0)
    
    return poly_rot


def build_physical_hamiltonian(point, max_deg: int) -> List[np.ndarray]:
    """
    Build the complete physical Hamiltonian in the rotating frame.
    
    Parameters
    ----------
    point : object
        Object representing a collinear point, with a _cn method that returns
        the nth coefficient in the potential expansion
    max_deg : int
        Maximum degree for the polynomial representation
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the complete Hamiltonian
        
    Notes
    -----
    The complete Hamiltonian in the rotating frame consists of:
    H = T + V + C
    
    where:
    T = 1/2(px² + py² + pz²) is the kinetic energy
    V = -∑_{n≥2} c_n T_n(x/(x² + y² + z²)^(1/2)) is the potential energy
    C = y·px - x·py are the Coriolis terms
    
    This function initializes all necessary polynomial data structures,
    builds each component separately, and combines them into the final Hamiltonian.
    """
    psi_table, clmo_table = init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    poly_H = polynomial_zero_list(max_deg, psi_table)

    poly_x, poly_y, poly_z, poly_px, poly_py, poly_pz = [
        polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list) for i in range(6)
    ]

    poly_kinetic = _build_kinetic_energy_terms(poly_px, poly_py, poly_pz, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(poly_H, poly_kinetic, 1.0)

    poly_rot = _build_rotational_terms(poly_x, poly_y, poly_px, poly_py, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(poly_H, poly_rot, 1.0)

    poly_T = _build_T_polynomials(poly_x, poly_y, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
    
    poly_U = _build_potential_U(poly_T, point, max_deg, psi_table)

    polynomial_add_inplace(poly_H, poly_U, 1.0)

    return poly_H


def build_lindstedt_poincare_rhs_polynomials(point, max_deg: int) -> Tuple[List, List, List]:
    """
    Build the polynomial representations of the right-hand sides (RHS)
    of the Lindstedt-Poincaré equations of motion (as in equation (14) of the reference image).

    The equations are:
    RHS_x = sum_{n>=2} c_{n+1} * (n+1) * T_n
    RHS_y = y * sum_{n>=2} c_{n+1} * R_{n-1}
    RHS_z = z * sum_{n>=2} c_{n+1} * R_{n-1}

    Parameters
    ----------
    point : object
        Object representing a collinear point, with a `_cn` method that returns
        the k-th coefficient c_k in the potential expansion.
    max_deg : int
        Maximum degree for the polynomial representation of the RHS terms.
        
    Returns
    -------
    Tuple[List, List, List]
        A tuple containing three polynomial lists (Numba typed lists of NumPy arrays):
        (rhs_x_poly, rhs_y_poly, rhs_z_poly)
    """
    psi_table, clmo_table = init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    poly_x, poly_y, poly_z = [
        polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list) for i in range(3)
    ]

    poly_T_list = _build_T_polynomials(poly_x, poly_y, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
    poly_R_list = _build_R_polynomials(poly_x, poly_y, poly_z, poly_T_list, max_deg, psi_table, clmo_table, encode_dict_list)

    rhs_x_poly = polynomial_zero_list(max_deg, psi_table)

    sum_term_for_y_z_eqs = polynomial_zero_list(max_deg, psi_table)

    for n in range(2, max_deg + 1):
        cn_plus_1 = point._cn(n + 1)
        coeff = cn_plus_1 * float(n + 1)
        polynomial_add_inplace(rhs_x_poly, poly_T_list[n], coeff)

    for n in range(2, max_deg + 1):
        cn_plus_1 = point._cn(n + 1)
        if (n - 1) < len(poly_R_list):
            polynomial_add_inplace(sum_term_for_y_z_eqs, poly_R_list[n - 1], cn_plus_1)

    rhs_y_poly = polynomial_multiply(poly_y, sum_term_for_y_z_eqs, max_deg, psi_table, clmo_table, encode_dict_list)

    rhs_z_poly = polynomial_multiply(poly_z, sum_term_for_y_z_eqs, max_deg, psi_table, clmo_table, encode_dict_list)
    
    return rhs_x_poly, rhs_y_poly, rhs_z_poly

