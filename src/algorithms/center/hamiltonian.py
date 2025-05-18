import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.polynomial.operations import (polynomial_add_inplace,
                                                     polynomial_multiply,
                                                     polynomial_variable,
                                                     polynomial_zero_list)


@njit(fastmath=True, cache=False)
def _build_T_polynomials(poly_x, poly_y, poly_z, max_deg: int, psi_table, clmo_table, encode_dict_list) -> list:
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
        List of Chebyshev polynomials T_0 through T_{max_deg}
        
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
    poly_T = [polynomial_zero_list(max_deg, psi_table) for _ in range(max_deg + 1)]
    if max_deg >= 0 and len(poly_T[0]) > 0 and len(poly_T[0][0]) > 0:
        poly_T[0][0][0] = 1.0
    if max_deg >= 1:
        poly_T[1] = poly_x # type: ignore

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        term1_mult = polynomial_multiply(poly_x, poly_T[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term1, term1_mult, a)

        poly_x_sq = polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_y_sq = polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_z_sq = polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)

        poly_sum_sq = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_sum_sq, poly_x_sq, 1.0)
        polynomial_add_inplace(poly_sum_sq, poly_y_sq, 1.0)
        polynomial_add_inplace(poly_sum_sq, poly_z_sq, 1.0)

        term2_mult = polynomial_multiply(poly_sum_sq, poly_T[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term2 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term2, term2_mult, -b)

        poly_Tn = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(poly_Tn, term1, 1.0)
        polynomial_add_inplace(poly_Tn, term2, 1.0)
        poly_T[n] = poly_Tn
    return poly_T


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

