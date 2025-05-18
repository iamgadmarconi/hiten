import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.polynomial.operations import (polynomial_add_inplace,
                                                     polynomial_multiply,
                                                     polynomial_variable,
                                                     polynomial_zero_list)


@njit(fastmath=True, cache=True)
def _build_T_polynomials(x, y, z, max_deg: int, psi_table, clmo_table, encode_dict_list) -> list:
    """Helper function to build the T polynomials."""
    T = [polynomial_zero_list(max_deg, psi_table) for _ in range(max_deg + 1)]
    if max_deg >= 0 and len(T[0]) > 0 and len(T[0][0]) > 0:
        T[0][0][0] = 1.0
    if max_deg >= 1:
        T[1] = x # type: ignore

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        term1_mult = polynomial_multiply(x, T[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term1, term1_mult, a)

        x_sq = polynomial_multiply(x, x, max_deg, psi_table, clmo_table, encode_dict_list)
        y_sq = polynomial_multiply(y, y, max_deg, psi_table, clmo_table, encode_dict_list)
        z_sq = polynomial_multiply(z, z, max_deg, psi_table, clmo_table, encode_dict_list)

        sum_sq = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(sum_sq, x_sq, 1.0)
        polynomial_add_inplace(sum_sq, y_sq, 1.0)
        polynomial_add_inplace(sum_sq, z_sq, 1.0)

        term2_mult = polynomial_multiply(sum_sq, T[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term2 = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(term2, term2_mult, -b)

        Tn = polynomial_zero_list(max_deg, psi_table)
        polynomial_add_inplace(Tn, term1, 1.0)
        polynomial_add_inplace(Tn, term2, 1.0)
        T[n] = Tn
    return T


def _build_potential_U(T_polynomials, point, max_deg: int, psi_table) -> np.ndarray:
    """Helper function to build the U_potential polynomial."""
    U_potential = polynomial_zero_list(max_deg, psi_table)
    for n in range(2, max_deg + 1):
        polynomial_add_inplace(U_potential, T_polynomials[n], -point._cn(n))
    return U_potential


def _build_kinetic_energy_terms(px, py, pz, max_deg: int, psi_table, clmo_table, encode_dict_list) -> np.ndarray:
    """Helper function to build the kinetic energy terms of the Hamiltonian."""
    kinetic_energy = polynomial_zero_list(max_deg, psi_table)
    for mom in (px, py, pz):
        term = polynomial_multiply(mom, mom, max_deg, psi_table, clmo_table, encode_dict_list)
        polynomial_add_inplace(kinetic_energy, term, 0.5)
    return kinetic_energy


def _build_rotational_terms(x, y, px, py, max_deg: int, psi_table, clmo_table, encode_dict_list) -> np.ndarray:
    """Helper function to build the rotational terms (y*px - x*py) of the Hamiltonian."""
    rotational_terms = polynomial_zero_list(max_deg, psi_table)
    
    term_ypx = polynomial_multiply(y, px, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(rotational_terms, term_ypx, 1.0)

    term_xpy = polynomial_multiply(x, py, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(rotational_terms, term_xpy, -1.0)
    
    return rotational_terms


def build_physical_hamiltonian(point, max_deg: int) -> List[np.ndarray]:
    """
    Return the physical-frame Hamiltonian as Poly in vars
    [x, y, z, px, py, pz]  (indices 0…5).
    """
    
    psi_table, clmo_table = init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    H = polynomial_zero_list(max_deg, psi_table)

    x, y, z, px, py, pz = [
        polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list) for i in range(6)
    ]

    kinetic_energy = _build_kinetic_energy_terms(px, py, pz, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(H, kinetic_energy, 1.0)

    rotational_terms = _build_rotational_terms(x, y, px, py, max_deg, psi_table, clmo_table, encode_dict_list)
    polynomial_add_inplace(H, rotational_terms, 1.0)

    T = _build_T_polynomials(x, y, z, max_deg, psi_table, clmo_table, encode_dict_list)
    
    U_potential = _build_potential_U(T, point, max_deg, psi_table)

    polynomial_add_inplace(H, U_potential, 1.0)

    return H

