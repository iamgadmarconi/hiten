import math
import symengine as se
import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import init_index_tables, decode_multiindex
from algorithms.center.polynomial.operations import (
    polynomial_multiply,
    polynomial_zero_list,
    polynomial_add_inplace,
    polynomial_power,
    polynomial_variable
)


def _build_T_polynomials(x, y, z, max_deg: int, psi_table, clmo_table, complex_dt) -> list:
    """Helper function to build the T polynomials."""
    T = [polynomial_zero_list(max_deg, psi_table, complex_dt) for _ in range(max_deg + 1)]
    if max_deg >= 0 and len(T[0]) > 0 and len(T[0][0]) > 0:
        T[0][0][0] = 1.0
    if max_deg >= 1:
        T[1] = x # type: ignore

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        term1_mult = polynomial_multiply(x, T[n - 1], max_deg, psi_table, clmo_table)
        term1 = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(term1, term1_mult, a)

        x_sq = polynomial_multiply(x, x, max_deg, psi_table, clmo_table)
        y_sq = polynomial_multiply(y, y, max_deg, psi_table, clmo_table)
        z_sq = polynomial_multiply(z, z, max_deg, psi_table, clmo_table)

        sum_sq = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(sum_sq, x_sq, 1.0)
        polynomial_add_inplace(sum_sq, y_sq, 1.0)
        polynomial_add_inplace(sum_sq, z_sq, 1.0)

        term2_mult = polynomial_multiply(sum_sq, T[n - 2], max_deg, psi_table, clmo_table)
        term2 = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(term2, term2_mult, -b)

        Tn = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(Tn, term1, 1.0)
        polynomial_add_inplace(Tn, term2, 1.0)
        T[n] = Tn
    return T


def _build_potential_U(T_polynomials, point, max_deg: int, psi_table, complex_dt) -> np.ndarray:
    """Helper function to build the U_potential polynomial."""
    U_potential = polynomial_zero_list(max_deg, psi_table, complex_dt)
    for n in range(2, max_deg + 1):
        polynomial_add_inplace(U_potential, T_polynomials[n], -point._cn(n))
    return U_potential


def _build_kinetic_energy_terms(px, py, pz, max_deg: int, psi_table, clmo_table, complex_dt) -> np.ndarray:
    """Helper function to build the kinetic energy terms of the Hamiltonian."""
    kinetic_energy = polynomial_zero_list(max_deg, psi_table, complex_dt)
    for mom in (px, py, pz):
        term = polynomial_multiply(mom, mom, max_deg, psi_table, clmo_table)
        polynomial_add_inplace(kinetic_energy, term, 0.5)
    return kinetic_energy


def _build_rotational_terms(x, y, px, py, max_deg: int, psi_table, clmo_table, complex_dt) -> np.ndarray:
    """Helper function to build the rotational terms (y*px - x*py) of the Hamiltonian."""
    rotational_terms = polynomial_zero_list(max_deg, psi_table, complex_dt)
    
    term_ypx = polynomial_multiply(y, px, max_deg, psi_table, clmo_table)
    polynomial_add_inplace(rotational_terms, term_ypx, 1.0)

    term_xpy = polynomial_multiply(x, py, max_deg, psi_table, clmo_table)
    polynomial_add_inplace(rotational_terms, term_xpy, -1.0)
    
    return rotational_terms


def build_physical_hamiltonian(point,
                            max_deg: int,
                            psi_config,
                            clmo_tables_deg: int) -> List[np.ndarray]:
    """
    Return the physical-frame Hamiltonian as Poly in vars
    [x, y, z, px, py, pz]  (indices 0…5).
    """
    complex_dt = psi_config[1]
    
    psi_table, clmo_table = init_index_tables(max_deg)

    H = polynomial_zero_list(max_deg, psi_table, complex_dt)

    x, y, z, px, py, pz = [
        polynomial_variable(i, max_deg, psi_table, complex_dtype=complex_dt) for i in range(6)
    ]

    kinetic_energy = _build_kinetic_energy_terms(px, py, pz, max_deg, psi_table, clmo_table, complex_dt)
    polynomial_add_inplace(H, kinetic_energy, 1.0)

    rotational_terms = _build_rotational_terms(x, y, px, py, max_deg, psi_table, clmo_table, complex_dt)
    polynomial_add_inplace(H, rotational_terms, 1.0)

    T = _build_T_polynomials(x, y, z, max_deg, psi_table, clmo_table, complex_dt)
    
    U_potential = _build_potential_U(T, point, max_deg, psi_table, complex_dt)

    polynomial_add_inplace(H, U_potential, 1.0)

    return H

def _linear_variable_polys(C: np.ndarray,
                        max_deg: int,
                        psi,
                        clmo,
                        complex_dtype=False) -> List[np.ndarray]:
    """
    Return list L[old_idx] = polynomial for the old variable expressed
    in the NEW variables (which are the basis vectors under C).

    C  is shape (6,6):   old_var_i  =  Σ_j  C[i,j] * new_var_j
    """
    new_basis = [polynomial_variable(j, max_deg, psi, complex_dtype) for j in range(6)]
    L: List[np.ndarray] = []
    for i in range(6):
        pol = polynomial_zero_list(max_deg, psi, complex_dtype)
        for j in range(6):
            if C[i, j] == 0:
                continue
            polynomial_add_inplace(pol, new_basis[j], C[i, j], max_deg)
        L.append(pol)
    return L

def substitute_linear(H_old: List[np.ndarray],
                      C: np.ndarray,
                      max_deg: int,
                      psi,
                      clmo,
                      complex_dtype=False) -> List[np.ndarray]:
    """
    Substitute each old variable via the linear map defined by C.
    Returns polynomial in the NEW variable set.
    """
    var_polys = _linear_variable_polys(C, max_deg, psi, clmo, complex_dtype)
    H_new = polynomial_zero_list(max_deg, psi, complex_dtype)

    for deg in range(max_deg + 1):
        coeff_vec = H_old[deg]
        if not coeff_vec.any():
            continue
        for pos, coeff in enumerate(coeff_vec):
            if coeff == 0:
                continue
            k = decode_multiindex(pos, deg, clmo)
            
            # build product  Π_i  (var_polys[i] ** k_i)
            term = polynomial_zero_list(max_deg, psi, complex_dtype)
            term[0][0] = coeff
            for i_var in range(6):
                if k[i_var] == 0:
                    continue
                pwr = polynomial_power(var_polys[i_var], k[i_var], max_deg, psi, clmo)
                term = polynomial_multiply(term, pwr, max_deg, psi, clmo)
                
            polynomial_add_inplace(H_new, term, 1.0, max_deg)

    return H_new

def phys2rn(point,
            H_phys: List[np.ndarray],
            max_deg: int,
            psi,
            clmo) -> List[np.ndarray]:
    """
    Apply the numeric normal-form matrix  C  returned by point.
    """
    C, _ = point.normal_form_transform()   # shape (6,6)  float64
    return substitute_linear(H_phys, C, max_deg, psi, clmo, complex_dtype=False)

def rn2cn(H_rn: List[np.ndarray],
        max_deg: int,
        psi,
        clmo) -> List[np.ndarray]:
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

    return substitute_linear(H_rn, C, max_deg, psi, clmo, complex_dtype=True)

def cn2rn(H_cn: List[np.ndarray],
        max_deg: int,
        psi,
        clmo) -> List[np.ndarray]:
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
    Cinv[4, 1] = -1j/sqrt2                         # py
    Cinv[4, 4] = 1/sqrt2
    Cinv[5, 2] = -1j/sqrt2                         # pz
    Cinv[5, 5] = 1/sqrt2

    return substitute_linear(H_cn, Cinv, max_deg, psi, clmo, complex_dtype=True)
