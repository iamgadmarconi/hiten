import symengine as se
import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import init_index_tables
from algorithms.center.polynomial.operations import (
    polynomial_multiply,
    polynomial_zero_list,
    polynomial_add_inplace,
    polynomial_power,
    polynomial_variable
)

def build_physical_hamiltonian(point,
                               max_deg: int,
                               psi_config,
                               clmo_tables_deg: int) -> List[np.ndarray]:
    """
    Return the physical-frame Hamiltonian as Poly in vars
    [x, y, z, px, py, pz]  (indices 0…5).
    """
    complex_dt = psi_config[1]
    
    psi_table, clmo_table = init_index_tables(clmo_tables_deg)

    H = polynomial_zero_list(max_deg, psi_table, complex_dt)

    x, y, z, px, py, pz = [
        polynomial_variable(i, max_deg, psi_table, complex_dtype=complex_dt) for i in range(6)
    ]

    for mom in (px, py, pz):
        term = polynomial_multiply(mom, mom, max_deg, psi_table, clmo_table)
        polynomial_add_inplace(H, term, 0.5)

    term_ypx = polynomial_multiply(y, px, max_deg, psi_table, clmo_table)
    polynomial_add_inplace(H, term_ypx, 1.0)

    term_xpy = polynomial_multiply(x, py, max_deg, psi_table, clmo_table)
    polynomial_add_inplace(H, term_xpy, -1.0)

    T = [polynomial_zero_list(max_deg, psi_table, complex_dt) for _ in range(max_deg + 1)]
    if max_deg >= 0 and len(T[0]) > 0 and len(T[0][0]) > 0:
        T[0][0][0] = 1.0
    if max_deg >= 1:
        T[1] = x

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2*n_-1)/n_
        b = (n_-1)/n_
        
        term1_mult = polynomial_multiply(x, T[n-1], max_deg, psi_table, clmo_table)
        term1 = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(term1, term1_mult, a)

        x_sq = polynomial_multiply(x, x, max_deg, psi_table, clmo_table)
        y_sq = polynomial_multiply(y, y, max_deg, psi_table, clmo_table)
        z_sq = polynomial_multiply(z, z, max_deg, psi_table, clmo_table)
        
        sum_sq = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(sum_sq, x_sq, 1.0)
        polynomial_add_inplace(sum_sq, y_sq, 1.0)
        polynomial_add_inplace(sum_sq, z_sq, 1.0)
        
        term2_mult = polynomial_multiply(sum_sq, T[n-2], max_deg, psi_table, clmo_table)
        term2 = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(term2, term2_mult, -b)

        Tn = polynomial_zero_list(max_deg, psi_table, complex_dt)
        polynomial_add_inplace(Tn, term1, 1.0)
        polynomial_add_inplace(Tn, term2, 1.0)
        T[n] = Tn

    U_potential = polynomial_zero_list(max_deg, psi_table, complex_dt)
    for n in range(2, max_deg + 1):
        polynomial_add_inplace(U_potential, T[n], -point._cn(n))

    polynomial_add_inplace(H, U_potential, 1.0)

    return H