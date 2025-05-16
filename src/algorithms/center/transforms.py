import math
import numpy as np
from numba.typed import List

from algorithms.center.polynomial.base import decode_multiindex
from algorithms.center.polynomial.operations import (
    polynomial_multiply,
    polynomial_zero_list,
    polynomial_add_inplace,
    polynomial_power,
    polynomial_variable,
    polynomial_clean
)


def _linear_variable_polys(C: np.ndarray, max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Return list L[old_idx] = polynomial for the old variable expressed
    in the NEW variables (which are the basis vectors under C).

    C  is shape (6,6):   old_var_i  =  Σ_j  C[i,j] * new_var_j
    """
    new_basis = [polynomial_variable(j, max_deg, psi, clmo) for j in range(6)]
    L: List[np.ndarray] = []
    for i in range(6):
        pol = polynomial_zero_list(max_deg, psi)
        for j in range(6):
            if C[i, j] == 0:
                continue
            polynomial_add_inplace(pol, new_basis[j], C[i, j], max_deg)
        L.append(pol)
    return L

def substitute_linear(H_old: List[np.ndarray], C: np.ndarray, max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Substitute each old variable via the linear map defined by C.
    Returns polynomial in the NEW variable set.
    """
    var_polys = _linear_variable_polys(C, max_deg, psi, clmo)
    H_new = polynomial_zero_list(max_deg, psi)

    for deg in range(max_deg + 1):
        coeff_vec = H_old[deg]
        if not coeff_vec.any():
            continue
        for pos, coeff in enumerate(coeff_vec):
            if coeff == 0:
                continue
            k = decode_multiindex(pos, deg, clmo)
            
            # build product  Π_i  (var_polys[i] ** k_i)
            term = polynomial_zero_list(max_deg, psi)
            
            # Fix: Preserve the full complex value instead of just the real part
            term[0][0] = coeff
                
            for i_var in range(6):
                if k[i_var] == 0:
                    continue
                pwr = polynomial_power(var_polys[i_var], k[i_var], max_deg, psi, clmo)
                term = polynomial_multiply(term, pwr, max_deg, psi, clmo)
                
            polynomial_add_inplace(H_new, term, 1.0, max_deg)

    return polynomial_clean(H_new, 1e-14)

def phys2rn(point, H_phys: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Apply the numeric normal-form matrix C returned by point.
    """
    C, _ = point.normal_form_transform()
    return substitute_linear(H_phys, C, max_deg, psi, clmo)

def rn2cn(H_rn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
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

    return substitute_linear(H_rn, C, max_deg, psi, clmo)

def cn2rn(H_cn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
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

    return substitute_linear(H_cn, Cinv, max_deg, psi, clmo)
