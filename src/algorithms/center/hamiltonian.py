import symengine as se
import numpy as np
from numba import njit

from algorithms.center.polynomial.base import (
    init_index_tables, decode_multiindex
)
from algorithms.center.polynomial.conversions import (
    poly2symengine, symengine2poly
)
from algorithms.center.lie import lie_transform
from algorithms.variables import (
    physical_vars, real_normal_vars, canonical_normal_vars,
    get_vars, create_symbolic_cn,
    linear_modes_vars, scale_factors_vars, N_VARS
)

x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)
s1, s2 = get_vars(scale_factors_vars)


def _build_T_polynomials(N: int) -> list[se.Basic]:
    """Return [T0 … TN] via the Legendre-type recurrence (JM99, eq. 6)."""
    if N == 0:
        return [se.Integer(1)]
    if N == 1:
        return [se.Integer(1), x]

    T = [se.Integer(1), x]
    for n in range(2, N + 1):
        n_ = se.Integer(n)
        Tn = ((2*n_ - 1)/n_) * x * T[n-1] - ((n_-1)/n_) * (x**2 + y**2 + z**2) * T[n-2]
        T.append(se.expand(Tn))
    return T


def hamiltonian_arrays(point, max_degree, psi, clmo):
    """Return physical Hamiltonian as degree‑indexed NumPy arrays (real)."""
    # symbolic expansion first
    T = _build_T_polynomials(max_degree)
    symbolic_coeffs = [None, None]  # c0, c1 unused
    for n in range(2, max_degree+1):
        symbolic_coeffs.append(create_symbolic_cn(n))

    U = -se.Add(*[symbolic_coeffs[n] * T[n] for n in range(2, max_degree+1)])
    K = se.Rational(1, 2) * (px**2 + py**2 + pz**2) + y*px - x*py
    expr = se.expand(K + U)

    subs_c = {create_symbolic_cn(n): point._cn(n) for n in range(2, max_degree+1)}
    expr_num = expr.subs(subs_c)

    return symengine2poly(expr_num,
                                    [x, y, z, px, py, pz],
                                    max_degree,
                                    psi, clmo,
                                    complex_dtype=False)


def _linear_substitution(expr: se.Basic, subs: dict, var_out: list[se.Symbol],
                        max_degree: int, psi, clmo, complex_out: bool = False):
    expr_sub = se.expand(expr.subs(subs))
    return symengine2poly(expr_sub,
                                    var_out,
                                    max_degree,
                                    psi, clmo,
                                    complex_dtype=complex_out)

def physical_to_real_normal_arrays(point, H_phys_arrays, max_degree, psi, clmo):
    expr_phys = poly2symengine(H_phys_arrays,
                            [x, y, z, px, py, pz],
                            psi, clmo)
    C, _ = point._symbolic_normal_form_transform()  # numeric matrix 6×6
    Z_new = se.Matrix([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn])
    subs = {var: se.expand(sum(C[i, j]*Z_new[j] for j in range(6)))
            for i, var in enumerate([x, y, z, px, py, pz])}
    return _linear_substitution(expr_phys, subs,
                                [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn],
                                max_degree, psi, clmo, complex_out=False)


def real_normal_to_complex_arrays(point, H_rn_arrays, max_degree, psi, clmo):
    expr_rn = poly2symengine(H_rn_arrays,
                            [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn],
                            psi, clmo)
    sqrt2 = se.sqrt(2)
    subs = {
        x_rn: q1,
        y_rn: (q2 + se.I*p2) / sqrt2,
        z_rn: (q3 + se.I*p3) / sqrt2,
        px_rn: p1,
        py_rn: (se.I*q2 + p2) / sqrt2,
        pz_rn: (se.I*q3 + p3) / sqrt2,
    }
    return _linear_substitution(expr_rn, subs,
                                [q1, q2, q3, p1, p2, p3],
                                max_degree, psi, clmo, complex_out=True)


def complex_to_real_arrays(point, H_cn_arrays, max_degree, psi, clmo):
    expr_cn = poly2symengine(H_cn_arrays,
                            [q1, q2, q3, p1, p2, p3],
                            psi, clmo)
    sqrt2 = se.sqrt(2)
    subs = {
        q1: x_rn,
        q2: (y_rn - se.I*py_rn) / sqrt2,
        q3: (z_rn - se.I*pz_rn) / sqrt2,
        p1: px_rn,
        p2: (py_rn - se.I*y_rn) / sqrt2,
        p3: (pz_rn - se.I*z_rn) / sqrt2,
    }
    return _linear_substitution(expr_cn, subs,
                                [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn],
                                max_degree, psi, clmo, complex_out=True)


@njit(fastmath=True, cache=True)
def _zero_q1_p1(H_list):
    """Set coefficients with q1 or p1 exponents >0 to zero (in-place)."""
    for d in range(len(H_list)):
        arr = H_list[d]
        if arr.size == 0:
            continue
        for idx in range(arr.size):
            if arr[idx] == 0:
                continue
            k = decode_multiindex(idx, d, clmo_cached)
            if k[0] > 0 or k[3] > 0:
                arr[idx] = 0

# `clmo_cached` will be set once via initialise() below so that the njit'ed
# function sees it as a global read‑only list.  (Numba limitation.)
clmo_cached = None  # type: ignore


def initialise_tables(max_degree):
    global psi_cached, clmo_cached
    psi_cached, clmo_cached = init_index_tables(max_degree)
    return psi_cached, clmo_cached


def compute_center_manifold_arrays(point, max_degree):
    psi, clmo = initialise_tables(max_degree)
    H_phys = hamiltonian_arrays(point, max_degree, psi, clmo)
    H_rn   = physical_to_real_normal_arrays(point, H_phys, max_degree, psi, clmo)
    H_cn   = real_normal_to_complex_arrays(point, H_rn, max_degree, psi, clmo)
    H_cnt, G_tot = lie_transform(point, H_cn, psi, clmo, max_degree)
    return H_cnt, G_tot


def reduce_center_manifold_arrays(point, max_degree):
    H_cnt, _ = compute_center_manifold_arrays(point, max_degree)
    _zero_q1_p1(H_cnt)
    return H_cnt


def real_normal_center_manifold_arrays(point, max_degree):
    psi, clmo = initialise_tables(max_degree)
    H_cnr = reduce_center_manifold_arrays(point, max_degree)
    return complex_to_real_arrays(point, H_cnr, max_degree, psi, clmo)
