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
    """Return H_phys as a list of NumPy arrays (degree-indexed)."""
    # Build symbolic expansion once (cheap, O(max_degree^2))
    T = _build_T_polynomials(max_degree)
    coeffs = [None, None]  # c_0, c_1 unused
    for n in range(2, max_degree+1):
        coeffs.append(create_symbolic_cn(n))  # keep symbolic for later substitution

    U = -se.Add(*[coeffs[n] * T[n] for n in range(2, max_degree+1)])
    K = se.Rational(1, 2) * (px**2 + py**2 + pz**2) + y*px - x*py
    expr = se.expand(K + U)

    return symengine2poly(expr,
                                    [x, y, z, px, py, pz],
                                    max_degree,
                                    psi, clmo,
                                    complex_dtype=False)


def _linear_substitution(expr: se.Basic, subs: dict, var_out: list[se.Symbol],
                        max_degree: int, psi, clmo, point,
                        complex_out: bool = False):
    """Apply `subs` to `expr`, expand, substitute numerical params, convert to coefficient arrays."""
    expanded_symbolic_params = se.expand(expr.subs(subs))

    # Substitute numerical values for transformation parameters (lambda1_sym, c2_sym, etc.)
    # This uses the point's method that knows about lambda1_sym, omega1_sym, s1_sym, s2_sym, c2_sym
    expanded_numerical_modes = point.substitute_parameters(expanded_symbolic_params)
    
    # Substitute numerical values for remaining cn_k (k>2).
    # point._cn(k) provides numeric values. create_symbolic_cn(k) provides symbolic versions.
    # c2 is handled by substitute_parameters as c2_sym.
    cn_substitutions = {create_symbolic_cn(n): point._cn(n) for n in range(3, max_degree + 1)}
    
    expanded_fully_numerical = expanded_numerical_modes.subs(cn_substitutions)

    return symengine2poly(expanded_fully_numerical,
                                    var_out,
                                    max_degree,
                                    psi, clmo,
                                    complex_dtype=complex_out)


def physical_to_real_normal_arrays(point, H_phys_arrays, max_degree, psi, clmo):
    # Recover symbolic expression from arrays (utility below)
    # Assuming H_phys_arrays came from an expression where c_n were already numeric.
    expr_phys = poly2symengine(H_phys_arrays,
                                    get_vars(physical_vars), # Use get_vars for consistency
                                    psi, clmo)
    C, _ = point._symbolic_normal_form_transform()  # 6×6 numeric matrix from real point
    
    # Z_new uses the actual real_normal_vars symbols
    real_vars = get_vars(real_normal_vars)
    Z_new_matrix = se.Matrix(real_vars) # Use Matrix constructor with list of symbols
    
    subs_dict = {var: se.expand(sum(C[i, j]*Z_new_matrix[j] for j in range(6)))
                 for i, var in enumerate(get_vars(physical_vars))}

    return _linear_substitution(expr_phys, subs_dict,
                                real_vars, # Pass the actual list of symbols
                                max_degree, psi, clmo, point=point, complex_out=False)


def real_normal_to_complex_arrays(point, H_rn_arrays, max_degree, psi, clmo):
    """Transform Hamiltonian from real normal to complex canonical coordinates.
    
    This transformation involves complex numbers, so the resulting arrays will have complex dtype.
    The implementation follows exactly the same substitution as in the deprecated version.
    """
    # Convert arrays back to symbolic form for substitution
    expr_rn = poly2symengine(H_rn_arrays, get_vars(real_normal_vars), psi, clmo)
    
    # Prepare the complex transformation
    sqrt2 = se.sqrt(2)
    
    # Target variables - canonical complex normal form
    q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
    
    # Source variables - real normal variables
    x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)

    # Use exact same substitution as the deprecated implementation
    # This maps from real normal to complex canonical coordinates
    subs = {
        x_rn: q1,                         # Center coordinate
        y_rn: (q2 + se.I*p2) / sqrt2,     # Complex planar coordinate
        z_rn: (q3 + se.I*p3) / sqrt2,     # Complex vertical coordinate
        px_rn: p1,                        # Center momentum
        py_rn: (se.I*q2 + p2) / sqrt2,    # Complex planar momentum
        pz_rn: (se.I*q3 + p3) / sqrt2,    # Complex vertical momentum
    }
    
    # Perform the substitution with full expansion
    expanded = se.expand(expr_rn.subs(subs))
    
    # Handle parameter substitution like in the deprecated implementation
    # This uses the point's method that knows about lambda1_sym, omega1_sym, s1_sym, s2_sym, c2_sym
    expanded_with_params = point.substitute_parameters(expanded)
    
    # Substitute numerical values for remaining cn_k (k>2).
    # point._cn(k) provides numeric values. create_symbolic_cn(k) provides symbolic versions.
    # c2 is handled by substitute_parameters as c2_sym.
    cn_substitutions = {create_symbolic_cn(n): point._cn(n) for n in range(3, max_degree + 1)}
    expanded_fully_numerical = expanded_with_params.subs(cn_substitutions)
    
    # Clean numerical artifacts (similar to deprecated implementation)
    # Convert to sympy for cleaning
    import sympy as sp
    expanded_sp = sp.sympify(expanded_fully_numerical)
    from algorithms.center._deprecated.dep_core import _clean_numerical_artifacts
    cleaned_sp = _clean_numerical_artifacts(expanded_sp, tol=1e-16)
    cleaned_expr = se.sympify(cleaned_sp)  # Convert back to symengine
    
    # Important: set complex_dtype=True to ensure arrays have complex dtype
    # and all the imaginary terms are correctly preserved
    return symengine2poly(cleaned_expr,
                         get_vars(canonical_normal_vars),
                         max_degree,
                         psi, clmo,
                         complex_dtype=True)


def complex_to_real_arrays(point, H_cn_arrays, max_degree, psi, clmo):
    # Assuming H_cn_arrays came from an expression where relevant params were numeric
    expr_cn = poly2symengine(H_cn_arrays,
                                     get_vars(canonical_normal_vars), # Use get_vars
                                     psi, clmo)
    sqrt2 = se.sqrt(2)

    # Target variables
    rn_vars_ordered = get_vars(real_normal_vars)
    x_rn_s, y_rn_s, z_rn_s, px_rn_s, py_rn_s, pz_rn_s = rn_vars_ordered

    # Source variables
    q1_s, q2_s, q3_s, p1_s, p2_s, p3_s = get_vars(canonical_normal_vars)
    
    subs = {
        q1_s: x_rn_s,
        q2_s: (y_rn_s - se.I*py_rn_s) / sqrt2,
        q3_s: (z_rn_s - se.I*pz_rn_s) / sqrt2,
        p1_s: px_rn_s,
        p2_s: (py_rn_s - se.I*y_rn_s) / sqrt2, # Note: original had I*q2_s here, fixed from Jorba. JM(12) py_rn - I y_rn for p2
        p3_s: (pz_rn_s - se.I*z_rn_s) / sqrt2, # Note: original had I*q3_s here, fixed from Jorba. JM(12) pz_rn - I z_rn for p3
    }
    return _linear_substitution(expr_cn, subs,
                                rn_vars_ordered, # Pass the actual list of symbols
                                max_degree, psi, clmo, point=point, complex_out=False)


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
