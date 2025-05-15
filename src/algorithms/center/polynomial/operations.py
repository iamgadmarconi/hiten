import numpy as np
from numba import njit
from numba.typed import List

from algorithms.variables import N_VARS
from algorithms.center.polynomial.algebra import _poly_mul, _poly_diff, _poly_scale, poisson, _poly_clean
from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex, PSI_GLOBAL, CLMO_GLOBAL


@njit(fastmath=True, cache=True)
def polynomial_zero_list(max_deg: int, psi) -> List[np.ndarray]:
    """Create a list of zero polynomials up to max_deg. Dispatches to JIT-compiled helpers."""
    lst = List()
    for d in range(max_deg + 1):
        lst.append(make_poly(d, psi))
    return lst

@njit(fastmath=True, cache=True)
def polynomial_variable(idx: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Create a polynomial representing a single variable x_idx (degree-1 monomial).
    idx: 0-5 corresponding to (x,y,z,px,py,pz) in the current frame.
    """
    pol = polynomial_zero_list(max_deg, psi)
    k = np.zeros(N_VARS, dtype=np.int64)
    k[idx] = 1
    if 1 < len(pol) and pol[1].size > 0:
        encoded_idx = encode_multiindex(k, 1, psi, clmo)
        if 0 <= encoded_idx < pol[1].shape[0]:
            pol[1][encoded_idx] = 1.0
    return pol

@njit(fastmath=True, cache=True)
def polynomial_variables_list(max_deg: int, psi, clmo) -> List[List[np.ndarray]]:
    """Return a list of 6 polynomials, each representing a variable."""
    var_polys = List()
    for var_idx in range(6):
        var_polys.append(polynomial_variable(var_idx, max_deg, psi, clmo))
    return var_polys

@njit(fastmath=True, cache=True)
def polynomial_add_inplace(dest: List[np.ndarray], src: List[np.ndarray], scale=1.0, max_deg: int = -1):
    """
    Add src to dest with optional scaling: dest += scale * src (in-place).
    dest[d] += scale * src[d] for each degree d.
    """
    if max_deg == -1:
        loop_limit = min(len(dest), len(src))
    else:
        loop_limit = min(max_deg + 1, len(dest), len(src))

    for d in range(loop_limit):
        if dest[d].size == 0 or src[d].size == 0:
            continue
        if scale == 1.0:
            dest[d] += src[d]
        elif scale == -1.0:
            dest[d] -= src[d]
        else:
            dest[d] += scale * src[d]

@njit(fastmath=True, cache=True)
def polynomial_multiply(a: List[np.ndarray], b: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    c = polynomial_zero_list(max_deg, psi)
    for d1 in range(max_deg + 1):
        if d1 >= len(a) or not np.any(a[d1]):
            continue
        for d2 in range(max_deg + 1 - d1):
            if d2 >= len(b) or not np.any(b[d2]):
                continue
            res_deg = d1 + d2
            prod = _poly_mul(a[d1], d1, b[d2], d2, psi, clmo)
            if prod.shape == c[res_deg].shape:
                c[res_deg] += prod
            elif prod.size == c[res_deg].size:
                c[res_deg] += prod.reshape(c[res_deg].shape)
    return c

@njit(fastmath=True, cache=True)
def polynomial_power(base: List[np.ndarray], k: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    if k == 0:
        res_poly = polynomial_zero_list(max_deg, psi)
        if max_deg >= 0 and len(res_poly) > 0 and res_poly[0].size > 0:
            res_poly[0][0] = 1.0 + 0.0j
        return res_poly

    result = polynomial_zero_list(max_deg, psi)
    if max_deg >= 0 and len(result) > 0 and result[0].size > 0:
        result[0][0] = 1.0 + 0.0j

    active_base = List()
    for arr_idx in range(len(base)):
        active_base.append(base[arr_idx].copy())
        
    exponent = k
    while exponent > 0:
        if exponent % 2 == 1:
            result = polynomial_multiply(result, active_base, max_deg, psi, clmo)
        
        if exponent > 1 :
            active_base = polynomial_multiply(active_base, active_base, max_deg, psi, clmo)
        exponent //= 2
    return result

@njit(fastmath=True, cache=True)
def polynomial_poisson_bracket(a: List[np.ndarray], b: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Compute the Poisson bracket {a, b} of two polynomials.
    Polynomials are represented as lists of coefficient arrays by degree.
    The result is truncated at max_deg.
    """
    c = polynomial_zero_list(max_deg, psi)
    for d1 in range(len(a)):
        if not np.any(a[d1]):
            continue
        for d2 in range(len(b)):
            if not np.any(b[d2]):
                continue

            res_deg = d1 + d2 - 2

            if res_deg < 0 or res_deg > max_deg:
                continue
            
            term_coeffs = poisson(a[d1], d1, b[d2], d2, psi, clmo)
            c[res_deg] += term_coeffs
    return c

@njit(fastmath=True, cache=True)
def polynomial_clean(polys, tol):
    """
    Given a list of coefficient-arrays, return cleaned copies.
    """
    cleaned = []
    for p in polys:
        out = np.empty_like(p)
        _poly_clean(p, tol, out)
        cleaned.append(out)
    return cleaned
