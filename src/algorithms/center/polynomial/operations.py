import numpy as np
from numba import njit
from numba.typed import List

from algorithms.variables import N_VARS
from algorithms.center.polynomial.algebra import _poly_mul, _poly_diff, _poly_scale
from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex, _make_poly_real, _make_poly_complex, PSI_GLOBAL, CLMO_GLOBAL


@njit(fastmath=True, cache=True)
def _polynomial_zero_list_real(max_deg: int, psi) -> List[np.ndarray]:
    """Helper: Create a List of zero real polynomials up to max_deg."""
    lst = List()
    for d in range(max_deg + 1):
        lst.append(_make_poly_real(d, psi))
    return lst

@njit(fastmath=True, cache=True)
def _polynomial_zero_list_complex(max_deg: int, psi) -> List[np.ndarray]:
    """Helper: Create a List of zero complex polynomials up to max_deg."""
    lst = List()
    for d in range(max_deg + 1):
        lst.append(_make_poly_complex(d, psi))
    return lst

def polynomial_zero_list(max_deg: int, psi, complex_dtype=False) -> List[np.ndarray]:
    """Create a list of zero polynomials up to max_deg. Dispatches to JIT-compiled helpers."""
    if complex_dtype:
        return _polynomial_zero_list_complex(max_deg, psi)
    else:
        return _polynomial_zero_list_real(max_deg, psi)


def polynomial_variable(idx: int, max_deg: int, psi, complex_dtype=False) -> List[np.ndarray]:
    """
    Create a polynomial representing a single variable x_idx (degree-1 monomial).
    idx: 0…5 corresponding to (x,y,z,px,py,pz) in the current frame.
    """
    pol = polynomial_zero_list(max_deg, psi, complex_dtype)
    k = np.zeros(N_VARS, dtype=np.int64)
    k[idx] = 1
    if 1 < len(pol) and pol[1].size > 0:
        encoded_idx = encode_multiindex(k, 1, psi, CLMO_GLOBAL)
        if 0 <= encoded_idx < pol[1].shape[0]:
             pol[1][encoded_idx] = 1.0
    return pol


def polynomial_variables_list(max_deg: int, psi, complex_dtype=False) -> List[List[np.ndarray]]:
    """Return a list of 6 polynomials, each representing a variable."""
    var_polys = List()
    for var_idx in range(6):
        var_polys.append(polynomial_variable(var_idx, max_deg, psi, complex_dtype))
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
def _polynomial_multiply_real(a: List[np.ndarray], b: List[np.ndarray],
                              out_max_deg: int, psi, clmo) -> List[np.ndarray]:
    c = _polynomial_zero_list_real(out_max_deg, psi)
    for d1 in range(out_max_deg + 1):
        if d1 >= len(a) or not np.any(a[d1]):
            continue
        for d2 in range(out_max_deg + 1 - d1):
            if d2 >= len(b) or not np.any(b[d2]):
                continue
            res_deg = d1 + d2
            prod = _poly_mul(a[d1], d1, b[d2], d2, psi, clmo)
            if prod.shape == c[res_deg].shape :
                 c[res_deg] += prod
            elif prod.size == c[res_deg].size:
                 c[res_deg] += prod.reshape(c[res_deg].shape)
    return c

@njit(fastmath=True, cache=True)
def _polynomial_multiply_complex(a: List[np.ndarray], b: List[np.ndarray],
                                 out_max_deg: int, psi, clmo) -> List[np.ndarray]:
    c = _polynomial_zero_list_complex(out_max_deg, psi)
    for d1 in range(out_max_deg + 1):
        if d1 >= len(a) or not np.any(a[d1]):
            continue
        for d2 in range(out_max_deg + 1 - d1):
            if d2 >= len(b) or not np.any(b[d2]):
                continue
            res_deg = d1 + d2
            prod = _poly_mul(a[d1], d1, b[d2], d2, psi, clmo)
            if prod.shape == c[res_deg].shape :
                 c[res_deg] += prod
            elif prod.size == c[res_deg].size:
                 c[res_deg] += prod.reshape(c[res_deg].shape)
    return c

def polynomial_multiply(a: List[np.ndarray], b: List[np.ndarray],
                       max_deg: int, psi, clmo) -> List[np.ndarray]:
    is_complex_computation = False
    if len(a) > 0 and a[0].dtype == np.complex128:
        is_complex_computation = True
    elif len(b) > 0 and b[0].dtype == np.complex128:
        is_complex_computation = True
    
    if is_complex_computation:
        return _polynomial_multiply_complex(a, b, max_deg, psi, clmo)
    else:
        return _polynomial_multiply_real(a, b, max_deg, psi, clmo)


@njit(fastmath=True, cache=True)
def _polynomial_power_real(base_in: List[np.ndarray], k: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    if k == 0:
        res_poly = _polynomial_zero_list_real(max_deg, psi)
        if max_deg >= 0 and len(res_poly) > 0 and res_poly[0].size > 0:
             res_poly[0][0] = 1.0
        return res_poly

    result = _polynomial_zero_list_real(max_deg, psi)
    if max_deg >= 0 and len(result) > 0 and result[0].size > 0:
        result[0][0] = 1.0

    active_base = List()
    for arr_idx in range(len(base_in)):
        active_base.append(base_in[arr_idx].copy())

    exponent = k
    while exponent > 0:
        if exponent % 2 == 1:
            result = _polynomial_multiply_real(result, active_base, max_deg, psi, clmo)
        
        if exponent > 1 : 
            active_base = _polynomial_multiply_real(active_base, active_base, max_deg, psi, clmo)
        exponent //= 2
    return result

@njit(fastmath=True, cache=True)
def _polynomial_power_complex(base_in: List[np.ndarray], k: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    if k == 0:
        res_poly = _polynomial_zero_list_complex(max_deg, psi)
        if max_deg >= 0 and len(res_poly) > 0 and res_poly[0].size > 0:
             res_poly[0][0] = 1.0 + 0.0j
        return res_poly

    result = _polynomial_zero_list_complex(max_deg, psi)
    if max_deg >= 0 and len(result) > 0 and result[0].size > 0:
        result[0][0] = 1.0 + 0.0j

    active_base = List()
    for arr_idx in range(len(base_in)):
        active_base.append(base_in[arr_idx].copy())
        
    exponent = k
    while exponent > 0:
        if exponent % 2 == 1:
            result = _polynomial_multiply_complex(result, active_base, max_deg, psi, clmo)
        
        if exponent > 1 :
            active_base = _polynomial_multiply_complex(active_base, active_base, max_deg, psi, clmo)
        exponent //= 2
    return result

def polynomial_power(base: List[np.ndarray], k: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    if k < 0:
        raise ValueError("Polynomial power k must be non-negative.")
    
    is_complex_computation = False
    if len(base) > 0 and base[0].dtype == np.complex128:
        is_complex_computation = True
        
    if is_complex_computation:
        return _polynomial_power_complex(base, k, max_deg, psi, clmo)
    else:
        return _polynomial_power_real(base, k, max_deg, psi, clmo)
