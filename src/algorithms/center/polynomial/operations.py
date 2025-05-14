import numpy as np
from numba.typed import List

from algorithms.variables import N_VARS
from algorithms.center.polynomial.algebra import _poly_mul, _poly_diff, _poly_scale
from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex


# -----------------------------------------------------------------------------
#  GLOBAL clmo cache (Numba functions need it at definition time)
# -----------------------------------------------------------------------------
PSI_GLOBAL, CLMO_GLOBAL = init_index_tables(30)  # default; will be overwritten


def polynomial_zero_list(max_deg: int, psi, complex_dtype=False) -> List[np.ndarray]:
    """Create a list of zero polynomials up to max_deg."""
    return [make_poly(d, psi, complex_dtype) for d in range(max_deg+1)]


def polynomial_variable(idx: int, max_deg: int, psi, complex_dtype=False) -> List[np.ndarray]:
    """
    Create a polynomial representing a single variable x_idx (degree-1 monomial).
    idx: 0…5 corresponding to (x,y,z,px,py,pz) in the current frame.
    """
    pol = polynomial_zero_list(max_deg, psi, complex_dtype)
    k = np.zeros(N_VARS, dtype=np.int64)
    k[idx] = 1
    pol[1][encode_multiindex(k, 1, psi, CLMO_GLOBAL)] = 1.0
    return pol


def polynomial_variables_list(max_deg: int, psi, complex_dtype=False) -> List[List[np.ndarray]]:
    """Return a list of 6 polynomials, each representing a variable."""
    var_polys = []
    for var_idx in range(6):
        var_polys.append(polynomial_variable(var_idx, max_deg, psi, complex_dtype))
    return var_polys


def polynomial_add_inplace(dest: List[np.ndarray], src: List[np.ndarray], scale=1.0):
    """
    Add src to dest with optional scaling: dest += scale * src (in-place).
    dest[d] += scale * src[d] for each degree d.
    """
    for d in range(len(dest)):
        if scale == 1:
            dest[d] += src[d]
        elif scale == -1:
            dest[d] -= src[d]
        else:
            dest[d] += scale * src[d]


def polynomial_multiply(a: List[np.ndarray], b: List[np.ndarray], 
                       max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Multiply polynomials: c = a * b (truncated to max_deg).
    Works on polynomial lists where each element represents a homogeneous 
    polynomial of different degree.
    """
    c = polynomial_zero_list(max_deg, psi, complex_dtype=a[0].dtype == np.complex128)
    for d1 in range(max_deg + 1):
        for d2 in range(max_deg + 1 - d1):
            if not a[d1].any() or not b[d2].any():
                continue
            prod = _poly_mul(a[d1], d1, b[d2], d2, psi, clmo)
            c[d1 + d2] += prod
    return c


def polynomial_power(base: List[np.ndarray], k: int, max_deg: int, psi, clmo) -> List[np.ndarray]:
    """
    Compute base^k (truncated to max_deg) using binary exponentiation.
    Much more efficient than repeated multiplication for large exponents.
    """
    if k == 0:
        # return 1
        out = polynomial_zero_list(max_deg, psi, complex_dtype=base[0].dtype == np.complex128)
        out[0][0] = 1.0
        return out
    out = None
    tmp = base
    e = k
    while e:
        if e & 1:
            out = tmp if out is None else polynomial_multiply(out, tmp, max_deg, psi, clmo)
        tmp = polynomial_multiply(tmp, tmp, max_deg, psi, clmo)
        e >>= 1
    return out
