import numpy as np
import math
from numba import njit
from numba.typed import List

from algorithms.variables import N_VARS
from algorithms.center.polynomial.base import encode_multiindex, decode_multiindex


@njit(fastmath=True, cache=True)
def _poly_add(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i]

@njit(fastmath=True, cache=True)
def _poly_scale(a: np.ndarray, alpha, out: np.ndarray) -> None:
    for i in range(a.shape[0]):
        out[i] = alpha * a[i]

@njit(fastmath=True, cache=True)
def _poly_mul(p: np.ndarray, deg_p: int, q: np.ndarray, deg_q: int, psi, clmo) -> np.ndarray:
    deg_r = deg_p + deg_q
    r = np.zeros(psi[N_VARS, deg_r], dtype=p.dtype)
    for i in range(p.shape[0]):
        pi = p[i]
        if pi == 0:
            continue
        ki = decode_multiindex(i, deg_p, clmo)
        for j in range(q.shape[0]):
            qj = q[j]
            if qj == 0:
                continue
            kj = decode_multiindex(j, deg_q, clmo)
            ks = np.empty(N_VARS, dtype=np.int64)
            for m in range(N_VARS):
                ks[m] = ki[m] + kj[m]
            idx = encode_multiindex(ks, deg_r, psi, clmo)
            r[idx] += pi * qj
    return r

@njit(fastmath=True, cache=True)
def _poly_diff(p: np.ndarray, var: int, degree: int, psi, clmo) -> np.ndarray:
    out_size = psi[N_VARS, degree-1]
    dp = np.zeros(out_size, dtype=p.dtype)
    for i in range(p.shape[0]):
        coeff = p[i]
        if coeff == 0:
            continue
        k = decode_multiindex(i, degree, clmo)
        exp = k[var]
        if exp == 0:
            continue
        k[var] = exp - 1
        idx = encode_multiindex(k, degree-1, psi, clmo)
        dp[idx] += coeff * exp
    return dp

@njit(fastmath=True, cache=True)
def _poly_poisson(p: np.ndarray, deg_p: int, q: np.ndarray, deg_q: int, psi, clmo) -> np.ndarray:
    deg_r = deg_p + deg_q - 2
    r = np.zeros(psi[N_VARS, deg_r], dtype=p.dtype)
    for m in range(3):
        dpx = _poly_diff(p, m, deg_p, psi, clmo)
        dqqp = _poly_diff(q, m+3, deg_q, psi, clmo)
        term1 = _poly_mul(dpx, deg_p-1, dqqp, deg_q-1, psi, clmo)
        for i in range(term1.shape[0]):
            r[i] += term1[i]
        dpq = _poly_diff(p, m+3, deg_p, psi, clmo)
        dqx = _poly_diff(q, m, deg_q, psi, clmo)
        term2 = _poly_mul(dpq, deg_p-1, dqx, deg_q-1, psi, clmo)
        for i in range(term2.shape[0]):
            r[i] -= term2[i]
    return r

@njit(fastmath=True, cache=True)
def _get_degree(poly: np.ndarray, psi) -> int:
    """
    Get the degree of a homogeneous polynomial in our custom representation.

    Parameters
    ----------
    poly : np.ndarray
        Polynomial coefficient array
    psi : 2D array
        Index table used in the polynomial representation.
        psi[N_VARS, d] gives the number of monomials of degree d.
        
    Returns
    -------
    int
        The degree of the polynomial. Returns -1 if degree cannot be determined.
    """
    num_coeffs = poly.shape[0]
    if num_coeffs == 0: # Should not happen for valid polynomials
        return -1 
    
    # N_VARS is imported from algorithms.variables
    # psi.shape[1] is max_degree + 1
    for d in range(psi.shape[1]): 
        if psi[N_VARS, d] == num_coeffs:
            return d
    return -1 # Should not be reached if poly and psi are consistent

@njit(fastmath=True, cache=True)
def _poly_clean_inplace(p: np.ndarray, tol: float) -> None:
    """
    Zero out noise terms in-place in the polynomial coefficient array p.
    Any coefficient whose absolute value is less than or equal to tol is set to zero.
    
    Parameters
    ----------
    p   : np.ndarray
        1D array of complex or real coefficients.
    tol : float
        Threshold below which coefficients are considered numerical noise.
    """
    for i in range(p.shape[0]):
        # np.abs works for real or complex types under numba
        if np.abs(p[i]) <= tol:
            p[i] = 0

@njit(fastmath=True, cache=True)
def _poly_clean(p: np.ndarray, tol: float, out: np.ndarray) -> None:
    """
    Zero out noise terms out-of-place: reads from p, writes cleaned result into out.
    Any coefficient in p whose magnitude is less than or equal to tol becomes 0 in out; otherwise it's copied.
    
    Parameters
    ----------
    p   : np.ndarray
        Input 1D coefficient array.
    tol : float
        Noise threshold.
    out : np.ndarray
        Pre-allocated array of same shape as p. Receives the cleaned coefficients.
    """
    for i in range(p.shape[0]):
        if np.abs(p[i]) <= tol:
            out[i] = 0
        else:
            out[i] = p[i]

@njit(fastmath=True, cache=True)
def _poly_evaluate(
    coeffs: np.ndarray, 
    degree: int, 
    point: np.ndarray, 
    clmo: List[np.ndarray]
) -> np.complex128:
    """
    Evaluate a single homogeneous polynomial at a given point.

    Parameters
    ----------
    coeffs : np.ndarray
        Coefficient array of the homogeneous polynomial.
    degree : int
        Degree of the homogeneous polynomial.
    point : np.ndarray
        The point (array of N_VARS values) at which to evaluate.
        It is assumed that point.shape[0] == N_VARS.
    clmo : numba.typed.List
        Packed multi-indices lookup table.

    Returns
    -------
    np.complex128
        The value of the homogeneous polynomial at the point.
    """
    current_sum = 0.0 + 0.0j
    if coeffs.shape[0] == 0: # Empty polynomial part
        return current_sum

    for i in range(coeffs.shape[0]):
        coeff_val = coeffs[i]
        if coeff_val == 0.0 + 0.0j:
            continue

        exponents = decode_multiindex(i, degree, clmo)
        
        term_val = 1.0 + 0.0j
        for var_idx in range(N_VARS): # N_VARS should be in scope
            exp = exponents[var_idx]
            if exp == 0:
                continue
            elif exp == 1:
                term_val *= point[var_idx]
            else:
                # Using ** operator handles complex base and integer exponent
                term_val *= point[var_idx] ** exp
        
        current_sum += coeff_val * term_val
    return current_sum
