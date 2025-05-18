import numpy as np
from numba import njit, prange
from numba.typed import List

from algorithms.center.polynomial.algebra import (_poly_clean, _poly_diff,
                                                  _poly_evaluate,
                                                  _poly_integrate, _poly_mul,
                                                  _poly_poisson)
from algorithms.center.polynomial.base import encode_multiindex, make_poly
from algorithms.variables import N_VARS


@njit(fastmath=True, cache=True)
def polynomial_zero_list(max_deg: int, psi) -> List[np.ndarray]:
    """Create a list of zero polynomials up to max_deg. Dispatches to JIT-compiled helpers."""
    lst = List()
    for d in range(max_deg + 1):
        lst.append(make_poly(d, psi))
    return lst

@njit(fastmath=True, cache=True)
def polynomial_variable(idx: int, max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
    """
    Create a polynomial representing a single variable x_idx (degree-1 monomial).
    idx: 0-5 corresponding to (x,y,z,px,py,pz) in the current frame.
    """
    pol = polynomial_zero_list(max_deg, psi)
    k = np.zeros(N_VARS, dtype=np.int64)
    k[idx] = 1
    if 1 < len(pol) and pol[1].size > 0:
        encoded_idx = encode_multiindex(k, 1, encode_dict_list)
        if 0 <= encoded_idx < pol[1].shape[0]:
            pol[1][encoded_idx] = 1.0
    return pol

@njit(fastmath=True, cache=True)
def polynomial_variables_list(max_deg: int, psi, clmo, encode_dict_list) -> List[List[np.ndarray]]:
    """Return a list of 6 polynomials, each representing a variable."""
    var_polys = List()
    for var_idx in range(6):
        var_polys.append(polynomial_variable(var_idx, max_deg, psi, clmo, encode_dict_list))
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
def polynomial_multiply(a: List[np.ndarray], b: List[np.ndarray], max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
    c = polynomial_zero_list(max_deg, psi)
    for d1 in range(max_deg + 1):
        if d1 >= len(a) or not np.any(a[d1]):
            continue
        for d2 in range(max_deg + 1 - d1):
            if d2 >= len(b) or not np.any(b[d2]):
                continue
            res_deg = d1 + d2
            prod = _poly_mul(a[d1], d1, b[d2], d2, psi, clmo, encode_dict_list)
            if prod.shape == c[res_deg].shape:
                c[res_deg] += prod
            elif prod.size == c[res_deg].size:
                c[res_deg] += prod.reshape(c[res_deg].shape)
    return c

@njit(fastmath=True, cache=True)
def polynomial_power(base: List[np.ndarray], k: int, max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
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
            result = polynomial_multiply(result, active_base, max_deg, psi, clmo, encode_dict_list)
        
        if exponent > 1 :
            active_base = polynomial_multiply(active_base, active_base, max_deg, psi, clmo, encode_dict_list)
        exponent //= 2
    return result

@njit(fastmath=True, cache=True)
def polynomial_poisson_bracket(a: List[np.ndarray], b: List[np.ndarray], max_deg: int, psi, clmo, encode_dict_list) -> List[np.ndarray]:
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
            
            term_coeffs = _poly_poisson(a[d1], d1, b[d2], d2, psi, clmo, encode_dict_list)
            if term_coeffs.shape == c[res_deg].shape:
                c[res_deg] += term_coeffs
            elif term_coeffs.size == c[res_deg].size and c[res_deg].size > 0:
                c[res_deg] += term_coeffs.reshape(c[res_deg].shape)
    return c

@njit(fastmath=True, cache=True)
def polynomial_clean(polys: List[np.ndarray], tol: float) -> List[np.ndarray]:
    """
    Given a list of coefficient-arrays, return cleaned copies.
    Ensures the returned list is a Numba Typed List.
    """
    # Initialize a Numba Typed List with the correct item type
    # The item type is complex128 1D array, matching the elements of polys.
    cleaned_list = List.empty_list(np.complex128[::1])
    for p_arr in polys:
        out_arr = np.empty_like(p_arr)
        _poly_clean(p_arr, tol, out_arr)
        cleaned_list.append(out_arr)
    return cleaned_list

@njit(fastmath=True, cache=True)
def polynomial_degree(polys: List[np.ndarray]) -> int:
    """
    Get the degree of a polynomial represented as a list of homogeneous parts.

    The degree is the highest index d for which polys[d] contains non-zero coefficients.

    Parameters
    ----------
    polys : List[np.ndarray]
        A list where polys[d] is a NumPy array of coefficients for the
        homogeneous part of degree d.

    Returns
    -------
    int
        The degree of the polynomial. Returns -1 if the polynomial is zero.
    """
    for d in range(len(polys) - 1, -1, -1):
        # Check if any element in the coefficient array for degree d is non-zero
        if np.any(polys[d]):
            return d
    return -1 # All parts are zero or polys is empty

@njit(fastmath=True, cache=True)
def polynomial_differentiate(
    original_coeffs: List[np.ndarray], 
    var_idx: int, 
    original_max_deg: int, 
    original_psi_table: np.ndarray, 
    original_clmo_table: List[np.ndarray],
    derivative_psi_table: np.ndarray,
    derivative_clmo_table: List[np.ndarray],
    encode_dict_list: List
):
    """
    Differentiates a polynomial (list of coefficient arrays) with respect to a variable.
    The caller is responsible for providing psi/clmo tables for the derivative.

    Parameters
    ----------
    original_coeffs : List[np.ndarray]
    var_idx : int
    original_max_deg : int
    original_psi_table : np.ndarray
    original_clmo_table : List[np.ndarray]
    derivative_psi_table : np.ndarray
        Pre-initialized psi_table for the derivative's max_deg.
    derivative_clmo_table : List[np.ndarray]
        Pre-initialized clmo_table for the derivative's max_deg.
    encode_dict_list : List
        Encode dictionary list for the derivative's max_deg.

    Returns
    -------
    Tuple[List[np.ndarray], int]
        A tuple containing:
        - derivative_coeffs_list: List of coefficient arrays for the derivative.
        - derivative_max_deg: Maximum degree of the derivative.
    """
    derivative_max_deg = original_max_deg - 1
    if derivative_max_deg < 0:
        derivative_max_deg = 0

    derivative_coeffs_list = polynomial_zero_list(derivative_max_deg, derivative_psi_table)

    for d_orig in range(1, original_max_deg + 1):
        d_res = d_orig - 1 
        
        if d_res <= derivative_max_deg:
            if d_orig < len(original_coeffs) and np.any(original_coeffs[d_orig]):
                term_diff_coeffs = _poly_diff(
                    original_coeffs[d_orig], 
                    var_idx, 
                    d_orig, 
                    original_psi_table, 
                    original_clmo_table,
                    encode_dict_list
                )
                
                if d_res < len(derivative_coeffs_list) and derivative_coeffs_list[d_res].shape[0] == term_diff_coeffs.shape[0]:
                    derivative_coeffs_list[d_res] = term_diff_coeffs

    return derivative_coeffs_list, derivative_max_deg

@njit(fastmath=True, cache=True)
def polynomial_jacobian(
    poly_coeffs: List[np.ndarray],
    original_max_deg: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List
) -> List[List[np.ndarray]]:
    """
    Computes the Jacobian of a polynomial with respect to all N_VARS variables.
    Returns a list of N_VARS polynomials, where each polynomial is the
    partial derivative of the input polynomial with respect to one variable.
    Each derivative polynomial is represented as a list of coefficient arrays.

    Parameters
    ----------
    poly_coeffs : List[np.ndarray]
        Coefficient arrays of the polynomial to differentiate.
    original_max_deg : int
        Maximum degree of the input polynomial.
    psi_table : np.ndarray
        PSI table for the input polynomial's degree structure. This table will
        also be used for the derivatives, assuming it's sufficiently large.
    clmo_table : List[np.ndarray]
        CLMO table for the input polynomial's degree structure. This table will
        also be used for the derivatives.
    encode_dict_list : List
        Encode dictionary list for the derivative's max_deg.

    Returns
    -------
    List[List[np.ndarray]]
        A Numba typed list containing N_VARS items. Each item is a
        derivative polynomial (itself a Numba typed list of coefficient arrays).
    """
    jacobian_list = List.empty_list(List.empty_list(np.complex128[::1])) # Typed list for list of polynomials

    for i in prange(N_VARS): # Iterate over all variables
        derivative_poly_coeffs, _ = polynomial_differentiate(
            original_coeffs=poly_coeffs,
            var_idx=i,
            original_max_deg=original_max_deg,
            original_psi_table=psi_table,
            original_clmo_table=clmo_table,
            derivative_psi_table=psi_table,  # Use original psi table for derivative
            derivative_clmo_table=clmo_table, # Use original clmo table for derivative
            encode_dict_list=encode_dict_list
        )
        jacobian_list.append(derivative_poly_coeffs)
    
    return jacobian_list

@njit(fastmath=True, cache=True)
def polynomial_evaluate(
    polys: List[np.ndarray], 
    point: np.ndarray, 
    clmo: List[np.ndarray] # Typically CLMO_GLOBAL
) -> np.complex128:
    """
    Evaluate a polynomial (represented as a list of homogeneous parts) at a point.

    Parameters
    ----------
    polys : List[np.ndarray]
        A list where polys[d] is the coefficient array for the homogeneous part of degree d.
    point : np.ndarray
        The point (array of N_VARS values) at which to evaluate.
        It is assumed that point.shape[0] == N_VARS.
    clmo : numba.typed.List
        Packed multi-indices lookup table.

    Returns
    -------
    np.complex128
        The total value of the polynomial at the point.
    """
    total_value = 0.0 + 0.0j
    for degree in range(len(polys)):
        coeffs_d = polys[degree]
        if coeffs_d.shape[0] > 0: # Check if there are coefficients for this degree
            total_value += _poly_evaluate(coeffs_d, degree, point, clmo)
    return total_value

@njit(fastmath=True, cache=True)
def polynomial_integrate(
    original_coeffs: List[np.ndarray],
    var_idx: int,
    original_max_deg: int,
    original_psi_table: np.ndarray,
    original_clmo_table: List[np.ndarray],
    integral_psi_table: np.ndarray,
    integral_clmo_table: List[np.ndarray],
    encode_dict_list: List
) -> tuple[List[np.ndarray], int]:
    """
    Integrates a polynomial (list of coefficient arrays) with respect to a variable.
    The caller is responsible for providing psi/clmo tables for the integral's degree structure.

    Returns
    -------
    Tuple[List[np.ndarray], int]
        A tuple containing:
        - integral_coeffs_list: List of coefficient arrays for the integral.
        - integral_max_deg: Maximum degree of the integral.
    """
    integral_max_deg = original_max_deg + 1
    # Ensure integral_coeffs_list is initialized up to integral_max_deg using integral_psi_table
    integral_coeffs_list = polynomial_zero_list(integral_max_deg, integral_psi_table)

    for d_orig in range(original_max_deg + 1): # Iterate through all degrees of original polynomial
        d_res = d_orig + 1 # Degree of the result of integrating this part
        
        # Ensure the resulting degree fits within the pre-allocated list for the integral
        if d_res <= integral_max_deg:
            if d_orig < len(original_coeffs) and np.any(original_coeffs[d_orig]):
                term_integral_coeffs = _poly_integrate(
                    original_coeffs[d_orig],
                    var_idx,
                    d_orig,
                    original_psi_table,
                    original_clmo_table,
                    encode_dict_list
                )
                
                # Add the integrated term to the correct degree in the result list
                # The term_integral_coeffs is for degree d_res (i.e., d_orig + 1)
                if d_res < len(integral_coeffs_list) and integral_coeffs_list[d_res].shape[0] == term_integral_coeffs.shape[0]:
                    integral_coeffs_list[d_res] += term_integral_coeffs
                elif d_res < len(integral_coeffs_list) and integral_coeffs_list[d_res].size == term_integral_coeffs.size and term_integral_coeffs.size > 0:
                    integral_coeffs_list[d_res] += term_integral_coeffs.reshape(integral_coeffs_list[d_res].shape)

    return integral_coeffs_list, integral_max_deg