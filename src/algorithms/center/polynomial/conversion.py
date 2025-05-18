import typing

import numpy as np
import sympy as sp
from numba.typed import List

from algorithms.center.polynomial.algebra import _get_degree
from algorithms.center.polynomial.base import (decode_multiindex,
                                               encode_multiindex, make_poly,
                                               PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
from algorithms.variables import N_VARS


def poly2sympy(poly_list: List[np.ndarray], vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray) -> sp.Expr:
    """
    Convert a polynomial, represented as a list of homogeneous component coefficient arrays,
    from numpy array to sympy expression using the provided list of sympy variables.
    Each element in poly_list is an np.ndarray of coefficients for a specific degree,
    starting from degree 0.
    vars_list must contain N_VARS sympy symbols.
    clmo is passed to decode_multiindex.
    """
    if len(vars_list) != N_VARS:
        raise ValueError(f"Expected {N_VARS} symbols in vars_list, but got {len(vars_list)}.")

    total_sympy_expr = sp.Integer(0)
    for degree, coeffs_for_degree in enumerate(poly_list):
        if coeffs_for_degree is not None and coeffs_for_degree.size > 0:
            # hpoly2sympy needs clmo for decode_multiindex
            homogeneous_expr = hpoly2sympy(coeffs_for_degree, vars_list, psi, clmo)
            total_sympy_expr += homogeneous_expr
    return total_sympy_expr


def sympy2poly(expr: sp.Expr, vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray, encode_dict_list: List) -> List[np.ndarray]:
    """
    Convert a sympy expression to a list of numpy arrays (custom polynomial representation)
    using the provided list of sympy variables.
    vars_list must contain N_VARS sympy symbols.
    The expression is assumed to be a polynomial in vars_list with numeric coefficients.
    psi is used for make_poly and degree checks.
    encode_dict_list is used for encode_multiindex.
    clmo is not directly used by sympy2poly itself but often passed alongside psi.
    """
    if len(vars_list) != N_VARS:
        raise ValueError(f"Expected {N_VARS} symbols in vars_list, but got {len(vars_list)}.")

    if expr == sp.S.Zero:
        # Return a list representing a zero polynomial (degree 0, coefficient 0)
        return [make_poly(0, psi)]

    # Attempt to convert the expression to a Sympy Poly object
    try:
        sp_poly = sp.Poly(expr, *vars_list)
    except Exception as e:
        raise TypeError(f"Could not convert expr to Sympy Poly object using vars_list: {vars_list}. Error: {e}")

    if not isinstance(sp_poly, sp.Poly):
        # This case might occur if expr is, for example, a list or non-convertible type
        raise TypeError(f"Input expr (type: {type(expr)}) did not convert to a Sympy Poly object.")
    
    if sp_poly.is_zero:
        return [make_poly(0, psi)]

    # Determine the maximum degree of the polynomial
    max_deg_expr = -1
    if not sp_poly.is_zero: # Should always be true if we passed the previous check
        for monom_exp, _ in sp_poly.terms():
            current_deg = sum(monom_exp)
            if current_deg > max_deg_expr:
                max_deg_expr = int(current_deg)
    
    if max_deg_expr == -1 : # Should only happen if sp_poly was zero, handled already. Safety.
        return [make_poly(0, psi)]

    # Check if the polynomial's degree exceeds precomputed table limits
    max_supported_degree = psi.shape[1] - 1
    if max_deg_expr > max_supported_degree:
        raise ValueError(
            f"Expression degree ({max_deg_expr}) exceeds precomputed table limit ({max_supported_degree}). "
            "Re-initialize psi/clmo with a higher max_degree if needed."
        )

    # Initialize list of coefficient arrays (one for each degree up to max_deg_expr)
    coeffs_list = [make_poly(d, psi) for d in range(max_deg_expr + 1)]

    # Populate coefficient arrays
    for monom_exp_tuple, coeff_val_sympy in sp_poly.terms():
        k_np = np.array(monom_exp_tuple, dtype=np.int64)
        
        if len(k_np) != N_VARS:
            # This should be guaranteed by sp.Poly if constructed with N_VARS generators
            raise ValueError(
                f"Monomial exponent tuple {monom_exp_tuple} from Sympy Poly does not have {N_VARS} elements "
                f"for vars_list: {vars_list}."
            )

        term_degree = int(sum(k_np))

        # Get position in our coefficient array using encode_multiindex
        pos = encode_multiindex(k_np, term_degree, encode_dict_list)

        if pos == -1:
            # This can happen if term_degree > max_degree for clmo or other encoding issues
            raise ValueError(
                f"Failed to encode multi-index {k_np.tolist()} for degree {term_degree}. "
                "This may indicate an unsupported monomial, a degree outside clmo table range, or an internal error."
            )
        
        if term_degree >= len(coeffs_list):
             raise IndexError(
                 f"Calculated term degree {term_degree} is out of bounds for pre-allocated "
                 f"coeffs_list (len {len(coeffs_list)}, max_deg_expr {max_deg_expr}). This indicates an internal logic error."
            )
        if pos >= coeffs_list[term_degree].shape[0]:
            raise IndexError(
                f"Encoded position {pos} is out of bounds for coefficient array of degree {term_degree} "
                f"(size {coeffs_list[term_degree].shape[0]}). This indicates an internal logic error or table inconsistency."
            )

        # Convert Sympy coefficient to a Python complex number and store it
        try:
            # Sympy's complex() or direct float()/int() conversion should handle its own numeric types (Number, Add, Mul for complex)
            # If coeff_val_sympy contains symbols, this will raise a TypeError.
            if coeff_val_sympy.is_imaginary or isinstance(coeff_val_sympy, sp.Add) or hasattr(coeff_val_sympy, 'as_real_imag'): # Check if it could be complex
                 numeric_coeff = complex(coeff_val_sympy)
            elif coeff_val_sympy.is_real:
                 numeric_coeff = float(coeff_val_sympy)
            else: # Fallback, attempt complex, or could be an int if is_integer
                 numeric_coeff = complex(coeff_val_sympy) # Default attempt complex for safety for other numeric types

            coeffs_list[term_degree][pos] = numeric_coeff
        except TypeError: # Catch if conversion fails (e.g., contains symbols)
            raise TypeError(
                f"Coefficient '{coeff_val_sympy}' (type: {type(coeff_val_sympy)}) could not be converted to a Python numeric type. "
                "Ensure the Sympy expression has purely numeric coefficients."
            )
        except Exception as e: # Catch any other unexpected errors during conversion
            raise TypeError(f"Failed to process Sympy coefficient '{coeff_val_sympy}': {e}")
            
    return coeffs_list


def hpoly2sympy(poly_coeffs: np.ndarray, vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray) -> sp.Expr:
    """
    Convert a homogeneous polynomial from numpy array of coefficients to sympy expression
    using the provided list of sympy variables.
    vars_list must contain N_VARS sympy symbols (checked by caller or implicitly here).
    clmo is used for decode_multiindex.
    """
    if poly_coeffs is None or poly_coeffs.size == 0:
        return sp.Integer(0)

    degree = _get_degree(poly_coeffs, psi)

    if degree == -1:
        if poly_coeffs.size > 0:
             raise ValueError(
                f"Cannot determine degree for homogeneous polynomial with {poly_coeffs.size} coefficients."
            )
        return sp.Integer(0)

    sympy_expr = sp.Integer(0)
    num_coefficients = psi[N_VARS, degree]

    if len(poly_coeffs) != num_coefficients:
        raise ValueError(
            f"Inconsistent coefficient array length. Expected {num_coefficients} for degree {degree}, got {len(poly_coeffs)}."
        )

    for pos in range(len(poly_coeffs)):
        coeff = poly_coeffs[pos]

        if isinstance(coeff, float) and np.isclose(coeff, 0.0):
            continue
        if isinstance(coeff, complex) and np.isclose(coeff.real, 0.0) and np.isclose(coeff.imag, 0.0):
            continue
        if coeff == 0:
            continue

        k_vector = decode_multiindex(pos, degree, clmo)
        
        monomial_expr = sp.Integer(1)
        for i in range(N_VARS):
            if k_vector[i] > 0:
                monomial_expr *= vars_list[i]**k_vector[i]
        
        sympy_expr += coeff * monomial_expr
        
    return sympy_expr

