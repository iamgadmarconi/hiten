r"""
polynomial.conversion
=====================

Helpers that convert between the internal *coefficient-array* representation of multivariate polynomials and symbolic :class:`sympy.Expr` objects.
The six canonical variables are ordered as :math:`(q_1, q_2, q_3, p_1, p_2, p_3)`.

The module exposes:

* :pyfunc:`poly2sympy` - coefficient arrays to SymPy expression.
* :pyfunc:`sympy2poly` - SymPy expression to coefficient arrays.
* :pyfunc:`hpoly2sympy` - single homogeneous block to SymPy expression.

These routines are mainly intended for debugging and diagnostic workflows where readability outweighs raw speed.
"""
import typing

import numpy as np
import sympy as sp
from numba.typed import List

from hiten.algorithms.polynomial.algebra import _get_degree
from hiten.algorithms.polynomial.base import (decode_multiindex,
                                               encode_multiindex, make_poly)
from hiten.utils.config import N_VARS


def poly2sympy(poly_p: List[np.ndarray], vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray) -> sp.Expr:
    r"""
    Convert a polynomial represented as a list of coefficient arrays to a SymPy expression.
    
    Parameters
    ----------
    poly_p : List[numpy.ndarray]
        List of coefficient arrays, where poly_p[d] contains coefficients for degree d
    vars_list : typing.List[sympy.Symbol]
        List of SymPy symbols used as variables in the expression
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
        
    Returns
    -------
    sympy.Expr
        SymPy expression equivalent to the input polynomial
        
    Raises
    ------
    ValueError
        If vars_list does not have exactly N_VARS elements
        
    Notes
    -----
    This function converts each homogeneous part of the polynomial separately 
    with :pyfunc:`hpoly2sympy`, then combines them into a single SymPy expression.
    """
    if len(vars_list) != N_VARS:
        raise ValueError(f"Expected {N_VARS} symbols in vars_list, but got {len(vars_list)}.")

    total_sympy_expr = sp.Integer(0)
    for degree, p in enumerate(poly_p):
        if p is not None and p.size > 0:
            # hpoly2sympy needs clmo for decode_multiindex
            homogeneous_expr = hpoly2sympy(p, vars_list, psi, clmo)
            total_sympy_expr += homogeneous_expr
    return total_sympy_expr


def sympy2poly(expr: sp.Expr, vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray, encode_dict_list: List) -> List[np.ndarray]:
    r"""
    Convert a SymPy expression to a polynomial represented as a list of coefficient arrays.
    
    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression to convert
    vars_list : typing.List[sympy.Symbol]
        List of SymPy symbols used as variables in the expression
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
        
    Returns
    -------
    List[numpy.ndarray]
        List of coefficient arrays, where the d-th element contains 
        coefficients for the homogeneous part of degree d
        
    Raises
    ------
    ValueError
        If vars_list does not have exactly N_VARS elements or if the expression's
        degree exceeds the precomputed table limits
    TypeError
        If the expression cannot be converted to a Sympy Poly object or if
        conversion of coefficients to numeric types fails
    IndexError
        If the encoded position for a term is out of bounds
        
    Notes
    -----
    The function works by converting the SymPy expression to a Poly object,
    then extracting terms and mapping them to the appropriate positions in
    the coefficient arrays.
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
    poly_p = [make_poly(d, psi) for d in range(max_deg_expr + 1)]

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
        
        if term_degree >= len(poly_p):
             raise IndexError(
                 f"Calculated term degree {term_degree} is out of bounds for pre-allocated "
                 f"coeffs_list (len {len(poly_p)}, max_deg_expr {max_deg_expr}). This indicates an internal logic error."
            )
        if pos >= poly_p[term_degree].shape[0]:
            raise IndexError(
                f"Encoded position {pos} is out of bounds for coefficient array of degree {term_degree} "
                f"(size {poly_p[term_degree].shape[0]}). This indicates an internal logic error or table inconsistency."
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

            poly_p[term_degree][pos] = numeric_coeff
        except TypeError: # Catch if conversion fails (e.g., contains symbols)
            raise TypeError(
                f"Coefficient '{coeff_val_sympy}' (type: {type(coeff_val_sympy)}) could not be converted to a Python numeric type. "
                "Ensure the Sympy expression has purely numeric coefficients."
            )
        except Exception as e: # Catch any other unexpected errors during conversion
            raise TypeError(f"Failed to process Sympy coefficient '{coeff_val_sympy}': {e}")
            
    return poly_p


def hpoly2sympy(p: np.ndarray, vars_list: typing.List[sp.Symbol], psi: np.ndarray, clmo: np.ndarray) -> sp.Expr:
    r"""
    Convert a homogeneous polynomial coefficient array to a SymPy expression.
    
    Parameters
    ----------
    p : numpy.ndarray
        Coefficient array for a homogeneous polynomial part of a specific degree
    vars_list : typing.List[sympy.Symbol]
        List of SymPy symbols used as variables in the expression
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
        
    Returns
    -------
    sympy.Expr
        SymPy expression equivalent to the input homogeneous polynomial
        
    Raises
    ------
    ValueError
        If the degree cannot be determined from the coefficient array size
        or if the coefficient array length is inconsistent with the expected size
        
    Notes
    -----
    This function converts each term of the homogeneous polynomial by decoding
    the multi-index with :pyfunc:`decode_multiindex` to determine the
    exponents, constructing the corresponding monomial, and multiplying it by
    the coefficient.
    """
    if p is None or p.size == 0:
        return sp.Integer(0)

    degree = _get_degree(p, psi)

    if degree == -1:
        if p.size > 0:
             raise ValueError(
                f"Cannot determine degree for homogeneous polynomial with {p.size} coefficients."
            )
        return sp.Integer(0)

    sympy_expr = sp.Integer(0)
    num_coefficients = psi[N_VARS, degree]

    if len(p) != num_coefficients:
        raise ValueError(
            f"Inconsistent coefficient array length. Expected {num_coefficients} for degree {degree}, got {len(p)}."
        )

    for pos in range(len(p)):
        coeff = p[pos]

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

