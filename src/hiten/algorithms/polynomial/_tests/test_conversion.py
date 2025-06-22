import numpy as np
import pytest
import sympy as sp
from numba.typed import List

from hiten.algorithms.polynomial.base import (CLMO_GLOBAL, ENCODE_DICT_GLOBAL,
                                               PSI_GLOBAL, encode_multiindex,
                                               make_poly)
from hiten.algorithms.polynomial.conversion import poly2sympy, sympy2poly
from hiten.utils.config import N_VARS

s_vars = list(sp.symbols(f'x_0:{N_VARS}'))
TEST_MAX_DEG = PSI_GLOBAL.shape[1] - 1


def _create_custom_poly_list(max_deg: int, terms: dict) -> List[np.ndarray]:
    py_list_of_coeffs = [make_poly(d, PSI_GLOBAL) for d in range(max_deg + 1)]

    for k_tuple, coeff_val in terms.items():
        k_np = np.array(k_tuple, dtype=np.int64)
        deg = int(sum(k_np))
        pos = encode_multiindex(k_np, deg, ENCODE_DICT_GLOBAL)
        py_list_of_coeffs[deg][pos] = complex(coeff_val)
    
    numba_list = List()
    for arr in py_list_of_coeffs:
        numba_list.append(arr)
    return numba_list


def _compare_poly_lists(list1: List[np.ndarray], list2: List[np.ndarray], tol=1e-12) -> bool:
    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        arr1, arr2 = list1[i], list2[i]
        if not np.allclose(arr1, arr2, atol=tol, rtol=tol):
            print(f"Data mismatch at degree {i}:\nArr1: {arr1}\nArr2: {arr2}")
            return False
    return True


def test_poly2sympy_zero():
    poly_list_zero = _create_custom_poly_list(0, {}) # Degree 0, no terms
    expr = poly2sympy(poly_list_zero, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr == sp.Integer(0)

    poly_list_zero_deg2 = _create_custom_poly_list(2, {}) # Up to degree 2, all zero
    expr2 = poly2sympy(poly_list_zero_deg2, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr2 == sp.Integer(0)
    
    empty_numba_list = List() 
    expr_empty = poly2sympy(empty_numba_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr_empty == sp.Integer(0)


def test_sympy2poly_zero():
    expr = sp.Integer(0)
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    expected_coeffs = make_poly(0, PSI_GLOBAL)
    expected_list = List()
    expected_list.append(expected_coeffs)
    
    assert _compare_poly_lists(poly_list, expected_list)


def test_poly2sympy_constant():
    const_val = 5.5
    poly_list = _create_custom_poly_list(0, {(0,)*N_VARS: const_val})
    expr = poly2sympy(poly_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr == sp.Float(const_val)


def test_sympy2poly_constant():
    const_val = 7.0
    expr = sp.Float(const_val)
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    expected_coeffs = make_poly(0, PSI_GLOBAL)
    expected_coeffs[encode_multiindex(np.zeros(N_VARS, dtype=np.int64), 0, ENCODE_DICT_GLOBAL)] = const_val
    expected_list = List()
    expected_list.append(expected_coeffs)
    
    assert _compare_poly_lists(poly_list, expected_list)


def test_poly2sympy_single_variable():
    target_var_idx = 1
    k_tuple = [0]*N_VARS
    k_tuple[target_var_idx] = 1
    k_tuple = tuple(k_tuple)

    poly_list = _create_custom_poly_list(1, {k_tuple: 1.0})
    expr = poly2sympy(poly_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert sp.simplify(expr - s_vars[target_var_idx]) == 0


def test_sympy2poly_single_variable():
    target_var_idx = 2
    expr = s_vars[target_var_idx]
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)

    k_tuple = [0]*N_VARS
    k_tuple[target_var_idx] = 1
    k_tuple = tuple(k_tuple)
    
    expected_list = _create_custom_poly_list(1, {k_tuple: 1.0})
    assert _compare_poly_lists(poly_list, expected_list)


def test_round_trip():
    # P(x) = 2.0*x_0 + 3.0*x_1^2
    expr_original = 2.0 * s_vars[0] + 3.0 * s_vars[1]**2
    
    poly_list_intermediate = sympy2poly(expr_original, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    expr_reconstructed = poly2sympy(poly_list_intermediate, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    
    assert sp.simplify(expr_original - expr_reconstructed) == 0

    poly_list_final = sympy2poly(expr_reconstructed, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    assert _compare_poly_lists(poly_list_intermediate, poly_list_final)

    # P(x) = (1+2j)*x_0*x_1 + (3-1j)*x_2^3
    expr_original = (1+2j)*s_vars[0]*s_vars[1] + (3-1j)*s_vars[2]**3
    
    poly_list_intermediate = sympy2poly(expr_original, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    expr_reconstructed = poly2sympy(poly_list_intermediate, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    poly_list_final = sympy2poly(expr_reconstructed, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)

    assert sp.simplify(expr_original - expr_reconstructed) == 0
    assert _compare_poly_lists(poly_list_intermediate, poly_list_final)


def test_heterogeneous_polynomial_conversion():

    terms_dict = {
        (0,0,0,0,0,0): 1.5,
        (1,0,0,0,0,0): 2.0,
        (0,1,1,0,0,0): -0.5,
        (0,0,0,3,0,0): 3.0
    }
    max_deg_poly = 3
    
    custom_poly = _create_custom_poly_list(max_deg_poly, terms_dict)
    
    expected_sympy_expr = (
        sp.Float(1.5) + 
        2.0 * s_vars[0] - 
        0.5 * s_vars[1] * s_vars[2] + 
        3.0 * s_vars[3]**3
    )
    
    generated_sympy_expr = poly2sympy(custom_poly, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert sp.simplify(generated_sympy_expr - expected_sympy_expr) == 0
    
    generated_custom_poly = sympy2poly(expected_sympy_expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    assert _compare_poly_lists(custom_poly, generated_custom_poly)



