import numpy as np
import pytest
from numba.typed import List
import symengine as se
import re # Import re module

from algorithms.center.polynomial.base import init_index_tables, encode_multiindex
from algorithms.center.polynomial.conversions import (
    _extract_symengine_term_details,
    symengine2poly,
    poly2symengine,
)
from algorithms.variables import N_VARS, q1, q2, q3, p1, p2, p3

# Initialize tables for tests
MAX_DEGREE = 5
PSI, CLMO = init_index_tables(MAX_DEGREE)

# Define Symengine variables for tests, matching N_VARS and typical order
# Assuming variables are x0, x1, x2, x3, x4, x5 based on N_VARS
if N_VARS == 6:
    s_vars = se.symbols(f'x0:{N_VARS}') 
else:
    # Fallback for different N_VARS, though tests are geared towards N_VARS=6
    s_vars = se.symbols(','.join([f'x{i}' for i in range(N_VARS)]))


def test_extract_symengine_term_details_simple_monomials():
    """Test _extract_symengine_term_details with simple monomials."""
    x0, x1, x2, x3, x4, x5 = s_vars

    # Test 1: Constant term
    term1 = se.Integer(5)
    coeff1, k1, deg1 = _extract_symengine_term_details(term1, list(s_vars))
    assert coeff1 == 5.0
    np.testing.assert_array_equal(k1, np.array([0]*N_VARS, dtype=np.int64))
    assert deg1 == 0

    # Test 2: Linear term: 3*x0
    term2 = 3 * x0
    coeff2, k2, deg2 = _extract_symengine_term_details(term2, list(s_vars))
    assert coeff2 == 3.0
    expected_k2 = np.zeros(N_VARS, dtype=np.int64)
    expected_k2[0] = 1
    np.testing.assert_array_equal(k2, expected_k2)
    assert deg2 == 1

    # Test 3: Quadratic term: 2*x1**2
    term3 = 2 * x1**2
    coeff3, k3, deg3 = _extract_symengine_term_details(term3, list(s_vars))
    assert coeff3 == 2.0
    expected_k3 = np.zeros(N_VARS, dtype=np.int64)
    expected_k3[1] = 2
    np.testing.assert_array_equal(k3, expected_k3)
    assert deg3 == 2

    # Test 4: Mixed term: 4*x0*x2**3
    term4 = 4 * x0 * x2**3
    coeff4, k4, deg4 = _extract_symengine_term_details(term4, list(s_vars))
    assert coeff4 == 4.0
    expected_k4 = np.zeros(N_VARS, dtype=np.int64)
    expected_k4[0] = 1
    expected_k4[2] = 3
    np.testing.assert_array_equal(k4, expected_k4)
    assert deg4 == 4

    # Test 5: Term with coefficient 1: x0
    term5 = x0
    coeff5, k5, deg5 = _extract_symengine_term_details(term5, list(s_vars))
    assert coeff5 == 1.0
    expected_k5 = np.zeros(N_VARS, dtype=np.int64)
    expected_k5[0] = 1
    np.testing.assert_array_equal(k5, expected_k5)
    assert deg5 == 1
    
    # Test 6: Term x0*x1 (product of symbols)
    term6 = x0 * x1
    coeff6, k6, deg6 = _extract_symengine_term_details(term6, list(s_vars))
    assert coeff6 == 1.0
    expected_k6 = np.zeros(N_VARS, dtype=np.int64)
    expected_k6[0] = 1
    expected_k6[1] = 1
    np.testing.assert_array_equal(k6, expected_k6)
    assert deg6 == 2

def test_extract_symengine_term_details_complex_coeffs():
    """Test _extract_symengine_term_details with complex coefficients."""
    x0, x1, x2, x3, x4, x5 = s_vars
    
    # Test 1: (2+3j)*x0
    term1 = (2 + 3*se.I) * x0
    coeff1, k1, deg1 = _extract_symengine_term_details(term1, list(s_vars))
    assert coeff1 == complex(2.0, 3.0)
    expected_k1 = np.zeros(N_VARS, dtype=np.int64)
    expected_k1[0] = 1
    np.testing.assert_array_equal(k1, expected_k1)
    assert deg1 == 1

    # Test 2: Pure imaginary coefficient: 4j*x1**2
    term2 = (4*se.I) * x1**2
    coeff2, k2, deg2 = _extract_symengine_term_details(term2, list(s_vars))
    assert coeff2 == complex(0.0, 4.0)
    expected_k2 = np.zeros(N_VARS, dtype=np.int64)
    expected_k2[1] = 2
    np.testing.assert_array_equal(k2, expected_k2)
    assert deg2 == 2

def test_extract_symengine_term_details_error_handling():
    """Test error handling in _extract_symengine_term_details."""
    x0, x1, x2, x3, x4, x5 = s_vars
    y = se.Symbol('y') # A variable not in s_vars

    # Error: Variable in term not in provided variables list
    term_unknown_var = 3 * y
    with pytest.raises(ValueError, match="Variable 'y' in term"):
        _extract_symengine_term_details(term_unknown_var, list(s_vars))

    c = se.Symbol('c')
    term_symbolic_coeff = c * x0
    expected_vars_str = str(list(s_vars))
    match_pattern = (f"Variable '{re.escape(str(c))}' in term '{re.escape(str(term_symbolic_coeff))}' "
                     f"not found in variables list: {re.escape(expected_vars_str)}")
    with pytest.raises(ValueError, match=match_pattern):
        _extract_symengine_term_details(term_symbolic_coeff, list(s_vars))

def test_symengine_to_custom_poly_zero_and_constant():
    """Test symengine2poly with zero and constant expressions."""
    expr_zero = se.Integer(0)
    poly_list_zero = symengine2poly(expr_zero, list(s_vars), MAX_DEGREE, PSI, CLMO)
    assert len(poly_list_zero) == MAX_DEGREE + 1
    for d_poly in poly_list_zero:
        assert np.all(d_poly == 0.0)
    expr_const = se.Integer(7)
    poly_list_const = symengine2poly(expr_const, list(s_vars), MAX_DEGREE, PSI, CLMO)
    assert len(poly_list_const) == MAX_DEGREE + 1
    for i, d_poly in enumerate(poly_list_const):
        if i == 0: # Degree 0
            assert d_poly.shape[0] == PSI[N_VARS, 0]
            if d_poly.shape[0] > 0: # Should be 1 for degree 0
                assert d_poly[0] == 7.0
                assert np.all(d_poly[1:] == 0.0) # if somehow size > 1
            else: # Should not happen for deg 0
                assert d_poly.shape[0] == 1 
        else: # Higher degrees
            assert np.all(d_poly == 0.0)

def test_symengine_to_custom_poly_simple_real():
    """Test symengine2poly with a simple real polynomial."""
    x0, x1, x2, x3, x4, x5 = s_vars
    expr = 1 + 2*x0 + 3*x1**2 - 4*x0*x2**3 # Max degree of this term is 4

    poly_list = symengine2poly(expr, list(s_vars), 4, PSI, CLMO)
    assert len(poly_list) == 4 + 1

    # Degree 0: constant term 1
    idx_const = encode_multiindex(np.array([0]*N_VARS, dtype=np.int64), 0, PSI, CLMO)
    assert poly_list[0][idx_const] == 1.0
    
    # Degree 1: 2*x0
    idx_x0 = encode_multiindex(np.array([1,0,0,0,0,0], dtype=np.int64), 1, PSI, CLMO)
    assert poly_list[1][idx_x0] == 2.0
    
    # Degree 2: 3*x1**2
    idx_x1sq = encode_multiindex(np.array([0,2,0,0,0,0], dtype=np.int64), 2, PSI, CLMO)
    assert poly_list[2][idx_x1sq] == 3.0

    # Degree 3: (no terms)
    assert np.all(poly_list[3] == 0.0)

    # Degree 4: -4*x0*x2**3
    k_x0x2_3 = np.zeros(N_VARS, dtype=np.int64)
    k_x0x2_3[0] = 1
    k_x0x2_3[2] = 3
    idx_x0x2_3 = encode_multiindex(k_x0x2_3, 4, PSI, CLMO)
    assert poly_list[4][idx_x0x2_3] == -4.0

    # Test with max_degree = 2 (truncation)
    poly_list_trunc = symengine2poly(expr, list(s_vars), 2, PSI, CLMO)
    assert len(poly_list_trunc) == 2 + 1
    assert poly_list_trunc[0][idx_const] == 1.0
    assert poly_list_trunc[1][idx_x0] == 2.0
    assert poly_list_trunc[2][idx_x1sq] == 3.0
    # Higher degree terms should not be present.

def test_symengine_to_custom_poly_complex():
    """Test symengine2poly with complex coefficients."""
    x0, x1, x2, x3, x4, x5 = s_vars
    expr = (1+2*se.I) + (3-1*se.I)*x0 + (0.5+0.5*se.I)*x1**2

    poly_list = symengine2poly(expr, list(s_vars), 2, PSI, CLMO, complex_dtype=True)
    assert len(poly_list) == 2 + 1
    assert poly_list[0].dtype == np.complex128

    # Degree 0: (1+2j)
    idx_const = encode_multiindex(np.array([0]*N_VARS, dtype=np.int64), 0, PSI, CLMO)
    assert poly_list[0][idx_const] == complex(1.0, 2.0)

    # Degree 1: (3-1j)*x0
    idx_x0 = encode_multiindex(np.array([1,0,0,0,0,0], dtype=np.int64), 1, PSI, CLMO)
    assert poly_list[1][idx_x0] == complex(3.0, -1.0)

    # Degree 2: (0.5+0.5j)*x1**2
    idx_x1sq = encode_multiindex(np.array([0,2,0,0,0,0], dtype=np.int64), 2, PSI, CLMO)
    assert poly_list[2][idx_x1sq] == complex(0.5, 0.5)

    # Test error if complex_dtype is False but expr has complex parts
    with pytest.raises(ValueError, match="Complex coefficient .* encountered .* but complex_dtype is False"):
        symengine2poly(expr, list(s_vars), 2, PSI, CLMO, complex_dtype=False)
    
    # Test with only real part if imag part is below tolerance
    expr_real_dominant = 1.0 + (2.0 + 1e-20 * se.I) * x0
    poly_list_real_dom = symengine2poly(expr_real_dominant, list(s_vars), 1, PSI, CLMO, complex_dtype=False)
    assert poly_list_real_dom[1][idx_x0] == pytest.approx(2.0)


def test_symengine_to_custom_poly_tolerance():
    """Test coefficient tolerance in symengine2poly."""
    x0, x1, x2, x3, x4, x5 = s_vars
    expr = 1e-20 * x0 + 5 * x1 
    
    # Default tolerance is 1e-18
    poly_list = symengine2poly(expr, list(s_vars), 1, PSI, CLMO)
    idx_x0 = encode_multiindex(np.array([1,0,0,0,0,0], dtype=np.int64), 1, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0,1,0,0,0,0], dtype=np.int64), 1, PSI, CLMO)
    
    assert poly_list[1][idx_x0] == 0.0 # Should be zeroed out
    assert poly_list[1][idx_x1] == 5.0

    # Test with a larger tolerance
    poly_list_larger_tol = symengine2poly(expr, list(s_vars), 1, PSI, CLMO, tolerance=1e-10)
    assert poly_list_larger_tol[1][idx_x0] == 0.0
    assert poly_list_larger_tol[1][idx_x1] == 5.0
    
    expr_just_above_default = 1e-17 * x0
    poly_list_jad = symengine2poly(expr_just_above_default, list(s_vars), 1, PSI, CLMO)
    assert poly_list_jad[1][idx_x0] == 1e-17


def test_symengine_to_custom_poly_error_handling():
    """Test various error conditions for symengine2poly."""
    x0, x1, x2, x3, x4, x5 = s_vars

    # Error: Incorrect number of variables
    wrong_vars = list(s_vars)[:N_VARS-1]
    expr = x0
    with pytest.raises(ValueError, match=f"Expected {N_VARS} variables, got {N_VARS-1}"):
        symengine2poly(expr, wrong_vars, 1, PSI, CLMO)

    # Error: Unresolved symbolic coefficient in expression
    c = se.Symbol('c')
    expr_sym_coeff = c * x0 + x1 # The term c*x0 will be processed by _extract_symengine_term_details.
    # Define the term that will actually cause the error within _extract_symengine_term_details
    term_causing_error = c * x0 
    expected_vars_str = str(list(s_vars))
    
    # The error will originate from _extract_symengine_term_details when processing the term_causing_error (c*x0)
    # because 'c' is not in list(s_vars).
    match_pattern = (f"Variable '{re.escape(str(c))}' in term '{re.escape(str(term_causing_error))}' "
                     f"not found in variables list: {re.escape(expected_vars_str)}")
    with pytest.raises(ValueError, match=match_pattern):
        symengine2poly(expr_sym_coeff, list(s_vars), 1, PSI, CLMO)

    # Test with expression containing variables not in the provided list
    y = se.Symbol('y_unlisted')
    expr_unknown_var = x0 + y
    with pytest.raises(ValueError, match="Variable 'y_unlisted' in term"):
        symengine2poly(expr_unknown_var, list(s_vars), 1, PSI, CLMO)


def test_symengine_to_custom_poly_variable_order():
    """Test that the order of variables matters and is respected."""
    x0, x1 = s_vars[0], s_vars[1] # Assuming N_VARS >= 2
    
    expr = 2*x0 + 3*x1
    
    # Standard order: [x0, x1, ...]
    poly_list_std = symengine2poly(expr, list(s_vars), 1, PSI, CLMO)
    
    k_x0_std = np.zeros(N_VARS, dtype=np.int64); k_x0_std[0] = 1
    idx_x0_std = encode_multiindex(k_x0_std, 1, PSI, CLMO)
    
    k_x1_std = np.zeros(N_VARS, dtype=np.int64); k_x1_std[1] = 1
    idx_x1_std = encode_multiindex(k_x1_std, 1, PSI, CLMO)
    
    assert poly_list_std[1][idx_x0_std] == 2.0
    assert poly_list_std[1][idx_x1_std] == 3.0

    # Swapped order for the first two variables in the `variables` list
    if N_VARS >= 2:
        swapped_s_vars = list(s_vars)
        swapped_s_vars[0], swapped_s_vars[1] = swapped_s_vars[1], swapped_s_vars[0] # now [x1, x0, ...]
        
        poly_list_swapped = symengine2poly(expr, swapped_s_vars, 1, PSI, CLMO)
        
        # In the swapped_s_vars list, x0 is now at index 1, x1 is at index 0.
        # So, for the term 2*x0: k should be [0,1,0,...] referring to swapped_s_vars
        # And for the term 3*x1: k should be [1,0,0,...] referring to swapped_s_vars
        
        # k for term 2*x0 (which is s_vars[0]), but its position in swapped_s_vars is 1
        k_x0_swapped_map = np.zeros(N_VARS, dtype=np.int64); k_x0_swapped_map[1] = 1
        idx_x0_in_swapped = encode_multiindex(k_x0_swapped_map, 1, PSI, CLMO)
        
        # k for term 3*x1 (which is s_vars[1]), but its position in swapped_s_vars is 0
        k_x1_swapped_map = np.zeros(N_VARS, dtype=np.int64); k_x1_swapped_map[0] = 1
        idx_x1_in_swapped = encode_multiindex(k_x1_swapped_map, 1, PSI, CLMO)

        assert poly_list_swapped[1][idx_x0_in_swapped] == 2.0 # Coeff of x0
        assert poly_list_swapped[1][idx_x1_in_swapped] == 3.0 # Coeff of x1


def test_pipeline():
    psi, clmo = init_index_tables(6)
    expr = (3+2j)*q1**2*p1 - 5*q2*p2 + 7      # toy H
    poly = symengine2poly(expr,
                                    [q1,q2,q3,p1,p2,p3],
                                    max_degree=6,
                                    psi=psi, clmo=clmo,
                                    complex_dtype=True)
    assert abs(poly[3][encode_multiindex(np.array([2,0,0,1,0,0]),3,psi,clmo)]-(3+2j)) < 1e-12
    assert abs(poly[2][encode_multiindex(np.array([0,1,0,0,1,0]),2,psi,clmo)]+5) < 1e-12
    assert poly[0][0] == 7
