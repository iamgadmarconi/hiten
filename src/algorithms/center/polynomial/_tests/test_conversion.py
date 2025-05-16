import numpy as np
import pytest
import sympy as sp
from numba.typed import List

from algorithms.center.polynomial.base import (CLMO_GLOBAL, PSI_GLOBAL,
                                               encode_multiindex, make_poly)
from algorithms.center.polynomial.conversion import poly2sympy, sympy2poly
from algorithms.variables import N_VARS

# Test Sympy variables (consistent with N_VARS)
s_vars = list(sp.symbols(f'x_0:{N_VARS}'))
MAX_DEGREE_FOR_PSI = PSI_GLOBAL.shape[1] - 1


# Helper function to create a List[np.ndarray] for poly2sympy input
def create_custom_poly_list(max_deg: int, terms: dict) -> List[np.ndarray]:
    """
    Helper to create a numba.typed.List[np.ndarray] for poly2sympy input.
    'terms' is a dict like: { (k0,k1,...k_N_VARS-1): coefficient_value, ... }
    Each k_i is the exponent for s_vars[i].
    """
    # Initialize Python list of numpy arrays
    py_list_of_coeffs = [make_poly(d, PSI_GLOBAL) for d in range(max_deg + 1)]

    for k_tuple, coeff_val in terms.items():
        if len(k_tuple) != N_VARS:
            raise ValueError(f"Exponent tuple {k_tuple} must have length {N_VARS}")
        
        k_np = np.array(k_tuple, dtype=np.int64)
        deg = int(sum(k_np))

        if deg > max_deg:
            # This term's degree exceeds the allocated list size, skip or error
            # For this helper, we'll assume terms match max_deg allocation
            print(f"Warning: Term {k_tuple} degree {deg} exceeds max_deg {max_deg} of list.")
            continue
        
        # Ensure this degree's array exists (it should due to pre-allocation)
        if deg >= len(py_list_of_coeffs):
             raise IndexError(f"Degree {deg} out of bounds for allocated list (max_deg {max_deg})")

        pos = encode_multiindex(k_np, deg, PSI_GLOBAL, CLMO_GLOBAL)
        if pos != -1:
            # Ensure pos is within bounds for the specific degree's coefficient array
            if pos >= py_list_of_coeffs[deg].shape[0]:
                raise IndexError(f"Encoded position {pos} is out of bounds for degree {deg} (size {py_list_of_coeffs[deg].shape[0]}).")
            py_list_of_coeffs[deg][pos] = complex(coeff_val)
        else:
            raise ValueError(f"Could not encode multi-index {k_tuple} for degree {deg}. Check if degree is too high for CLMO table for that specific degree array, or if exponents are too large for packing.")
    
    # Convert to numba.typed.List
    numba_list = List()
    for arr in py_list_of_coeffs:
        numba_list.append(arr)
    return numba_list

# Helper to compare two poly_lists (List[np.ndarray])
def compare_poly_lists(list1: List[np.ndarray], list2: List[np.ndarray], tol=1e-12) -> bool:
    if len(list1) != len(list2):
        print(f"Length mismatch: {len(list1)} vs {len(list2)}")
        return False
    for i in range(len(list1)):
        arr1, arr2 = list1[i], list2[i]
        if not np.allclose(arr1, arr2, atol=tol, rtol=tol):
            print(f"Data mismatch at degree {i}:\nArr1: {arr1}\nArr2: {arr2}")
            return False
    return True

# --- Test Cases ---

def test_poly2sympy_zero():
    poly_list_zero = create_custom_poly_list(0, {}) # Degree 0, no terms
    expr = poly2sympy(poly_list_zero, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr == sp.Integer(0)

    poly_list_zero_deg2 = create_custom_poly_list(2, {}) # Up to degree 2, all zero
    expr2 = poly2sympy(poly_list_zero_deg2, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr2 == sp.Integer(0)
    
    empty_numba_list = List() 
    # for numpy array in Python list: empty_numba_list.append(array) # No, this is how you append an existing array.
    # An empty typed list:
    expr_empty = poly2sympy(empty_numba_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr_empty == sp.Integer(0)


def test_sympy2poly_zero():
    expr = sp.Integer(0)
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    
    expected_coeffs = make_poly(0, PSI_GLOBAL) # array for degree 0, all zero
    expected_list = List()
    expected_list.append(expected_coeffs)
    
    assert compare_poly_lists(poly_list, expected_list)

def test_poly2sympy_constant():
    const_val = 5.5
    poly_list = create_custom_poly_list(0, {(0,)*N_VARS: const_val})
    expr = poly2sympy(poly_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert expr == sp.Float(const_val)

def test_sympy2poly_constant():
    const_val = 7.0
    expr = sp.Float(const_val)
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    
    expected_coeffs = make_poly(0, PSI_GLOBAL)
    expected_coeffs[encode_multiindex(np.zeros(N_VARS, dtype=np.int64), 0, PSI_GLOBAL, CLMO_GLOBAL)] = const_val
    expected_list = List()
    expected_list.append(expected_coeffs)
    
    assert compare_poly_lists(poly_list, expected_list)


def test_poly2sympy_single_variable():
    # Test for s_vars[1] (x_1)
    target_var_idx = 1
    k_tuple = [0]*N_VARS
    k_tuple[target_var_idx] = 1
    k_tuple = tuple(k_tuple)

    poly_list = create_custom_poly_list(1, {k_tuple: 1.0})
    expr = poly2sympy(poly_list, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert sp.simplify(expr - s_vars[target_var_idx]) == 0

def test_sympy2poly_single_variable():
    target_var_idx = 2 # s_vars[2] (x_2)
    expr = s_vars[target_var_idx]
    poly_list = sympy2poly(expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL)

    k_tuple = [0]*N_VARS
    k_tuple[target_var_idx] = 1
    k_tuple = tuple(k_tuple)
    
    expected_list = create_custom_poly_list(1, {k_tuple: 1.0})
    assert compare_poly_lists(poly_list, expected_list)


def test_round_trip_simple_expression():
    # P(x) = 2.0*x_0 + 3.0*x_1^2
    expr_original = 2.0 * s_vars[0] + 3.0 * s_vars[1]**2
    
    # Sympy -> Custom Poly
    poly_list_intermediate = sympy2poly(expr_original, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    
    # Custom Poly -> Sympy
    expr_reconstructed = poly2sympy(poly_list_intermediate, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    
    assert sp.simplify(expr_original - expr_reconstructed) == 0

    # Custom Poly (from previous step) -> Sympy -> Custom Poly
    poly_list_final = sympy2poly(expr_reconstructed, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert compare_poly_lists(poly_list_intermediate, poly_list_final)


def test_round_trip_complex_coeffs():
    # P(x) = (1+2j)*x_0*x_1 + (3-1j)*x_2^3
    expr_original = (1+2j)*s_vars[0]*s_vars[1] + (3-1j)*s_vars[2]**3
    
    poly_list_intermediate = sympy2poly(expr_original, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    expr_reconstructed = poly2sympy(poly_list_intermediate, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    poly_list_final = sympy2poly(expr_reconstructed, s_vars, PSI_GLOBAL, CLMO_GLOBAL)

    assert sp.simplify(expr_original - expr_reconstructed) == 0
    assert compare_poly_lists(poly_list_intermediate, poly_list_final)

def test_heterogeneous_polynomial_conversion():
    # P(x) = 1.5 + 2.0*x_0 - 0.5*x_1*x_2 + 3.0*x_3^3
    # Exponents for N_VARS=6:
    # (0,0,0,0,0,0): 1.5    (Constant)
    # (1,0,0,0,0,0): 2.0    (x_0)
    # (0,1,1,0,0,0): -0.5   (x_1*x_2)
    # (0,0,0,3,0,0): 3.0    (x_3^3)
    
    terms_dict = {
        (0,0,0,0,0,0): 1.5,
        (1,0,0,0,0,0): 2.0,
        (0,1,1,0,0,0): -0.5,
        (0,0,0,3,0,0): 3.0
    }
    max_deg_poly = 3
    
    # Create custom poly list manually
    custom_poly = create_custom_poly_list(max_deg_poly, terms_dict)
    
    # Expected Sympy expression
    expected_sympy_expr = (
        sp.Float(1.5) + 
        2.0 * s_vars[0] - 
        0.5 * s_vars[1] * s_vars[2] + 
        3.0 * s_vars[3]**3
    )
    
    # Test poly2sympy
    generated_sympy_expr = poly2sympy(custom_poly, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert sp.simplify(generated_sympy_expr - expected_sympy_expr) == 0
    
    # Test sympy2poly using the expected Sympy expression
    generated_custom_poly = sympy2poly(expected_sympy_expr, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    assert compare_poly_lists(custom_poly, generated_custom_poly)


def test_poly2sympy_vars_list_length_mismatch():
    poly_list_dummy = create_custom_poly_list(0, {}) # Empty degree 0 poly
    wrong_vars = s_vars[:-1] # N_VARS-1 symbols
    
    with pytest.raises(ValueError, match=f"Expected {N_VARS} symbols"):
        poly2sympy(poly_list_dummy, wrong_vars, PSI_GLOBAL, CLMO_GLOBAL)

def test_sympy2poly_vars_list_length_mismatch():
    expr_dummy = s_vars[0]
    wrong_vars = s_vars + [sp.Symbol('extra_var')] # N_VARS+1 symbols
    
    with pytest.raises(ValueError, match=f"Expected {N_VARS} symbols"):
        sympy2poly(expr_dummy, wrong_vars, PSI_GLOBAL, CLMO_GLOBAL)

def test_sympy2poly_degree_exceeds_psi_limit():
    # PSI_GLOBAL default max_degree is 30. Test with degree 31.
    # Make sure the test is meaningful based on actual MAX_DEGREE_FOR_PSI
    if MAX_DEGREE_FOR_PSI < 35: # Arbitrary check to ensure MAX_DEGREE_FOR_PSI is around 30
        expr_high_degree = s_vars[0]**(MAX_DEGREE_FOR_PSI + 1)
        with pytest.raises(ValueError, match="Expression degree .* exceeds precomputed table limit"):
            sympy2poly(expr_high_degree, s_vars, PSI_GLOBAL, CLMO_GLOBAL)
    else:
        pytest.skip("PSI_GLOBAL max_degree too high to reliably test exceeding limit.")


def test_sympy2poly_non_polynomial_expression():
    expr_non_poly = sp.sin(s_vars[0])
    with pytest.raises(TypeError, match="Could not convert expr to Sympy Poly object"):
        sympy2poly(expr_non_poly, s_vars, PSI_GLOBAL, CLMO_GLOBAL)

def test_sympy2poly_non_numeric_coefficients():
    a = sp.Symbol('a') # Symbolic coefficient
    expr_symbolic_coeff = a * s_vars[0]
    with pytest.raises(TypeError, match="could not be converted to a Python numeric type"):
        sympy2poly(expr_symbolic_coeff, s_vars, PSI_GLOBAL, CLMO_GLOBAL)


def test_round_trip():
    q1, q2, q3, p1, p2, p3 = sp.symbols('q1 q2 q3 p1 p2 p3')
    expr = q1**2 + q2**2 + q3**2 + p1**2 + p2**2 + p3**2 + p1*p2*p3 - 1j*p1
    poly_list = sympy2poly(expr, [q1, q2, q3, p1, p2, p3], PSI_GLOBAL, CLMO_GLOBAL)
    expr_reconstructed = poly2sympy(poly_list, [q1, q2, q3, p1, p2, p3], PSI_GLOBAL, CLMO_GLOBAL)
    assert sp.simplify(expr - expr_reconstructed) == 0
