import numpy as np
import pytest
from numba.typed import List

from algorithms.variables import N_VARS
from algorithms.center.polynomial.base import init_index_tables, encode_multiindex
from algorithms.center.polynomial.operations import (
    polynomial_zero_list, 
    polynomial_add_inplace, 
    polynomial_multiply, 
    polynomial_power, 
    polynomial_poisson_bracket,
    polynomial_clean,
    polynomial_differentiate as op_differentiate # Alias to avoid clash
)
# Import JITPolynomial and its factory functions
from algorithms.center.polynomial.polynomial import (
    JITPolynomial, 
    create_zero_jitpolynomial, 
    create_variable_jitpolynomial
)

# Helper to compare JITPolynomial instances
def assert_jit_polynomials_equal(p1: JITPolynomial, p2: JITPolynomial, msg=""):
    assert p1.max_deg == p2.max_deg, f"Max degrees differ: {p1.max_deg} vs {p2.max_deg}. {msg}"
    # Compare psi_tables (shape and content)
    assert p1.psi_table.shape == p2.psi_table.shape, f"Psi table shapes differ. {msg}"
    np.testing.assert_array_equal(p1.psi_table, p2.psi_table, err_msg=f"Psi tables differ. {msg}")
    
    # Compare clmo_tables (list length and content of each array)
    assert len(p1.clmo_table) == len(p2.clmo_table), f"Clmo table list lengths differ. {msg}"
    for i in range(len(p1.clmo_table)):
        np.testing.assert_array_equal(p1.clmo_table[i], p2.clmo_table[i], 
                                     err_msg=f"Clmo table array at index {i} differs. {msg}")

    # Compare coefficient lists
    assert len(p1.polynomials) == len(p2.polynomials), f"Coefficient list lengths differ. {msg}"
    for d in range(len(p1.polynomials)):
        np.testing.assert_array_almost_equal(p1.polynomials[d], p2.polynomials[d],
                                             err_msg=f"Coefficients at degree {d} differ. {msg}")

# Helper to compare a JITPolynomial with a raw coefficient list
def assert_jit_poly_vs_coeffs_equal(jit_poly: JITPolynomial, expected_coeffs: List[np.ndarray], expected_max_deg: int, msg=""):
    assert jit_poly.max_deg == expected_max_deg, f"JITPoly max_deg mismatch. Expected {expected_max_deg}, got {jit_poly.max_deg}. {msg}"
    assert len(jit_poly.polynomials) == len(expected_coeffs), f"JITPoly coeff list len mismatch. {msg}"
    for d in range(len(expected_coeffs)):
        np.testing.assert_array_almost_equal(jit_poly.polynomials[d], expected_coeffs[d],
                                             err_msg=f"JITPoly coeffs at degree {d} differ. {msg}")


# --- Global test setup ---
MAX_TEST_DEG = 10 # Keep tests manageable

# --- Test Cases ---

def test_create_zero_jitpolynomial():
    """Test factory function for zero JITPolynomial."""
    p_zero = create_zero_jitpolynomial(MAX_TEST_DEG)
    assert p_zero.max_deg == MAX_TEST_DEG
    assert len(p_zero.polynomials) == MAX_TEST_DEG + 1
    for d_coeffs in p_zero.polynomials:
        assert np.all(d_coeffs == 0.0)
    
    # Check table consistency (basic check)
    psi_expected, _ = init_index_tables(MAX_TEST_DEG)
    np.testing.assert_array_equal(p_zero.psi_table, psi_expected)
    assert len(p_zero.clmo_table) == MAX_TEST_DEG + 1

def test_create_variable_jitpolynomial():
    """Test factory function for variable JITPolynomial."""
    for var_idx in range(N_VARS):
        p_var = create_variable_jitpolynomial(var_idx, MAX_TEST_DEG)
        assert p_var.max_deg == MAX_TEST_DEG
        
        # Check degree 1 term
        k = np.zeros(N_VARS, dtype=np.int64)
        k[var_idx] = 1
        idx = encode_multiindex(k, 1, p_var.psi_table, p_var.clmo_table)
        if idx != -1 and 1 < len(p_var.polynomials) and idx < p_var.polynomials[1].size:
            assert p_var.polynomials[1][idx] == 1.0
            # Check other terms in degree 1 are zero
            temp_deg1 = p_var.polynomials[1].copy()
            temp_deg1[idx] = 0.0
            assert np.all(temp_deg1 == 0.0)
        
        # Check other degrees are zero
        for d in range(MAX_TEST_DEG + 1):
            if d != 1:
                assert np.all(p_var.polynomials[d] == 0.0)

def test_jitpolynomial_copy():
    """Test the copy() method of JITPolynomial."""
    p1 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0
    p1.polynomials[0][0] = 5.0 # Add a constant term for diversity
    
    p2 = p1.copy()

    # Assert they are equal in value but not the same object
    assert_jit_polynomials_equal(p1, p2, msg="Copy content check")
    assert p1 is not p2, "Copy should be a new object (polynomials)"
    assert p1.polynomials is not p2.polynomials, "Copied list of arrays should be new"
    assert p1.psi_table is not p2.psi_table, "Copied psi_table should be new"
    assert p1.clmo_table is not p2.clmo_table, "Copied clmo_table list should be new"
    if len(p1.polynomials) > 0 and len(p2.polynomials) > 0:
      assert p1.polynomials[0] is not p2.polynomials[0], "Arrays within copied list should be new"
    if len(p1.clmo_table) > 0 and len(p2.clmo_table) > 0:
      assert p1.clmo_table[0] is not p2.clmo_table[0], "Arrays within copied clmo list should be new"


def test_jitpolynomial_degree_property():
    """Test the degree property."""
    p_zero = create_zero_jitpolynomial(MAX_TEST_DEG)
    assert p_zero.degree == -1

    p_const = create_zero_jitpolynomial(MAX_TEST_DEG)
    if p_const.polynomials[0].size > 0: p_const.polynomials[0][0] = 5.0
    assert p_const.degree == 0

    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0
    assert p_x0.degree == 1
    
    p_x0_sq = p_x0 * p_x0 # x0^2
    assert p_x0_sq.degree == 2
    
    # A polynomial that is zero up to its max_deg
    p_zero_high_max_deg = create_zero_jitpolynomial(MAX_TEST_DEG + 2)
    assert p_zero_high_max_deg.degree == -1


def test_jitpolynomial_add_sub():
    """Test JITPolynomial addition and subtraction."""
    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0
    p_x1 = create_variable_jitpolynomial(1, MAX_TEST_DEG) # x1
    p_const5 = create_zero_jitpolynomial(MAX_TEST_DEG)
    if p_const5.polynomials[0].size > 0: p_const5.polynomials[0][0] = 5.0

    # Addition: x0 + x1
    p_sum = p_x0 + p_x1
    assert p_sum.max_deg == MAX_TEST_DEG # Max_deg should be max of operands
    psi_expected, clmo_expected = init_index_tables(MAX_TEST_DEG)
    expected_coeffs_sum = polynomial_zero_list(MAX_TEST_DEG, psi_expected)
    polynomial_add_inplace(expected_coeffs_sum, p_x0.polynomials)
    polynomial_add_inplace(expected_coeffs_sum, p_x1.polynomials)
    assert_jit_poly_vs_coeffs_equal(p_sum, expected_coeffs_sum, MAX_TEST_DEG, "x0 + x1")

    # Subtraction: (x0+x1) - x1 = x0
    p_diff = p_sum - p_x1
    assert_jit_poly_vs_coeffs_equal(p_diff, p_x0.polynomials, MAX_TEST_DEG, "(x0+x1) - x1")

    # Add constant: x0 + 5
    p_sum_const = p_x0 + p_const5
    expected_coeffs_sum_const = polynomial_zero_list(MAX_TEST_DEG, psi_expected)
    polynomial_add_inplace(expected_coeffs_sum_const, p_x0.polynomials)
    polynomial_add_inplace(expected_coeffs_sum_const, p_const5.polynomials)
    assert_jit_poly_vs_coeffs_equal(p_sum_const, expected_coeffs_sum_const, MAX_TEST_DEG, "x0 + 5")


def test_jitpolynomial_mul():
    """Test JITPolynomial multiplication."""
    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0, max_deg=3
    p_x1 = create_variable_jitpolynomial(1, MAX_TEST_DEG) # x1, max_deg=3

    # Product: x0 * x1 = x0x1
    p_prod = p_x0 * p_x1
    
    # Expected max_deg for product is sum of operands' max_deg for tables,
    # but the internal list max_deg for polynomial_multiply will be this new table max_deg.
    expected_prod_max_deg = p_x0.max_deg + p_x1.max_deg
    assert p_prod.max_deg == expected_prod_max_deg
    
    psi_prod, clmo_prod = init_index_tables(expected_prod_max_deg)
    expected_coeffs_prod = polynomial_multiply(p_x0.polynomials, p_x1.polynomials, expected_prod_max_deg, psi_prod, clmo_prod)
    assert_jit_poly_vs_coeffs_equal(p_prod, expected_coeffs_prod, expected_prod_max_deg, "x0 * x1")
    assert p_prod.degree == 2 # Effective degree of x0*x1 is 2

    # Multiply by zero
    p_zero = create_zero_jitpolynomial(MAX_TEST_DEG)
    p_prod_zero = p_x0 * p_zero
    expected_zero_max_deg = p_x0.max_deg + p_zero.max_deg
    psi_zero_prod, clmo_zero_prod = init_index_tables(expected_zero_max_deg)
    expected_coeffs_zero = polynomial_multiply(p_x0.polynomials, p_zero.polynomials, expected_zero_max_deg, psi_zero_prod, clmo_zero_prod)
    assert_jit_poly_vs_coeffs_equal(p_prod_zero, expected_coeffs_zero, expected_zero_max_deg, "x0 * 0")
    assert p_prod_zero.degree == -1


def test_jitpolynomial_pow():
    """Test JITPolynomial power."""
    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0

    # x0^2
    exponent = 2
    p_x0_sq = p_x0 ** exponent
    
    # The max_deg of the result P^k should be P.max_deg * k, unless P.max_deg is 0.
    expected_res_max_deg_sq = p_x0.max_deg * exponent if p_x0.max_deg > 0 else 0
    assert p_x0_sq.max_deg == expected_res_max_deg_sq
    
    # For comparison, expected_coeffs_sq needs tables based on expected_res_max_deg_sq
    psi_sq, clmo_sq = init_index_tables(expected_res_max_deg_sq)
    expected_coeffs_sq = polynomial_power(p_x0.polynomials, exponent, expected_res_max_deg_sq, psi_sq, clmo_sq)
    assert_jit_poly_vs_coeffs_equal(p_x0_sq, expected_coeffs_sq, expected_res_max_deg_sq, "x0^2")
    assert p_x0_sq.degree == 2

    # x0^0 = 1
    exponent_zero = 0
    p_x0_p0 = p_x0 ** exponent_zero
    
    # Result of P^0 is a constant polynomial (degree 0) with max_deg 0.
    expected_res_max_deg_p0 = 0 # Max degree of a constant is 0
    assert p_x0_p0.max_deg == expected_res_max_deg_p0
    
    psi_p0, clmo_p0 = init_index_tables(expected_res_max_deg_p0)
    expected_coeffs_p0 = polynomial_power(p_x0.polynomials, exponent_zero, expected_res_max_deg_p0, psi_p0, clmo_p0)
    assert_jit_poly_vs_coeffs_equal(p_x0_p0, expected_coeffs_p0, expected_res_max_deg_p0, "x0^0")
    assert p_x0_p0.degree == 0
    if p_x0_p0.polynomials[0].size > 0:
        assert np.isclose(p_x0_p0.polynomials[0][0], 1.0)


def test_jitpolynomial_scale():
    """Test JITPolynomial scaling."""
    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0
    
    factor = complex(2.5, -1.5)
    p_scaled = p_x0.scale(factor)

    assert p_scaled.max_deg == p_x0.max_deg # Scaling doesn't change max_deg
    # Manually scale for comparison
    expected_coeffs_scaled = polynomial_zero_list(p_x0.max_deg, p_x0.psi_table)
    polynomial_add_inplace(expected_coeffs_scaled, p_x0.polynomials, scale=factor) # Use add_inplace with scale
    
    assert_jit_poly_vs_coeffs_equal(p_scaled, expected_coeffs_scaled, p_x0.max_deg, "p_x0.scale()")

def test_jitpolynomial_poisson_bracket():
    """Test JITPolynomial Poisson bracket {x0, p0} = 1."""
    # Use a local max_deg for this test to manage result degree
    local_max_deg = 2
    p_x0 = create_variable_jitpolynomial(0, local_max_deg) # x0 (var index 0)
    p_p0 = create_variable_jitpolynomial(3, local_max_deg) # p0 (var index 3)

    p_pb = p_x0.poisson_bracket(p_p0)
    
    # Expected result max_deg for {A,B} is max(deg_A, deg_B, deg_A+deg_B-2)
    # Here, deg_A=1, deg_B=1. So, max(1,1, 1+1-2=0) = 1.
    # The JITPoly instance table max_deg is based on max of operands' max_deg or theoretic_res_deg
    expected_pb_max_deg = max(p_x0.max_deg, p_p0.max_deg, p_x0.degree + p_p0.degree - 2)
    if expected_pb_max_deg <0: expected_pb_max_deg = 0

    assert p_pb.max_deg == expected_pb_max_deg 

    psi_pb, clmo_pb = init_index_tables(expected_pb_max_deg)
    expected_coeffs_pb = polynomial_poisson_bracket(p_x0.polynomials, p_p0.polynomials, expected_pb_max_deg, psi_pb, clmo_pb)
    
    assert_jit_poly_vs_coeffs_equal(p_pb, expected_coeffs_pb, expected_pb_max_deg, "{x0, p0}")
    assert p_pb.degree == 0 # {x0,p0} = 1
    if p_pb.polynomials[0].size > 0:
        assert np.isclose(p_pb.polynomials[0][0], 1.0)


def test_jitpolynomial_differentiate():
    """Test JITPolynomial differentiation."""
    # P = x0^2
    p_x0 = create_variable_jitpolynomial(0, MAX_TEST_DEG)
    p_x0_sq = p_x0 * p_x0 # x0^2, max_deg = p_x0.max_deg * 2 (from mul logic)

    # d(x0^2)/dx0 = 2x0
    p_deriv_x0 = p_x0_sq.differentiate(0) # Differentiate w.r.t x0
    
    expected_deriv_max_deg = p_x0_sq.max_deg - 1
    if expected_deriv_max_deg < 0: expected_deriv_max_deg = 0
    assert p_deriv_x0.max_deg == expected_deriv_max_deg

    # Manually compute expected coeffs using op_differentiate
    psi_deriv, clmo_deriv = init_index_tables(expected_deriv_max_deg)
    expected_coeffs_list, _ = op_differentiate(
        p_x0_sq.polynomials, 0, p_x0_sq.max_deg, p_x0_sq.psi_table, p_x0_sq.clmo_table,
        psi_deriv, clmo_deriv
    )
    assert_jit_poly_vs_coeffs_equal(p_deriv_x0, expected_coeffs_list, expected_deriv_max_deg, "d(x0^2)/dx0")
    assert p_deriv_x0.degree == 1

    # d(x0^2)/dx1 = 0
    p_deriv_x1 = p_x0_sq.differentiate(1) # Differentiate w.r.t x1
    assert p_deriv_x1.max_deg == expected_deriv_max_deg # Max deg logic is the same
    
    psi_deriv_x1, clmo_deriv_x1 = init_index_tables(expected_deriv_max_deg)
    expected_coeffs_zero, _ = op_differentiate(
        p_x0_sq.polynomials, 1, p_x0_sq.max_deg, p_x0_sq.psi_table, p_x0_sq.clmo_table,
        psi_deriv_x1, clmo_deriv_x1
    )
    assert_jit_poly_vs_coeffs_equal(p_deriv_x1, expected_coeffs_zero, expected_deriv_max_deg, "d(x0^2)/dx1")
    assert p_deriv_x1.degree == -1 # Should be zero polynomial


def test_jitpolynomial_clean():
    """Test JITPolynomial cleaning."""
    p_noisy = create_variable_jitpolynomial(0, MAX_TEST_DEG) # x0
    # Add some noise
    if p_noisy.polynomials[0].size > 0: p_noisy.polynomials[0][0] = 1e-12
    if len(p_noisy.polynomials) > 1 and p_noisy.polynomials[1].size > 0:
        p_noisy.polynomials[1][0] += 1e-14 # Noise on existing x0 term
    
    tol = 1e-10
    p_cleaned = p_noisy.clean(tol)

    assert p_cleaned.max_deg == p_noisy.max_deg
    
    expected_coeffs_cleaned = polynomial_clean(p_noisy.polynomials, tol)
    assert_jit_poly_vs_coeffs_equal(p_cleaned, expected_coeffs_cleaned, p_noisy.max_deg, "clean()")

    # Check that the noisy constant term was removed
    if p_cleaned.polynomials[0].size > 0:
        assert np.isclose(p_cleaned.polynomials[0][0], 0.0)
    # Check that noise on x0 term was handled (it might still be non-zero if original x0 coeff was large enough)
    # This requires comparing with manually cleaned version, which is what assert_jit_poly_vs_coeffs_equal does.
