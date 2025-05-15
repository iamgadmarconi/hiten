import numpy as np
import pytest
from numba.typed import List

from algorithms.center.polynomial.base import init_index_tables, encode_multiindex
from algorithms.center.polynomial.operations import (
    polynomial_zero_list,
    polynomial_variable,
    polynomial_variables_list,
    polynomial_add_inplace,
    polynomial_multiply,
    polynomial_power
)
from algorithms.variables import N_VARS

# Initialize tables for tests
MAX_DEGREE = 5
PSI, CLMO = init_index_tables(MAX_DEGREE)


def test_polynomial_zero_list():
    """Test creation of a list of zero polynomials"""
    # Test for real polynomials
    poly_list = polynomial_zero_list(MAX_DEGREE, PSI)
    
    # Check the length of the list
    assert len(poly_list) == MAX_DEGREE + 1
    
    # Check that each array has the correct size and all zeros
    for d in range(MAX_DEGREE + 1):
        assert poly_list[d].shape[0] == PSI[N_VARS, d]
        assert np.all(poly_list[d] == 0.0)
        assert poly_list[d].dtype == np.float64
    
    # Test for complex polynomials
    complex_poly_list = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    
    for d in range(MAX_DEGREE + 1):
        assert complex_poly_list[d].shape[0] == PSI[N_VARS, d]
        assert np.all(complex_poly_list[d] == 0.0)
        assert complex_poly_list[d].dtype == np.complex128


def test_polynomial_variable():
    """Test creation of a polynomial representing a variable"""
    # Test for each variable
    for var_idx in range(N_VARS):
        poly = polynomial_variable(var_idx, MAX_DEGREE, PSI)
        
        # Check that it's a list of correct length
        assert len(poly) == MAX_DEGREE + 1
        
        # Only degree 1 should have a non-zero entry
        for d in range(MAX_DEGREE + 1):
            if d == 1:
                # The variable should be represented in the degree-1 array
                k = np.zeros(N_VARS, dtype=np.int64)
                k[var_idx] = 1
                idx = encode_multiindex(k, 1, PSI, CLMO)
                assert poly[1][idx] == 1.0
                
                # Make a copy to check that the rest is zero
                tmp = poly[1].copy()
                tmp[idx] = 0.0
                assert np.all(tmp == 0.0)
            else:
                # All other degrees should be zero
                assert np.all(poly[d] == 0.0)


def test_polynomial_variables_list():
    """Test creation of polynomials for all variables"""
    var_polys = polynomial_variables_list(MAX_DEGREE, PSI)
    
    # Should be a list of 6 polynomials
    assert len(var_polys) == N_VARS
    
    # Each element should be a polynomial representing a variable
    for i, poly in enumerate(var_polys):
        # Should be a list of MAX_DEGREE + 1 arrays
        assert len(poly) == MAX_DEGREE + 1
        
        # Check that it represents the correct variable
        k = np.zeros(N_VARS, dtype=np.int64)
        k[i] = 1
        idx = encode_multiindex(k, 1, PSI, CLMO)
        assert poly[1][idx] == 1.0


def test_polynomial_add_inplace():
    """Test in-place addition of polynomial lists"""
    # Create two test polynomial lists
    a = polynomial_zero_list(MAX_DEGREE, PSI)
    b = polynomial_zero_list(MAX_DEGREE, PSI)
    
    # Set some values in both lists
    for d in range(1, MAX_DEGREE + 1):
        for i in range(min(PSI[N_VARS, d], 3)):  # Just set a few values
            a[d][i] = d + i
            b[d][i] = 2 * d + i
    
    # Create a copy of a for testing
    a_copy = [arr.copy() for arr in a]
    
    # Test addition without scaling
    polynomial_add_inplace(a, b)
    
    # Check results
    for d in range(MAX_DEGREE + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] + b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]  # Other values should be unchanged
    
    # Reset a
    a = [arr.copy() for arr in a_copy]
    
    # Test addition with positive scaling
    scale = 2.0
    polynomial_add_inplace(a, b, scale)
    
    # Check results
    for d in range(MAX_DEGREE + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] + scale * b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]
    
    # Reset a
    a = [arr.copy() for arr in a_copy]
    
    # Test addition with negative scaling (subtraction)
    scale = -1.0
    polynomial_add_inplace(a, b, scale)
    
    # Check results
    for d in range(MAX_DEGREE + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] - b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]


def test_polynomial_multiply():
    """Test multiplication of polynomial lists"""
    # Create polynomials for x and y
    x_poly = polynomial_variable(0, MAX_DEGREE, PSI)  # x
    y_poly = polynomial_variable(1, MAX_DEGREE, PSI)  # y
    
    # Multiply x * y
    result = polynomial_multiply(x_poly, y_poly, MAX_DEGREE, PSI, CLMO)
    
    # Check that it's a list of correct length
    assert len(result) == MAX_DEGREE + 1
    
    # The result should be x*y, which is a degree-2 monomial
    # Locate the x*y term in the degree-2 array
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 1  # x
    k[1] = 1  # y
    idx = encode_multiindex(k, 2, PSI, CLMO)
    
    # The coefficient should be 1.0
    assert result[2][idx] == 1.0
    
    # All other terms should be zero
    result[2][idx] = 0.0
    for d in range(MAX_DEGREE + 1):
        assert np.all(result[d] == 0.0)
    
    # Test with truncation
    # Create a high-degree polynomial
    trunc_deg = 3
    x_squared = polynomial_multiply(x_poly, x_poly, MAX_DEGREE, PSI, CLMO)
    y_squared = polynomial_multiply(y_poly, y_poly, MAX_DEGREE, PSI, CLMO)
    
    # x^2 * y^2 would be degree 4, but we'll truncate to degree 3
    truncated = polynomial_multiply(x_squared, y_squared, trunc_deg, PSI, CLMO)
    
    # Should only go up to degree 3
    assert len(truncated) == trunc_deg + 1
    
    # All elements should be zero since the lowest degree term would be x^2*y^2 (degree 4)
    for d in range(trunc_deg + 1):
        assert np.all(truncated[d] == 0.0)
    
    # Test commutativity: x*y = y*x
    xy = polynomial_multiply(x_poly, y_poly, MAX_DEGREE, PSI, CLMO)
    yx = polynomial_multiply(y_poly, x_poly, MAX_DEGREE, PSI, CLMO)
    
    for d in range(MAX_DEGREE + 1):
        np.testing.assert_array_almost_equal(xy[d], yx[d])


def test_polynomial_power():
    """Test power of polynomial lists using binary exponentiation"""
    # Create polynomial for x
    x_poly = polynomial_variable(0, MAX_DEGREE, PSI)  # x
    
    # Test x^0 = 1
    x_pow_0 = polynomial_power(x_poly, 0, MAX_DEGREE, PSI, CLMO)
    
    # Should be the constant polynomial 1
    assert x_pow_0[0][0] == 1.0
    
    # All other terms should be zero
    x_pow_0[0][0] = 0.0
    for d in range(MAX_DEGREE + 1):
        assert np.all(x_pow_0[d] == 0.0)
    
    # Test x^1 = x
    x_pow_1 = polynomial_power(x_poly, 1, MAX_DEGREE, PSI, CLMO)
    
    for d in range(MAX_DEGREE + 1):
        np.testing.assert_array_almost_equal(x_pow_1[d], x_poly[d])
    
    # Test x^2
    x_pow_2 = polynomial_power(x_poly, 2, MAX_DEGREE, PSI, CLMO)
    
    # x^2 should be a degree-2 monomial with coefficient 1.0
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 2
    idx = encode_multiindex(k, 2, PSI, CLMO)
    assert x_pow_2[2][idx] == 1.0
    
    # All other terms should be zero
    x_pow_2[2][idx] = 0.0
    for d in range(MAX_DEGREE + 1):
        assert np.all(x_pow_2[d] == 0.0)
    
    # Test x^4 with truncation to degree 3
    trunc_deg = 3
    x_pow_4_trunc = polynomial_power(x_poly, 4, trunc_deg, PSI, CLMO)
    
    # Should only go up to degree 3
    assert len(x_pow_4_trunc) == trunc_deg + 1
    
    # All terms should be zero since x^4 is degree 4 > trunc_deg
    for d in range(trunc_deg + 1):
        assert np.all(x_pow_4_trunc[d] == 0.0)
    
    # Test binary exponentiation efficiency for large powers
    # (x+y)^8 should be computed efficiently

    # Define a local max degree for this specific test case
    LOCAL_MAX_DEGREE_8 = 8
    PSI_8, CLMO_8 = init_index_tables(LOCAL_MAX_DEGREE_8)

    xy_poly = polynomial_zero_list(LOCAL_MAX_DEGREE_8, PSI_8)

    # Set x+y in degree 1
    k_x = np.zeros(N_VARS, dtype=np.int64)
    k_y = np.zeros(N_VARS, dtype=np.int64)
    k_x[0] = 1
    k_y[1] = 1
    idx_x = encode_multiindex(k_x, 1, PSI_8, CLMO_8)
    idx_y = encode_multiindex(k_y, 1, PSI_8, CLMO_8)
    xy_poly[1][idx_x] = 1.0
    xy_poly[1][idx_y] = 1.0

    # Compute (x+y)^8
    xy_pow_8 = polynomial_power(xy_poly, 8, LOCAL_MAX_DEGREE_8, PSI_8, CLMO_8)

    # The binomial expansion (x+y)^8 = sum(C(8,k) * x^k * y^(8-k))
    # Check a few terms
    # C(8,4) * x^4 * y^4 = 70 * x^4 * y^4
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 4
    k[1] = 4
    idx = encode_multiindex(k, 8, PSI_8, CLMO_8)
    assert xy_pow_8[8][idx] == 70.0

    # C(8,0) * x^0 * y^8 = y^8
    k = np.zeros(N_VARS, dtype=np.int64)
    # k[0] is already 0 from np.zeros
    k[1] = 8
    idx = encode_multiindex(k, 8, PSI_8, CLMO_8)
    assert xy_pow_8[8][idx] == 1.0

    # C(8,8) * x^8 * y^0 = x^8
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 8
    # k[1] is already 0 from np.zeros
    idx = encode_multiindex(k, 8, PSI_8, CLMO_8)
    assert xy_pow_8[8][idx] == 1.0


def test_complex_polynomials():
    """Test operations with complex polynomials"""
    # Create complex polynomial for x + iy
    x_plus_iy = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    
    # Set x+iy in degree 1
    k_x = np.zeros(N_VARS, dtype=np.int64)
    # In the original test, k_x[0] = 1 makes x_plus_iy[1][idx_x] complex(1.0,1.0) refer to x0.
    # Let's make it (1+i)*x0 by setting the coefficient for x0.
    k_x0_var = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0_var, 1, PSI, CLMO)
    x_plus_iy[1][idx_x0] = complex(1.0, 1.0) # x_plus_iy represents (1+i)*x0
    
    # Create another complex polynomial for y - i*z
    y_minus_iz = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    
    # Set y in degree 1
    k_y_var = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_y = encode_multiindex(k_y_var, 1, PSI, CLMO)
    y_minus_iz[1][idx_y] = 1.0 # Coefficient for y is 1.0
    
    # Set -i*z in degree 1
    k_z_var = np.array([0,0,1,0,0,0], dtype=np.int64)
    idx_z = encode_multiindex(k_z_var, 1, PSI, CLMO)
    y_minus_iz[1][idx_z] = complex(0.0, -1.0) # Coefficient for z is -i
    
    # Test addition
    result_add = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    # Make copies for addition to avoid modifying original test polynomials if reused.
    x_plus_iy_copy_for_add = [arr.copy() for arr in x_plus_iy]
    y_minus_iz_copy_for_add = [arr.copy() for arr in y_minus_iz]

    polynomial_add_inplace(result_add, x_plus_iy_copy_for_add)
    polynomial_add_inplace(result_add, y_minus_iz_copy_for_add)
    
    # Check x0 coefficient: should be 1+i
    assert result_add[1][idx_x0] == complex(1.0, 1.0)
    
    # Check y coefficient: should be 1
    assert result_add[1][idx_y] == 1.0
    
    # Check z coefficient: should be -i
    assert result_add[1][idx_z] == complex(0.0, -1.0)
    
    # Test multiplication: ((1+i)x0) * (y - iz)
    product = polynomial_multiply(x_plus_iy, y_minus_iz, MAX_DEGREE, PSI, CLMO)
    
    # (1+i)x0 * y = (1+i)x0y
    k_x0y = np.array([1,1,0,0,0,0], dtype=np.int64)
    idx_prod_x0y = encode_multiindex(k_x0y, 2, PSI, CLMO) # degree is 1+1=2
    assert product[2][idx_prod_x0y] == complex(1.0, 1.0)
    
    # (1+i)x0 * (-i)z = -(1+i)i*x0z = (-i-i^2)*x0z = (-i+1)*x0z = (1-i)x0z
    k_x0z = np.array([1,0,1,0,0,0], dtype=np.int64)
    idx_prod_x0z = encode_multiindex(k_x0z, 2, PSI, CLMO) # degree is 1+1=2
    assert product[2][idx_prod_x0z] == complex(1.0, -1.0)

    # Ensure other terms in degree 2 are zero
    product[2][idx_prod_x0y] = 0.0
    product[2][idx_prod_x0z] = 0.0
    assert np.all(product[2] == 0.0)
    # Ensure other degrees are zero
    for d_idx in range(MAX_DEGREE + 1):
        if d_idx != 2:
            assert np.all(product[d_idx] == 0.0)


def test_polynomial_multiply_complex_inputs():
    """Test multiplication of polynomials with multiple terms e.g. (x0+x1)*(x0-x2)."""
    # P1 = x0 + x1
    p1 = polynomial_zero_list(MAX_DEGREE, PSI)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    k_x1 = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_x0_p1 = encode_multiindex(k_x0, 1, PSI, CLMO)
    idx_x1_p1 = encode_multiindex(k_x1, 1, PSI, CLMO)
    p1[1][idx_x0_p1] = 1.0
    p1[1][idx_x1_p1] = 1.0

    # P2 = x0 - x2
    p2 = polynomial_zero_list(MAX_DEGREE, PSI)
    k_x2 = np.array([0,0,1,0,0,0], dtype=np.int64)
    idx_x0_p2 = encode_multiindex(k_x0, 1, PSI, CLMO) # same x0 as above
    idx_x2_p2 = encode_multiindex(k_x2, 1, PSI, CLMO)
    p2[1][idx_x0_p2] = 1.0
    p2[1][idx_x2_p2] = -1.0

    # Expected P1 * P2 = (x0+x1)*(x0-x2) = x0*x0 - x0*x2 + x1*x0 - x1*x2
    # Resulting polynomial is degree 2.
    result = polynomial_multiply(p1, p2, MAX_DEGREE, PSI, CLMO)

    assert len(result) == MAX_DEGREE + 1

    # Check degree 2 terms
    # x0*x0 (x0^2)
    k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0_sq = encode_multiindex(k_x0_sq, 2, PSI, CLMO)
    assert abs(result[2][idx_x0_sq] - 1.0) < 1e-9

    # -x0*x2
    k_x0x2 = np.array([1,0,1,0,0,0], dtype=np.int64)
    idx_x0x2 = encode_multiindex(k_x0x2, 2, PSI, CLMO)
    assert abs(result[2][idx_x0x2] - (-1.0)) < 1e-9
    
    # +x1*x0
    k_x1x0 = np.array([1,1,0,0,0,0], dtype=np.int64) # Standard order is x0 then x1
    idx_x1x0 = encode_multiindex(k_x1x0, 2, PSI, CLMO)
    assert abs(result[2][idx_x1x0] - 1.0) < 1e-9

    # -x1*x2
    k_x1x2 = np.array([0,1,1,0,0,0], dtype=np.int64)
    idx_x1x2 = encode_multiindex(k_x1x2, 2, PSI, CLMO)
    assert abs(result[2][idx_x1x2] - (-1.0)) < 1e-9
    
    # Zero out checked terms and verify rest of degree 2 is zero
    result[2][idx_x0_sq] = 0.0
    result[2][idx_x0x2] = 0.0
    result[2][idx_x1x0] = 0.0
    result[2][idx_x1x2] = 0.0
    assert np.allclose(result[2], 0.0)

    # Verify other degrees are zero
    for d in range(MAX_DEGREE + 1):
        if d != 2:
            assert np.allclose(result[d], 0.0)


def test_polynomial_multiply_with_zero_components():
    """Test multiplication where one input has zero components for some degrees."""
    # A = x0
    poly_A = polynomial_variable(0, MAX_DEGREE, PSI)

    # B = 2.0 + x1^2
    poly_B = polynomial_zero_list(MAX_DEGREE, PSI)
    poly_B[0][0] = 2.0 # Constant term
    k_x1_sq = np.array([0,1,0,0,0,0], dtype=np.int64) # Error here: x1^2 means k[1]=2
    k_x1_sq_corrected = np.array([0,1,0,0,0,0], dtype=np.int64)
    k_x1_sq_corrected[1] = 2 # Corrected: k for x1^2 is [0,2,0,0,0,0]
    idx_x1_sq = encode_multiindex(k_x1_sq_corrected, 2, PSI, CLMO)
    poly_B[2][idx_x1_sq] = 1.0

    # poly_B[1] is all zeros, which tests the `not np.any(b[d2])` condition

    # Expected result: A * B = x0 * (2.0 + x1^2) = 2.0*x0 + x0*x1^2
    # Output max degree is MAX_DEGREE
    result = polynomial_multiply(poly_A, poly_B, MAX_DEGREE, PSI, CLMO)

    assert len(result) == MAX_DEGREE + 1

    # Check for 2.0*x0 (degree 1)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, PSI, CLMO)
    assert abs(result[1][idx_x0] - 2.0) < 1e-9
    result[1][idx_x0] = 0.0 # Zero out for later check
    assert np.allclose(result[1], 0.0)

    # Check for x0*x1^2 (degree 3)
    k_x0_x1sq = np.array([1,2,0,0,0,0], dtype=np.int64) # Corrected: k for x0*x1^2 is [1,2,0,0,0,0]
    idx_x0_x1sq = encode_multiindex(k_x0_x1sq, 3, PSI, CLMO)
    assert abs(result[3][idx_x0_x1sq] - 1.0) < 1e-9
    result[3][idx_x0_x1sq] = 0.0 # Zero out for later check
    assert np.allclose(result[3], 0.0)
    
    # Check other degrees are zero
    for d in range(MAX_DEGREE + 1):
        if d not in [1, 3]: # Degrees 1 and 3 were checked and zeroed out
            assert np.allclose(result[d], 0.0)


def test_polynomial_power_complex_base():
    """Test polynomial_power with a complex base polynomial."""
    # P = (1+2j) * x0
    base_poly_complex = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, PSI, CLMO)
    base_poly_complex[1][idx_x0] = complex(1.0, 2.0)

    # Test P^2 = ((1+2j)x0)^2 = (1+2j)^2 * x0^2 = (1 + 4j + 4j^2) * x0^2 = (-3 + 4j) * x0^2
    pow2_result = polynomial_power(base_poly_complex, 2, MAX_DEGREE, PSI, CLMO)
    
    assert len(pow2_result) == MAX_DEGREE + 1
    k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0_sq = encode_multiindex(k_x0_sq, 2, PSI, CLMO)
    
    expected_coeff_pow2 = complex(-3.0, 4.0)
    assert np.isclose(pow2_result[2][idx_x0_sq].real, expected_coeff_pow2.real)
    assert np.isclose(pow2_result[2][idx_x0_sq].imag, expected_coeff_pow2.imag)

    # Zero out and check rest of degree 2
    pow2_result[2][idx_x0_sq] = 0j
    assert np.allclose(pow2_result[2], 0j)
    for d in range(MAX_DEGREE + 1):
        if d != 2:
            assert np.allclose(pow2_result[d], 0j)

    # Test P^3 = P^2 * P = ((-3+4j)x0^2) * ((1+2j)x0)
    # = (-3+4j)(1+2j) * x0^3
    # = (-3 -6j +4j +8j^2) * x0^3
    # = (-3 -2j -8) * x0^3 = (-11 -2j) * x0^3
    pow3_result = polynomial_power(base_poly_complex, 3, MAX_DEGREE, PSI, CLMO)
    assert len(pow3_result) == MAX_DEGREE + 1
    k_x0_cb = np.array([3,0,0,0,0,0], dtype=np.int64)
    idx_x0_cb = encode_multiindex(k_x0_cb, 3, PSI, CLMO)

    expected_coeff_pow3 = complex(-11.0, -2.0)
    assert np.isclose(pow3_result[3][idx_x0_cb].real, expected_coeff_pow3.real)
    assert np.isclose(pow3_result[3][idx_x0_cb].imag, expected_coeff_pow3.imag)

    # Zero out and check rest of degree 3
    pow3_result[3][idx_x0_cb] = 0j
    assert np.allclose(pow3_result[3], 0j)
    for d in range(MAX_DEGREE + 1):
        if d != 3:
             assert np.allclose(pow3_result[d], 0j)


def test_polynomial_power_negative_exponent():
    """Test that polynomial_power raises ValueError for k < 0."""
    x_poly = polynomial_variable(0, MAX_DEGREE, PSI)
    with pytest.raises(ValueError, match="Polynomial power k must be non-negative."):
        polynomial_power(x_poly, -1, MAX_DEGREE, PSI, CLMO)

    with pytest.raises(ValueError, match="Polynomial power k must be non-negative."):
        polynomial_power(x_poly, -5, MAX_DEGREE, PSI, CLMO)
