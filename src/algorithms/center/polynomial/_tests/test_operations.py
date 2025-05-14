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
    k_x[0] = 1
    idx_x = encode_multiindex(k_x, 1, PSI, CLMO)
    x_plus_iy[1][idx_x] = complex(1.0, 1.0)  # 1 + i
    
    # Create another complex polynomial
    y_minus_iz = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    
    # Set y-iz in degree 1
    k_y = np.zeros(N_VARS, dtype=np.int64)
    k_z = np.zeros(N_VARS, dtype=np.int64)
    k_y[1] = 1
    k_z[2] = 1
    idx_y = encode_multiindex(k_y, 1, PSI, CLMO)
    idx_z = encode_multiindex(k_z, 1, PSI, CLMO)
    y_minus_iz[1][idx_y] = 1.0
    y_minus_iz[1][idx_z] = complex(0.0, -1.0)  # -i
    
    # Test addition
    result = polynomial_zero_list(MAX_DEGREE, PSI, complex_dtype=True)
    polynomial_add_inplace(result, x_plus_iy)
    polynomial_add_inplace(result, y_minus_iz)
    
    # Check x coefficient: should be 1+i
    assert result[1][idx_x] == complex(1.0, 1.0)
    
    # Check y coefficient: should be 1
    assert result[1][idx_y] == 1.0
    
    # Check z coefficient: should be -i
    assert result[1][idx_z] == complex(0.0, -1.0)
    
    # Test multiplication
    product = polynomial_multiply(x_plus_iy, y_minus_iz, MAX_DEGREE, PSI, CLMO)
    
    # (1+i)x * y = (1+i)xy
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 1
    k[1] = 1
    idx = encode_multiindex(k, 2, PSI, CLMO)
    assert product[2][idx] == complex(1.0, 1.0)
    
    # (1+i)x * (-i)z = -(1+i)ixz = -(i-1)xz = (1-i)xz
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 1
    k[2] = 1
    idx = encode_multiindex(k, 2, PSI, CLMO)
    assert product[2][idx] == complex(1.0, -1.0)
