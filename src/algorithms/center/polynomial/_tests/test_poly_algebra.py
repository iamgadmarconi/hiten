import numpy as np
import pytest
import symengine as se

from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex
from algorithms.center.polynomial.algebra import poly_add, poly_scale, poly_mul, differentiate, poisson
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


def test_poly_add():
    """Test polynomial addition"""
    # Test for various degrees
    for degree in range(MAX_DEGREE + 1):
        # Create test polynomials
        a = make_poly(degree, PSI)
        b = make_poly(degree, PSI)
        result = make_poly(degree, PSI)
        
        # Skip degree 0 as it's trivial
        if degree == 0:
            a[0] = 1.5
            b[0] = 2.5
            poly_add(a, b, result)
            assert result[0] == 4.0
            continue
        
        size = PSI[N_VARS, degree]
        
        # Test addition with specific values
        # Set some coefficients in polynomials a and b
        for i in range(min(size, 10)):
            a[i] = i * 1.5
            b[i] = i * 0.5
        
        # Test zero addition
        zero_poly = make_poly(degree, PSI)
        result_zero = make_poly(degree, PSI)
        poly_add(a, zero_poly, result_zero)
        np.testing.assert_array_equal(a, result_zero)
        
        # Test regular addition
        expected = np.zeros_like(a)
        for i in range(a.shape[0]):
            expected[i] = a[i] + b[i]
        
        poly_add(a, b, result)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test commutativity: a + b = b + a
        result_commute = make_poly(degree, PSI)
        poly_add(b, a, result_commute)
        np.testing.assert_array_almost_equal(result, result_commute)
        
        # Test adding a polynomial to itself
        double_a = make_poly(degree, PSI)
        poly_add(a, a, double_a)
        
        # Should be equivalent to scaling by 2
        expected_double = make_poly(degree, PSI)
        poly_scale(a, 2.0, expected_double)
        np.testing.assert_array_almost_equal(double_a, expected_double)

def test_add_negative_numbers():
    """Test polynomial addition with negative numbers"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    a = make_poly(degree, PSI)
    b = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set alternating positive and negative coefficients
    for i in range(size):
        a[i] = (-1)**i * (i + 1.0)  # 1, -2, 3, -4, ...
        b[i] = (-1)**(i+1) * (i + 2.0)  # -2, 3, -4, 5, ...
    
    # Expected: coefficients should cancel out to (-1, 1, -1, 1, ...)
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)

def test_add_large_values():
    """Test polynomial addition with very large coefficients"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    a = make_poly(degree, PSI)
    b = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set large coefficients
    for i in range(size):
        a[i] = 1e15 + i
        b[i] = 2e15 - i
    
    # Expected result
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)

def test_add_small_values():
    """Test polynomial addition with very small coefficients"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    a = make_poly(degree, PSI)
    b = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set small coefficients
    for i in range(size):
        a[i] = 1e-15 * (i + 1)
        b[i] = 2e-15 * (i + 1)
    
    # Expected result
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)

def test_add_mixed_values():
    """Test polynomial addition with mixed signs and magnitudes"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    a = make_poly(degree, PSI)
    b = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set coefficients with varying magnitudes and signs
    for i in range(size):
        if i % 3 == 0:
            a[i] = 1e10 * (i + 1)
            b[i] = -1e10 * i
        elif i % 3 == 1:
            a[i] = -1e-10 * (i + 1)
            b[i] = 1e-10 * i
        else:
            a[i] = (-1)**i * i
            b[i] = (-1)**(i+1) * (i + 1)
    
    # Expected result
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)

def test_add_special_values():
    """Test polynomial addition with special values (inf, nan)"""
    degree = 3
    
    a = make_poly(degree, PSI)
    b = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set some special values
    a[0] = np.inf
    a[1] = -np.inf
    a[2] = np.nan
    
    b[0] = 1.0
    b[1] = -np.inf
    b[2] = 1.0
    
    poly_add(a, b, result)
    
    # Check expected behavior for inf + finite = inf
    assert np.isinf(result[0])
    assert result[0] > 0  # positive infinity
    
    # Check expected behavior for -inf + (-inf) = -inf
    assert np.isinf(result[1])
    assert result[1] < 0  # negative infinity
    
    # Check expected behavior for nan + anything = nan
    assert np.isnan(result[2])

def test_add_complex_numbers():
    """Test polynomial addition with complex coefficients"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    # Create complex polynomials
    a = make_poly(degree, PSI, complex_dtype=True)
    b = make_poly(degree, PSI, complex_dtype=True)
    result = make_poly(degree, PSI, complex_dtype=True)
    
    # Set complex coefficients in a
    for i in range(size):
        a[i] = complex(i, i+1)  # a[i] = i + (i+1)j
    
    # Set complex coefficients in b
    for i in range(size):
        b[i] = complex(2*i, -(i+1))  # b[i] = 2i - (i+1)j
    
    # Expected result: a[i] + b[i] = (3i) + 0j
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    # Perform addition
    poly_add(a, b, result)
    
    # Verify result
    np.testing.assert_array_almost_equal(result, expected)
    
    # Verify specific values
    for i in range(size):
        # Real part should be i + 2i = 3i
        assert result[i].real == 3*i
        # Imaginary part should be (i+1) - (i+1) = 0
        assert abs(result[i].imag) < 1e-10
    
    # Test with different patterns of complex numbers
    a = make_poly(degree, PSI, complex_dtype=True)
    b = make_poly(degree, PSI, complex_dtype=True)
    result = make_poly(degree, PSI, complex_dtype=True)
    
    # Pattern 1: Alternating real and imaginary
    for i in range(size):
        if i % 2 == 0:
            a[i] = complex(i, 0)  # pure real
            b[i] = complex(0, i)  # pure imaginary
        else:
            a[i] = complex(0, i)  # pure imaginary
            b[i] = complex(i, 0)  # pure real
    
    # Expected: all coefficients should have both real and imaginary parts = i
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test with complex conjugates
    a = make_poly(degree, PSI, complex_dtype=True)
    b = make_poly(degree, PSI, complex_dtype=True)
    result = make_poly(degree, PSI, complex_dtype=True)
    
    # a and b are complex conjugates of each other
    for i in range(size):
        a[i] = complex(i, i+1)
        b[i] = complex(i, -(i+1))  # complex conjugate of a[i]
    
    # Expected: result should be purely real
    expected = np.zeros_like(a)
    for i in range(size):
        expected[i] = a[i] + b[i]
    
    poly_add(a, b, result)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Verify all results have zero imaginary part
    for i in range(size):
        assert abs(result[i].imag) < 1e-10
        assert result[i].real == 2*i  # real part should be 2*i

def test_poly_scale_basic():
    """Test basic polynomial scaling with various factors"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    # Create test polynomial
    p = make_poly(degree, PSI)
    
    # Set some coefficients
    for i in range(size):
        p[i] = i * 1.5
    
    # Test scaling by different factors
    factors = [0.0, 1.0, -1.0, 2.0, -0.5, 10.0, 0.1]
    
    for factor in factors:
        # Create result polynomial
        result = make_poly(degree, PSI)
        
        # Scale the polynomial
        poly_scale(p, factor, result)
        
        # Check results
        for i in range(size):
            expected = p[i] * factor
            assert result[i] == expected
            
        # Verify that original polynomial hasn't changed
        for i in range(size):
            assert p[i] == i * 1.5

def test_poly_scale_zero():
    """Test scaling polynomial by zero"""
    degree = 3
    
    # Create test polynomial with non-zero values
    p = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set non-zero values
    for i in range(p.shape[0]):
        p[i] = 10.0 * (i + 1)
    
    # Scale by zero
    poly_scale(p, 0.0, result)
    
    # All values should be zero
    assert np.all(result == 0.0)
    
    # Original should be unchanged
    for i in range(p.shape[0]):
        assert p[i] == 10.0 * (i + 1)

def test_poly_scale_large_values():
    """Test scaling with large factors and values"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    # Create test polynomial
    p = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Test with large coefficients
    for i in range(size):
        p[i] = 1e8 * (i + 1)
    
    # Scale by large factor
    large_factor = 1e8
    poly_scale(p, large_factor, result)
    
    # Check results
    for i in range(size):
        expected = p[i] * large_factor
        assert result[i] == expected

def test_poly_scale_small_values():
    """Test scaling with small factors and values"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    # Create test polynomial
    p = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Test with small coefficients
    for i in range(size):
        p[i] = 1e-8 * (i + 1)
    
    # Scale by small factor
    small_factor = 1e-8
    poly_scale(p, small_factor, result)
    
    # Check results
    expected = np.zeros_like(p)
    for i in range(size):
        expected[i] = p[i] * small_factor
    
    np.testing.assert_array_almost_equal(result, expected)

def test_poly_scale_special_values():
    """Test scaling with special values (inf, nan)"""
    degree = 3
    
    # Create test polynomial
    p = make_poly(degree, PSI)
    result = make_poly(degree, PSI)
    
    # Set special values
    p[0] = np.inf
    p[1] = -np.inf
    p[2] = np.nan
    p[3] = 1.0
    
    # Scale by positive value
    poly_scale(p, 2.0, result)
    
    # Check results
    assert np.isinf(result[0])
    assert result[0] > 0  # positive infinity
    assert np.isinf(result[1])
    assert result[1] < 0  # negative infinity
    assert np.isnan(result[2])
    assert result[3] == 2.0
    
    # Scale by negative value
    poly_scale(p, -1.0, result)
    
    # Check results
    assert np.isinf(result[0])
    assert result[0] < 0  # negative infinity (sign flipped)
    assert np.isinf(result[1])
    assert result[1] > 0  # positive infinity (sign flipped)
    assert np.isnan(result[2])
    assert result[3] == -1.0
    
    # Scale by zero
    poly_scale(p, 0.0, result)
    
    # Check results (0 * inf = nan in IEEE 754)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert result[3] == 0.0

def test_poly_scale_complex():
    """Test scaling complex polynomials"""
    degree = 3
    size = PSI[N_VARS, degree]
    
    # Create complex polynomial
    p = make_poly(degree, PSI, complex_dtype=True)
    result = make_poly(degree, PSI, complex_dtype=True)
    
    # Set complex coefficients
    for i in range(size):
        p[i] = complex(i, i+1)
    
    # Test with real scalar
    real_factor = 2.0
    poly_scale(p, real_factor, result)
    
    # Check results
    for i in range(size):
        expected = p[i] * real_factor
        assert result[i].real == expected.real
        assert result[i].imag == expected.imag
    
    # Test with complex scalar
    complex_factor = complex(1.0, 2.0)
    poly_scale(p, complex_factor, result)
    
    # Check results
    for i in range(size):
        expected = p[i] * complex_factor
        assert abs(result[i].real - expected.real) < 1e-10
        assert abs(result[i].imag - expected.imag) < 1e-10
    
    # Test with zero complex scalar
    zero_complex = complex(0.0, 0.0)
    poly_scale(p, zero_complex, result)
    
    # All values should be zero
    for i in range(size):
        assert result[i].real == 0.0
        assert result[i].imag == 0.0

def test_poly_mul_monomials():
    """Test multiplication of monomials (single-term polynomials)"""
    # Test with low degrees to keep the test easy to reason about
    deg_p = 1  # Linear monomial
    deg_q = 2  # Quadratic monomial
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    
    # Set p to represent x0
    # Find the index for the monomial x0
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x0] = 2.0  # p = 2*x0
    
    # Set q to represent x0*x1
    # Find the index for the monomial x0*x1
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x0x1] = 3.0  # q = 3*x0*x1
    
    # Multiply the polynomials
    result = poly_mul(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Verify the degree of the result
    assert result.shape[0] == PSI[N_VARS, deg_p + deg_q]
    
    # The result should be 6*x0^2*x1
    idx_x0x0x1 = encode_multiindex(np.array([2, 1, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    assert abs(result[idx_x0x0x1] - 6.0) < 1e-10
    
    # Verify all other terms are zero
    result[idx_x0x0x1] = 0.0
    assert np.all(result == 0.0)
    
    # Test with a different pair of monomials
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    
    # p = 4*x1
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x1] = 4.0
    
    # q = 5*x2^2
    idx_x2sq = encode_multiindex(np.array([0, 0, 2, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x2sq] = 5.0
    
    # Multiply
    result = poly_mul(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Result should be 20*x1*x2^2
    idx_result = encode_multiindex(np.array([0, 1, 2, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    assert abs(result[idx_result] - 20.0) < 1e-10
    
    # Verify all other terms are zero
    result[idx_result] = 0.0
    assert np.all(result == 0.0)

def test_poly_mul_basic():
    """Test basic polynomial multiplication"""
    # Use small degrees to keep the test manageable
    deg_p = 1
    deg_q = 1
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    
    # Set p = 1 + 2*x0 + 3*x1
    idx_const = encode_multiindex(np.array([0, 0, 0, 0, 0, 0], dtype=np.int64), 0, PSI, CLMO)
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    
    # Note: constant term (degree 0) is not part of the degree 1 polynomial p
    # So we work with the linear terms only
    p[idx_x0] = 2.0
    p[idx_x1] = 3.0
    
    # Set q = 4 + 5*x0 + 6*x1
    idx_x0q = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    idx_x1q = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    
    q[idx_x0q] = 5.0
    q[idx_x1q] = 6.0
    
    # Multiply the polynomials
    result = poly_mul(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Expected terms in the result:
    # 2*x0 * 5*x0 = 10*x0^2
    # 2*x0 * 6*x1 = 12*x0*x1
    # 3*x1 * 5*x0 = 15*x0*x1
    # 3*x1 * 6*x1 = 18*x1^2
    
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    
    assert abs(result[idx_x0sq] - 10.0) < 1e-10
    assert abs(result[idx_x0x1] - 27.0) < 1e-10  # 12 + 15
    assert abs(result[idx_x1sq] - 18.0) < 1e-10

def test_poly_mul_commutativity():
    """Test that polynomial multiplication is commutative: p * q = q * p"""
    deg_p = 2
    deg_q = 1
    
    # Create two polynomials with random coefficients
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    
    # Set random coefficients
    for i in range(p.shape[0]):
        p[i] = i * 1.5
    
    for i in range(q.shape[0]):
        q[i] = i * 0.75
    
    # Compute p * q and q * p
    result1 = poly_mul(p, deg_p, q, deg_q, PSI, CLMO)
    result2 = poly_mul(q, deg_q, p, deg_p, PSI, CLMO)
    
    # They should be equal
    np.testing.assert_array_almost_equal(result1, result2)

def test_poly_mul_zero():
    """Test multiplication with zero polynomials"""
    deg_p = 2
    deg_q = 2
    
    # Create polynomials
    p = make_poly(deg_p, PSI)
    zero_poly = make_poly(deg_q, PSI)  # All zeros
    
    # Set some values in p
    for i in range(p.shape[0]):
        p[i] = i * 2.0
    
    # Multiply by zero
    result = poly_mul(p, deg_p, zero_poly, deg_q, PSI, CLMO)
    
    # Result should be all zeros
    assert np.all(result == 0.0)
    
    # Also test zero times anything is zero
    result = poly_mul(zero_poly, deg_q, p, deg_p, PSI, CLMO)
    assert np.all(result == 0.0)

def test_poly_mul_identity():
    """Test multiplication by the constant polynomial 1.0"""
    deg_p = 2
    
    # Constant polynomial is degree 0
    deg_const = 0
    
    # Create polynomials
    p = make_poly(deg_p, PSI)
    const_one = make_poly(deg_const, PSI)
    
    # Set const_one to be the constant polynomial 1.0
    const_one[0] = 1.0
    
    # Set some values in p
    for i in range(p.shape[0]):
        p[i] = i * 1.5
    
    # p * 1 should equal p
    result = poly_mul(p, deg_p, const_one, deg_const, PSI, CLMO)
    
    # The result should have the same degree as p
    assert result.shape[0] == PSI[N_VARS, deg_p]
    
    # The coefficients should match p
    np.testing.assert_array_almost_equal(result, p)
    
    # 1 * p should also equal p
    result = poly_mul(const_one, deg_const, p, deg_p, PSI, CLMO)
    np.testing.assert_array_almost_equal(result, p)

def test_poly_mul_complex():
    """Test multiplication of complex polynomials"""
    deg_p = 1
    deg_q = 1
    
    # Create complex polynomials
    p = make_poly(deg_p, PSI, complex_dtype=True)
    q = make_poly(deg_q, PSI, complex_dtype=True)
    
    # Set values in p and q
    # p = (1+i)*x0 + (2+2i)*x1
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    
    p[idx_x0] = complex(1.0, 1.0)
    p[idx_x1] = complex(2.0, 2.0)
    
    # q = (3-i)*x0 + (1-i)*x1
    idx_x0q = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    idx_x1q = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    
    q[idx_x0q] = complex(3.0, -1.0)
    q[idx_x1q] = complex(1.0, -1.0)
    
    # Multiply
    result = poly_mul(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Expected result:
    # (1+i)*(3-i)*x0^2 = (3-i+3i-i^2)*x0^2 = (3-i+3i+1)*x0^2 = (4+2i)*x0^2
    # (1+i)*(1-i)*x0*x1 = (1-i+i-i^2)*x0*x1 = (1-i+i+1)*x0*x1 = 2*x0*x1
    # (2+2i)*(3-i)*x0*x1 = (6-2i+6i-2i^2)*x0*x1 = (6-2i+6i+2)*x0*x1 = (8+4i)*x0*x1
    # (2+2i)*(1-i)*x1^2 = (2-2i+2i-2i^2)*x1^2 = (2-2i+2i+2)*x1^2 = 4*x1^2
    
    # So combined terms:
    # (4+2i)*x0^2 + (10+4i)*x0*x1 + (4+0i)*x1^2
    
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), deg_p + deg_q, PSI, CLMO)
    
    # Check results
    assert abs(result[idx_x0sq].real - 4.0) < 1e-10
    assert abs(result[idx_x0sq].imag - 2.0) < 1e-10
    
    assert abs(result[idx_x0x1].real - 10.0) < 1e-10
    assert abs(result[idx_x0x1].imag - 4.0) < 1e-10
    
    assert abs(result[idx_x1sq].real - 4.0) < 1e-10
    assert abs(result[idx_x1sq].imag) < 1e-10

def test_differentiate_monomial():
    """Test differentiation of a monomial"""
    # Test differentiation of a simple monomial: 3*x0^2
    degree = 2
    p = make_poly(degree, PSI)
    
    # Set p = 3*x0^2
    idx = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    p[idx] = 3.0
    
    # Differentiate with respect to x0
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # Result should be 6*x0 (d/dx0(3*x0^2) = 6*x0)
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    assert dp[idx_x0] == 6.0
    
    # All other terms should be zero
    dp[idx_x0] = 0.0
    assert np.all(dp == 0.0)
    
    # Differentiate with respect to a variable that doesn't appear in the term
    dp2 = differentiate(p, 1, degree, PSI, CLMO)
    
    # Result should be all zeros
    assert np.all(dp2 == 0.0)

def test_differentiate_mixed_terms():
    """Test differentiation of a polynomial with multiple terms"""
    # Create p = 2*x0^2 + 3*x0*x1 + 4*x1^2
    degree = 2
    p = make_poly(degree, PSI)
    
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    
    p[idx_x0sq] = 2.0
    p[idx_x0x1] = 3.0
    p[idx_x1sq] = 4.0
    
    # Differentiate with respect to x0
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # d/dx0(2*x0^2 + 3*x0*x1 + 4*x1^2) = 4*x0 + 3*x1
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    
    assert dp[idx_x0] == 4.0
    assert dp[idx_x1] == 3.0
    
    # All other terms should be zero
    dp[idx_x0] = 0.0
    dp[idx_x1] = 0.0
    assert np.all(dp == 0.0)
    
    # Differentiate with respect to x1
    dp2 = differentiate(p, 1, degree, PSI, CLMO)
    
    # d/dx1(2*x0^2 + 3*x0*x1 + 4*x1^2) = 3*x0 + 8*x1
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    
    assert dp2[idx_x0] == 3.0
    assert dp2[idx_x1] == 8.0
    
    # All other terms should be zero
    dp2[idx_x0] = 0.0
    dp2[idx_x1] = 0.0
    assert np.all(dp2 == 0.0)

def test_differentiate_higher_degree():
    """Test differentiation of higher-degree polynomials"""
    # Create p = x0^3 + x0^2*x1 + x0*x1^2 + x1^3
    degree = 3
    p = make_poly(degree, PSI)
    
    idx_x0_3 = encode_multiindex(np.array([3, 0, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x0_2_x1 = encode_multiindex(np.array([2, 1, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x0_x1_2 = encode_multiindex(np.array([1, 2, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x1_3 = encode_multiindex(np.array([0, 3, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    
    p[idx_x0_3] = 1.0
    p[idx_x0_2_x1] = 1.0
    p[idx_x0_x1_2] = 1.0
    p[idx_x1_3] = 1.0
    
    # Differentiate with respect to x0
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # d/dx0(x0^3 + x0^2*x1 + x0*x1^2 + x1^3) = 3*x0^2 + 2*x0*x1 + x1^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    
    assert dp[idx_x0sq] == 3.0
    assert dp[idx_x0x1] == 2.0
    assert dp[idx_x1sq] == 1.0
    
    # Check all other terms are zero
    dp[idx_x0sq] = 0.0
    dp[idx_x0x1] = 0.0
    dp[idx_x1sq] = 0.0
    assert np.all(dp == 0.0)

def test_differentiate_zero_polynomial():
    """Test differentiation of zero polynomial"""
    degree = 2
    p = make_poly(degree, PSI)  # All zeros by default
    
    # Differentiate zero polynomial
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # Result should be a zero polynomial of degree-1
    assert np.all(dp == 0.0)

def test_differentiate_constant():
    """Test differentiation of a constant term (should be zero)"""
    # Note: In this implementation, a degree-0 polynomial is just a constant
    degree = 0
    p = make_poly(degree, PSI)
    p[0] = 5.0  # Constant polynomial with value 5
    
    # For a Numba JIT function, trying to differentiate a constant will not raise an exception
    # but should return an appropriate result (likely an empty array or zeros)
    for var in range(N_VARS):
        dp = differentiate(p, var, degree, PSI, CLMO)
        
        # The size should be PSI[N_VARS, -1] which should be 0 or a very small array
        # Just verify that the result doesn't contain any non-zero values
        for i in range(dp.shape[0]):
            assert dp[i] == 0.0

def test_differentiate_multi_variable():
    """Test differentiation with respect to different variables"""
    # Create p = x0*x1*x2
    degree = 3
    p = make_poly(degree, PSI)
    
    idx = encode_multiindex(np.array([1, 1, 1, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    p[idx] = 2.0
    
    # Differentiate with respect to x0
    dp0 = differentiate(p, 0, degree, PSI, CLMO)
    
    # d/dx0(2*x0*x1*x2) = 2*x1*x2
    idx_x1x2 = encode_multiindex(np.array([0, 1, 1, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    assert dp0[idx_x1x2] == 2.0
    
    # Differentiate with respect to x1
    dp1 = differentiate(p, 1, degree, PSI, CLMO)
    
    # d/dx1(2*x0*x1*x2) = 2*x0*x2
    idx_x0x2 = encode_multiindex(np.array([1, 0, 1, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    assert dp1[idx_x0x2] == 2.0
    
    # Differentiate with respect to x2
    dp2 = differentiate(p, 2, degree, PSI, CLMO)
    
    # d/dx2(2*x0*x1*x2) = 2*x0*x1
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    assert dp2[idx_x0x1] == 2.0
    
    # Differentiate with respect to x3 (not in the polynomial)
    dp3 = differentiate(p, 3, degree, PSI, CLMO)
    
    # d/dx3(2*x0*x1*x2) = 0
    assert np.all(dp3 == 0.0)

def test_differentiate_high_exponent():
    """Test differentiation of terms with high exponents"""
    # Create p = x0^4
    degree = 4
    p = make_poly(degree, PSI)
    
    idx = encode_multiindex(np.array([4, 0, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    p[idx] = 1.0
    
    # Differentiate with respect to x0
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # d/dx0(x0^4) = 4*x0^3
    idx_x0_3 = encode_multiindex(np.array([3, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    assert dp[idx_x0_3] == 4.0
    
    # Differentiate again
    dp2 = differentiate(dp, 0, degree-1, PSI, CLMO)
    
    # d/dx0(4*x0^3) = 12*x0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), degree-2, PSI, CLMO)
    assert dp2[idx_x0sq] == 12.0
    
    # Differentiate a third time
    dp3 = differentiate(dp2, 0, degree-2, PSI, CLMO)
    
    # d/dx0(12*x0^2) = 24*x0
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-3, PSI, CLMO)
    assert dp3[idx_x0] == 24.0
    
    # Differentiate a fourth time
    dp4 = differentiate(dp3, 0, degree-3, PSI, CLMO)
    
    # d/dx0(24*x0) = 24
    idx_const = encode_multiindex(np.array([0, 0, 0, 0, 0, 0], dtype=np.int64), degree-4, PSI, CLMO)
    assert dp4[idx_const] == 24.0

def test_differentiate_complex():
    """Test differentiation of complex polynomials"""
    # Create a complex polynomial
    degree = 2
    p = make_poly(degree, PSI, complex_dtype=True)
    
    # Set p = (1+2i)*x0^2 + (3+4i)*x0*x1 + (5+6i)*x1^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), degree, PSI, CLMO)
    
    p[idx_x0sq] = complex(1.0, 2.0)
    p[idx_x0x1] = complex(3.0, 4.0)
    p[idx_x1sq] = complex(5.0, 6.0)
    
    # Differentiate with respect to x0
    dp = differentiate(p, 0, degree, PSI, CLMO)
    
    # d/dx0((1+2i)*x0^2 + (3+4i)*x0*x1 + (5+6i)*x1^2) = (2+4i)*x0 + (3+4i)*x1
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    
    assert dp[idx_x0].real == 2.0
    assert dp[idx_x0].imag == 4.0
    assert dp[idx_x1].real == 3.0
    assert dp[idx_x1].imag == 4.0
    
    # Differentiate with respect to x1
    dp2 = differentiate(p, 1, degree, PSI, CLMO)
    
    # d/dx1((1+2i)*x0^2 + (3+4i)*x0*x1 + (5+6i)*x1^2) = (3+4i)*x0 + (10+12i)*x1
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), degree-1, PSI, CLMO)
    
    assert dp2[idx_x0].real == 3.0
    assert dp2[idx_x0].imag == 4.0
    assert dp2[idx_x1].real == 10.0
    assert dp2[idx_x1].imag == 12.0

def test_poisson_antisymmetry():
    """Test antisymmetry property of Poisson bracket: {P, Q} = -{Q, P}"""
    # Create two test polynomials
    deg_p = 2
    deg_q = 2
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    
    # Set values in p and q
    # p = x0^2 + x0*x1
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    idx_x0x1 = encode_multiindex(np.array([1, 1, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    
    p[idx_x0sq] = 1.0
    p[idx_x0x1] = 1.0
    
    # q = p0^2 + p0*p1
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    idx_p0p1 = encode_multiindex(np.array([0, 0, 0, 1, 1, 0], dtype=np.int64), deg_q, PSI, CLMO)
    
    q[idx_p0sq] = 1.0
    q[idx_p0p1] = 1.0
    
    # Compute {P, Q} and {Q, P}
    pq = poisson(p, deg_p, q, deg_q, PSI, CLMO)
    qp = poisson(q, deg_q, p, deg_p, PSI, CLMO)
    
    # Compute -{Q, P} (scale qp by -1)
    neg_qp = np.zeros_like(qp)
    poly_scale(qp, -1.0, neg_qp)
    
    # {P, Q} should equal -{Q, P}
    np.testing.assert_array_almost_equal(pq, neg_qp)

def test_poisson_first_argument_linearity():
    """Test linearity in the first argument: {aP+bQ, R} = a{P,R} + b{Q,R}"""
    # Create three test polynomials
    deg_p = 2
    deg_q = 2
    deg_r = 2
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    r = make_poly(deg_r, PSI)
    
    # Set values
    # p = x0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x0sq] = 1.0
    
    # q = x1^2
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x1sq] = 1.0
    
    # r = p0^2
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_r, PSI, CLMO)
    r[idx_p0sq] = 1.0
    
    # Define coefficients a and b
    a = 2.0
    b = 3.0
    
    # Compute a*p + b*q
    ap = make_poly(deg_p, PSI)
    bq = make_poly(deg_q, PSI)
    apbq = make_poly(deg_p, PSI)  # assuming deg_p == deg_q
    
    poly_scale(p, a, ap)
    poly_scale(q, b, bq)
    poly_add(ap, bq, apbq)
    
    # Compute {aP+bQ, R}
    apbq_r = poisson(apbq, deg_p, r, deg_r, PSI, CLMO)
    
    # Compute a{P,R} + b{Q,R}
    pr = poisson(p, deg_p, r, deg_r, PSI, CLMO)
    qr = poisson(q, deg_q, r, deg_r, PSI, CLMO)
    
    apr = make_poly(deg_p + deg_r - 2, PSI)
    bqr = make_poly(deg_q + deg_r - 2, PSI)
    result = make_poly(deg_p + deg_r - 2, PSI)
    
    poly_scale(pr, a, apr)
    poly_scale(qr, b, bqr)
    poly_add(apr, bqr, result)
    
    # {aP+bQ, R} should equal a{P,R} + b{Q,R}
    np.testing.assert_array_almost_equal(apbq_r, result)

def test_poisson_second_argument_linearity():
    """Test linearity in the second argument: {P, aQ+bR} = a{P,Q} + b{P,R}"""
    # Create three test polynomials
    deg_p = 2
    deg_q = 2
    deg_r = 2
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    r = make_poly(deg_r, PSI)
    
    # Set values
    # p = x0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x0sq] = 1.0
    
    # q = x1^2
    idx_x1sq = encode_multiindex(np.array([0, 2, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x1sq] = 1.0
    
    # r = p0^2
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_r, PSI, CLMO)
    r[idx_p0sq] = 1.0
    
    # Define coefficients a and b
    a = 2.0
    b = 3.0
    
    # Compute a*q + b*r
    aq = make_poly(deg_q, PSI)
    br = make_poly(deg_r, PSI)
    aqbr = make_poly(deg_q, PSI)  # assuming deg_q == deg_r
    
    poly_scale(q, a, aq)
    poly_scale(r, b, br)
    poly_add(aq, br, aqbr)
    
    # Compute {P, aQ+bR}
    p_aqbr = poisson(p, deg_p, aqbr, deg_q, PSI, CLMO)
    
    # Compute a{P,Q} + b{P,R}
    pq = poisson(p, deg_p, q, deg_q, PSI, CLMO)
    pr = poisson(p, deg_p, r, deg_r, PSI, CLMO)
    
    apq = make_poly(deg_p + deg_q - 2, PSI)
    bpr = make_poly(deg_p + deg_r - 2, PSI)
    result = make_poly(deg_p + deg_q - 2, PSI)
    
    poly_scale(pq, a, apq)
    poly_scale(pr, b, bpr)
    poly_add(apq, bpr, result)
    
    # {P, aQ+bR} should equal a{P,Q} + b{P,R}
    np.testing.assert_array_almost_equal(p_aqbr, result)

def test_poisson_jacobi_identity():
    """Test Jacobi identity: {P, {Q, R}} + {Q, {R, P}} + {R, {P, Q}} = 0"""
    # Create three test polynomials of low degree to keep test manageable
    deg_p = 2
    deg_q = 2
    deg_r = 2
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    r = make_poly(deg_r, PSI)
    
    # Set values - simple polynomials to test the identity
    # p = x0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x0sq] = 1.0
    
    # q = p0^2
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_p0sq] = 1.0
    
    # r = x0*p0
    idx_x0p0 = encode_multiindex(np.array([1, 0, 0, 1, 0, 0], dtype=np.int64), deg_r, PSI, CLMO)
    r[idx_x0p0] = 1.0
    
    # Compute {Q, R}
    qr_deg = deg_q + deg_r - 2
    qr = poisson(q, deg_q, r, deg_r, PSI, CLMO)
    
    # Compute {P, {Q, R}}
    p_qr_deg = deg_p + qr_deg - 2
    p_qr = poisson(p, deg_p, qr, qr_deg, PSI, CLMO)
    
    # Compute {R, P}
    rp_deg = deg_r + deg_p - 2
    rp = poisson(r, deg_r, p, deg_p, PSI, CLMO)
    
    # Compute {Q, {R, P}}
    q_rp_deg = deg_q + rp_deg - 2
    q_rp = poisson(q, deg_q, rp, rp_deg, PSI, CLMO)
    
    # Compute {P, Q}
    pq_deg = deg_p + deg_q - 2
    pq = poisson(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Compute {R, {P, Q}}
    r_pq_deg = deg_r + pq_deg - 2
    r_pq = poisson(r, deg_r, pq, pq_deg, PSI, CLMO)
    
    # Compute the sum: {P, {Q, R}} + {Q, {R, P}} + {R, {P, Q}}
    sum_deg = max(p_qr_deg, q_rp_deg, r_pq_deg)
    sum_result = make_poly(sum_deg, PSI)
    
    temp = make_poly(sum_deg, PSI)
    poly_add(p_qr, q_rp, temp)
    poly_add(temp, r_pq, sum_result)
    
    # The sum should be zero
    for i in range(sum_result.shape[0]):
        assert abs(sum_result[i]) < 1e-10

def test_poisson_hamiltonian():
    """Test Poisson bracket properties with the Hamiltonian H = 0.5*p0^2 + q0^2"""
    # Create Hamiltonian H = 0.5*p0^2 + q0^2
    deg_h = 2
    h = make_poly(deg_h, PSI)
    
    # 0.5*p0^2 term
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_h, PSI, CLMO)
    h[idx_p0sq] = 0.5
    
    # q0^2 term (x0^2)
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_h, PSI, CLMO)
    h[idx_x0sq] = 1.0
    
    # Test 1: {H, H} = 0
    hh = poisson(h, deg_h, h, deg_h, PSI, CLMO)
    
    # {H, H} should be zero
    for i in range(hh.shape[0]):
        assert abs(hh[i]) < 1e-10
    
    # Test 2: {H, q0} = -p0
    # Create q0 polynomial (x0)
    deg_q0 = 1
    q0 = make_poly(deg_q0, PSI)
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), deg_q0, PSI, CLMO)
    q0[idx_x0] = 1.0
    
    # Compute {H, q0}
    h_q0 = poisson(h, deg_h, q0, deg_q0, PSI, CLMO)
    
    # Expected result: -p0
    idx_p0 = encode_multiindex(np.array([0, 0, 0, 1, 0, 0], dtype=np.int64), 1, PSI, CLMO)
    expected_h_q0 = make_poly(1, PSI)
    expected_h_q0[idx_p0] = -1.0
    
    # {H, q0} should equal -p0
    np.testing.assert_array_almost_equal(h_q0, expected_h_q0)
    
    # Test 3: {H, p0} = 2*q0 (for this implementation)
    # Create p0 polynomial
    deg_p0 = 1
    p0 = make_poly(deg_p0, PSI)
    idx_p0 = encode_multiindex(np.array([0, 0, 0, 1, 0, 0], dtype=np.int64), deg_p0, PSI, CLMO)
    p0[idx_p0] = 1.0
    
    # Compute {H, p0}
    h_p0 = poisson(h, deg_h, p0, deg_p0, PSI, CLMO)
    
    # Expected result: 2*q0 (2*x0)
    idx_x0 = encode_multiindex(np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), 1, PSI, CLMO)
    expected_h_p0 = make_poly(1, PSI)
    expected_h_p0[idx_x0] = 2.0
    
    # {H, p0} should equal 2*q0
    np.testing.assert_array_almost_equal(h_p0, expected_h_p0)

def test_poisson_complex():
    """Test Poisson bracket with complex polynomials"""
    # Create complex polynomials
    deg_p = 2
    deg_q = 2
    
    p = make_poly(deg_p, PSI, complex_dtype=True)
    q = make_poly(deg_q, PSI, complex_dtype=True)
    
    # Set values in p and q
    # p = (1+i)*x0^2 + (2+2i)*p0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    
    p[idx_x0sq] = complex(1.0, 1.0)
    p[idx_p0sq] = complex(2.0, 2.0)
    
    # q = (3-i)*x0*p0
    idx_x0p0 = encode_multiindex(np.array([1, 0, 0, 1, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x0p0] = complex(3.0, -1.0)
    
    # Compute {P, Q} and {Q, P}
    pq = poisson(p, deg_p, q, deg_q, PSI, CLMO)
    qp = poisson(q, deg_q, p, deg_p, PSI, CLMO)
    
    # Test antisymmetry with complex polynomials
    neg_qp = make_poly(deg_p + deg_q - 2, PSI, complex_dtype=True)
    poly_scale(qp, -1.0, neg_qp)
    
    # {P, Q} should equal -{Q, P}
    np.testing.assert_array_almost_equal(pq, neg_qp)
    
    # Verify a specific term in the result to check complex arithmetic correctness
    # We'd need to calculate the expected result manually for a specific term

def test_poisson_leibniz_rule():
    """Test Leibniz product rule: {P, Q*R} = {P, Q}*R + Q*{P, R}"""
    # Create test polynomials of low degree
    deg_p = 2
    deg_q = 1
    deg_r = 1
    
    p = make_poly(deg_p, PSI)
    q = make_poly(deg_q, PSI)
    r = make_poly(deg_r, PSI)
    
    # Set values for simple test case
    # p = x0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_p, PSI, CLMO)
    p[idx_x0sq] = 1.0
    
    # q = x1
    idx_x1 = encode_multiindex(np.array([0, 1, 0, 0, 0, 0], dtype=np.int64), deg_q, PSI, CLMO)
    q[idx_x1] = 1.0
    
    # r = p0
    idx_p0 = encode_multiindex(np.array([0, 0, 0, 1, 0, 0], dtype=np.int64), deg_r, PSI, CLMO)
    r[idx_p0] = 1.0
    
    # Compute Q*R
    qr = poly_mul(q, deg_q, r, deg_r, PSI, CLMO)
    deg_qr = deg_q + deg_r
    
    # Compute {P, Q*R}
    p_qr = poisson(p, deg_p, qr, deg_qr, PSI, CLMO)
    
    # Compute {P, Q}
    pq = poisson(p, deg_p, q, deg_q, PSI, CLMO)
    
    # Compute {P, Q}*R
    pq_r = poly_mul(pq, deg_p + deg_q - 2, r, deg_r, PSI, CLMO)
    
    # Compute {P, R}
    pr = poisson(p, deg_p, r, deg_r, PSI, CLMO)
    
    # Compute Q*{P, R}
    q_pr = poly_mul(q, deg_q, pr, deg_p + deg_r - 2, PSI, CLMO)
    
    # Compute {P, Q}*R + Q*{P, R}
    result = make_poly(deg_p + deg_qr - 2, PSI)
    poly_add(pq_r, q_pr, result)
    
    # {P, Q*R} should equal {P, Q}*R + Q*{P, R}
    np.testing.assert_array_almost_equal(p_qr, result)

def test_poisson_constant():
    """Test Poisson bracket with constant function: {1, F} = 0"""
    # Create constant polynomial 1
    deg_const = 0
    const_one = make_poly(deg_const, PSI)
    const_one[0] = 1.0
    
    # Create test polynomial
    deg_f = 2
    f = make_poly(deg_f, PSI)
    
    # Set f = x0^2 + p0^2
    idx_x0sq = encode_multiindex(np.array([2, 0, 0, 0, 0, 0], dtype=np.int64), deg_f, PSI, CLMO)
    idx_p0sq = encode_multiindex(np.array([0, 0, 0, 2, 0, 0], dtype=np.int64), deg_f, PSI, CLMO)
    
    f[idx_x0sq] = 1.0
    f[idx_p0sq] = 1.0
    
    # Compute {1, F}
    one_f = poisson(const_one, deg_const, f, deg_f, PSI, CLMO)
    
    # {1, F} should be zero
    for i in range(one_f.shape[0]):
        assert abs(one_f[i]) < 1e-10
    
    # Test the other way: {F, 1}
    f_one = poisson(f, deg_f, const_one, deg_const, PSI, CLMO)
    
    # {F, 1} should also be zero
    for i in range(f_one.shape[0]):
        assert abs(f_one[i]) < 1e-10

def test_poisson_canonical_relations():
    """Test canonical Poisson bracket relations: {q_i,q_j}=0, {p_i,p_j}=0, {q_i,p_j}=δ_ij"""
    # Test for the first few position and momentum variables
    for i in range(3):  # Test x0, x1, x2
        for j in range(3):  # Test x0, x1, x2
            # Create position variables q_i and q_j
            deg_q = 1
            qi = make_poly(deg_q, PSI)
            qj = make_poly(deg_q, PSI)
            
            # Create arrays for exponents first, then encode them
            idx_qi_array = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            idx_qi_array[i] = 1  # Set exponent for variable i
            
            idx_qj_array = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            idx_qj_array[j] = 1  # Set exponent for variable j
            
            qi_idx = encode_multiindex(idx_qi_array, deg_q, PSI, CLMO)
            qj_idx = encode_multiindex(idx_qj_array, deg_q, PSI, CLMO)
            
            qi[qi_idx] = 1.0
            qj[qj_idx] = 1.0
            
            # Create momentum variables p_i and p_j
            pi = make_poly(deg_q, PSI)
            pj = make_poly(deg_q, PSI)
            
            # Create arrays for exponents first, then encode them
            idx_pi_array = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            idx_pi_array[i+3] = 1  # p0,p1,p2 are at indices 3,4,5
            
            idx_pj_array = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            idx_pj_array[j+3] = 1  # p0,p1,p2 are at indices 3,4,5
            
            pi_idx = encode_multiindex(idx_pi_array, deg_q, PSI, CLMO)
            pj_idx = encode_multiindex(idx_pj_array, deg_q, PSI, CLMO)
            
            pi[pi_idx] = 1.0
            pj[pj_idx] = 1.0
            
            # Test {q_i, q_j} = 0
            qi_qj = poisson(qi, deg_q, qj, deg_q, PSI, CLMO)
            
            # Should be zero (positions Poisson-commute)
            for k in range(qi_qj.shape[0]):
                assert abs(qi_qj[k]) < 1e-10
            
            # Test {p_i, p_j} = 0
            pi_pj = poisson(pi, deg_q, pj, deg_q, PSI, CLMO)
            
            # Should be zero (momenta Poisson-commute)
            for k in range(pi_pj.shape[0]):
                assert abs(pi_pj[k]) < 1e-10
            
            # Test {q_i, p_j} = δ_ij (Kronecker delta)
            qi_pj = poisson(qi, deg_q, pj, deg_q, PSI, CLMO)
            
            if i == j:
                # {q_i, p_i} should be 1
                # The polynomial should have a single constant term equal to 1
                assert qi_pj.shape[0] == 1  # Degree 0 polynomial (constant)
                assert abs(qi_pj[0] - 1.0) < 1e-10
            else:
                # {q_i, p_j} should be 0 for i ≠ j
                for k in range(qi_pj.shape[0]):
                    assert abs(qi_pj[k]) < 1e-10