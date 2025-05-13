import numpy as np
import pytest
from numba.typed import List

from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex, decode_multiindex
from algorithms.variables import N_VARS
from algorithms.center.lie import (_get_homogeneous_terms, _select_terms_for_elimination, 
                                  _solve_homological_equation, _apply_lie_transform, lie_transform)
from algorithms.center.polynomial.algebra import poisson

# Initialize tables for tests
MAX_DEGREE = 5
PSI, CLMO = init_index_tables(MAX_DEGREE)

def test_get_homogeneous_terms():
    """Test extracting homogeneous terms of a specific degree from a polynomial."""
    # Create a list to hold polynomial coefficients for each degree
    H_coeffs = List()
    
    # Add empty arrays for degree 0 and 1 (we'll focus on degrees 2 and 3)
    H_coeffs.append(np.zeros(PSI[N_VARS, 0], dtype=np.float64))
    H_coeffs.append(np.zeros(PSI[N_VARS, 1], dtype=np.float64))
    
    # Create a degree 2 polynomial: 2*q1^2 + 3*q1*q2 + 7*p1*p2
    degree2 = np.zeros(PSI[N_VARS, 2], dtype=np.float64)
    
    # Set coefficient for q1^2 (exponents: [2,0,0,0,0,0])
    k = np.array([2, 0, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 2.0
    
    # Set coefficient for q1*q2 (exponents: [1,1,0,0,0,0])
    k = np.array([1, 1, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 3.0
    
    # Set coefficient for p1*p2 (exponents: [0,0,0,1,1,0])
    k = np.array([0, 0, 0, 1, 1, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 7.0
    
    H_coeffs.append(degree2)
    
    # Create a degree 3 polynomial: 5*q1*q2*q3
    degree3 = np.zeros(PSI[N_VARS, 3], dtype=np.float64)
    
    # Set coefficient for q1*q2*q3 (exponents: [1,1,1,0,0,0])
    k = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 3, PSI, CLMO)
    degree3[idx] = 5.0
    
    H_coeffs.append(degree3)
    
    # Test extracting degree 2 terms
    homogeneous_degree2 = _get_homogeneous_terms(H_coeffs, 2, PSI)
    
    # Verify it's the same as the degree 2 array we created
    np.testing.assert_array_equal(homogeneous_degree2, degree2)
    
    # Test extracting degree 3 terms
    homogeneous_degree3 = _get_homogeneous_terms(H_coeffs, 3, PSI)
    
    # Verify it's the same as the degree 3 array we created
    np.testing.assert_array_equal(homogeneous_degree3, degree3)
    
    # Test extracting degree 4 terms (which don't exist, should return zeros)
    homogeneous_degree4 = _get_homogeneous_terms(H_coeffs, 4, PSI)
    
    # Verify it's a zero array of the correct size
    expected_zeros = np.zeros(PSI[N_VARS, 4], dtype=np.float64)
    np.testing.assert_array_equal(homogeneous_degree4, expected_zeros)
    
    # Test extracting degree 4 terms with complex_dtype=True
    homogeneous_degree4_complex = _get_homogeneous_terms(H_coeffs, 4, PSI, complex_dtype=True)
    
    # Verify it's a zero array of the correct size and complex type
    expected_zeros_complex = np.zeros(PSI[N_VARS, 4], dtype=np.complex128)
    np.testing.assert_array_equal(homogeneous_degree4_complex, expected_zeros_complex)

def test_select_terms_for_elimination():
    """Test selecting terms for elimination based on exponent criteria."""
    # Create a degree 3 polynomial: 2*q1*q2*p3 + 4*q1*p1*p2 + 3*q2^2*p2 + 5*q1^2*p1
    degree = 3
    poly = np.zeros(PSI[N_VARS, degree], dtype=np.float64)
    
    # Set coefficient for q1*q2*p3 (exponents: [1,1,0,0,0,1])
    k = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    poly[idx] = 2.0
    
    # Set coefficient for q1*p1*p2 (exponents: [1,0,0,1,1,0])
    k = np.array([1, 0, 0, 1, 1, 0], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    poly[idx] = 4.0
    
    # Set coefficient for q2^2*p2 (exponents: [0,2,0,0,1,0])
    k = np.array([0, 2, 0, 0, 1, 0], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    poly[idx] = 3.0
    
    # Set coefficient for q1^2*p1 (exponents: [2,0,0,1,0,0])
    k = np.array([2, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    poly[idx] = 5.0
    
    # Call the function to select terms for elimination
    result = _select_terms_for_elimination(poly, degree, PSI, CLMO)
    
    # We expect terms where exponent of q1 != exponent of p1 to be kept
    # Create expected result with only those terms that should be selected
    expected = np.zeros_like(poly)
    
    # q1*q2*p3 should be selected (q1 exp=1, p1 exp=0)
    k = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    expected[idx] = 2.0
    
    # q2^2*p2 should NOT be selected (q1 exp=0, p1 exp=0)
    # This term is eliminated because the exponents are equal
    
    # q1^2*p1 should be selected (q1 exp=2, p1 exp=1)
    k = np.array([2, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    expected[idx] = 5.0
    
    # q1*p1*p2 should NOT be selected (q1 exp=1, p1 exp=1)
    # This term is eliminated because the exponents are equal
    
    # Verify the result matches our expectation
    np.testing.assert_array_equal(result, expected)
    
    # Test with all terms that should be eliminated
    poly_all_equal = np.zeros(PSI[N_VARS, 2], dtype=np.float64)
    
    # q1*p1 (exponents: [1,0,0,1,0,0])
    k = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    poly_all_equal[idx] = 3.0
    
    # q2*p2 (exponents: [0,1,0,0,1,0])
    k = np.array([0, 1, 0, 0, 1, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    poly_all_equal[idx] = 4.0
    
    result_all_equal = _select_terms_for_elimination(poly_all_equal, 2, PSI, CLMO)
    expected_all_equal = np.zeros_like(poly_all_equal)
    
    # Verify all terms are eliminated
    np.testing.assert_array_equal(result_all_equal, expected_all_equal)

def test_solve_homological_equation():
    """Test solving the homological equation for normal form transformation."""
    # Create a homogeneous polynomial to eliminate: 2*q1*q2*p3
    degree = 3
    poly_to_eliminate = np.zeros(PSI[N_VARS, degree], dtype=np.complex128)
    
    # Set coefficient for q1*q2*p3 (exponents: [1,1,0,0,0,1])
    k = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, degree, PSI, CLMO)
    poly_to_eliminate[idx] = 2.0
    
    # Set up eta vector (eigenvalues)
    eta = np.array([0.5, 1.2j, 0.8j], dtype=np.complex128)
    
    # Call the function to solve the homological equation
    result = _solve_homological_equation(poly_to_eliminate, degree, eta, PSI, CLMO)
    
    # Manually calculate what the result should be
    # For term 2*q1*q2*p3:
    # kq = (1,1,0), kp = (0,0,1)
    # kp - kq = (-1,-1,1)
    # denominator = (-1)*0.5 + (-1)*1.2j + (1)*0.8j = -0.5 - 0.4j
    # coefficient = -2/(-0.5-0.4j) ≈ 2.94 - 2.35j
    expected_coeff = -2.0 / (-0.5 - 0.4j)
    
    # Create expected result with the calculated coefficient
    expected = np.zeros_like(poly_to_eliminate)
    expected[idx] = expected_coeff
    
    # Since we're comparing complex values, use np.testing.assert_allclose
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    
    # Test multiple terms
    # Create another polynomial with multiple terms to eliminate
    poly_multi = np.zeros(PSI[N_VARS, degree], dtype=np.complex128)
    
    # Term 1: 2*q1*q2*p3 (exponents: [1,1,0,0,0,1])
    k1 = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx1 = encode_multiindex(k1, degree, PSI, CLMO)
    poly_multi[idx1] = 2.0
    
    # Term 2: 3*q1^2*p3 (exponents: [2,0,0,0,0,1])
    k2 = np.array([2, 0, 0, 0, 0, 1], dtype=np.int64)
    idx2 = encode_multiindex(k2, degree, PSI, CLMO)
    poly_multi[idx2] = 3.0
    
    # Solve homological equation for multiple terms
    result_multi = _solve_homological_equation(poly_multi, degree, eta, PSI, CLMO)
    
    # Expected result:
    expected_multi = np.zeros_like(poly_multi)
    
    # For term 2*q1*q2*p3:
    # Same as before
    expected_multi[idx1] = expected_coeff
    
    # For term 3*q1^2*p3:
    # kq = (2,0,0), kp = (0,0,1)
    # kp - kq = (-2,0,1)
    # denominator = (-2)*0.5 + (0)*1.2j + (1)*0.8j = -1.0 + 0.8j
    # coefficient = -3/(-1.0+0.8j)
    denom2 = (-2.0) * 0.5 + 0 * 1.2j + 1 * 0.8j
    expected_coeff2 = -3.0 / denom2
    expected_multi[idx2] = expected_coeff2
    
    # Verify the result for multiple terms
    np.testing.assert_allclose(result_multi, expected_multi, rtol=1e-10, atol=1e-10)
    
    # Test zero divisor case (should handle it gracefully)
    poly_zero_div = np.zeros(PSI[N_VARS, 2], dtype=np.complex128)
    
    # Term q1*p1 (exponents: [1,0,0,1,0,0])
    # This will give denom = 0 because kp-kq = (0,0,0)
    k_zero = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)
    idx_zero = encode_multiindex(k_zero, 2, PSI, CLMO)
    poly_zero_div[idx_zero] = 5.0
    
    result_zero_div = _solve_homological_equation(poly_zero_div, 2, eta, PSI, CLMO)
    
    # The function should handle the zero divisor case by leaving the coefficient as 0
    expected_zero_div = np.zeros_like(poly_zero_div)
    
    # Verify the zero divisor case
    np.testing.assert_allclose(result_zero_div, expected_zero_div)

def test_apply_lie_transform():
    """Test applying the Lie transform to a polynomial."""
    # Create an initial Hamiltonian with terms up to degree 4
    # H = q1*p1 + (q1^2 + q2^2 + p1^2 + p2^2)/2 + q1^3
    H_coeffs = []
    
    # Degree 0: constant term (none)
    H_coeffs.append(np.zeros(PSI[N_VARS, 0], dtype=np.complex128))
    
    # Degree 1: linear term (none)
    H_coeffs.append(np.zeros(PSI[N_VARS, 1], dtype=np.complex128))
    
    # Degree 2: quadratic term - q1*p1 + (q1^2 + q2^2 + p1^2 + p2^2)/2
    degree2 = np.zeros(PSI[N_VARS, 2], dtype=np.complex128)
    
    # q1*p1 (exponents: [1,0,0,1,0,0])
    k = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 1.0
    
    # q1^2/2 (exponents: [2,0,0,0,0,0])
    k = np.array([2, 0, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 0.5
    
    # q2^2/2 (exponents: [0,2,0,0,0,0])
    k = np.array([0, 2, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 0.5
    
    # p1^2/2 (exponents: [0,0,0,2,0,0])
    k = np.array([0, 0, 0, 2, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 0.5
    
    # p2^2/2 (exponents: [0,0,0,0,2,0])
    k = np.array([0, 0, 0, 0, 2, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = 0.5
    
    H_coeffs.append(degree2)
    
    # Degree 3: cubic term - q1^3
    degree3 = np.zeros(PSI[N_VARS, 3], dtype=np.complex128)
    
    # q1^3 (exponents: [3,0,0,0,0,0])
    k = np.array([3, 0, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 3, PSI, CLMO)
    degree3[idx] = 1.0
    
    H_coeffs.append(degree3)
    
    # Degree 4 (empty, but will be filled by the transform)
    H_coeffs.append(np.zeros(PSI[N_VARS, 4], dtype=np.complex128))
    
    # Create a generating function G_n
    # G_n = q1^2*p1  (degree 3)
    deg_G = 3
    G_n = np.zeros(PSI[N_VARS, deg_G], dtype=np.complex128)
    
    # q1^2*p1 (exponents: [2,0,0,1,0,0])
    k = np.array([2, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, deg_G, PSI, CLMO)
    G_n[idx] = 2.0
    
    # Apply the Lie transform
    max_degree = 4
    transformed_H = _apply_lie_transform(H_coeffs, G_n, deg_G, max_degree, PSI, CLMO)
    
    # Verify that the returned list has the correct length
    assert len(transformed_H) == max_degree + 1
    
    # Verify the quadratic part remains unchanged
    np.testing.assert_allclose(transformed_H[2], H_coeffs[2], rtol=1e-10, atol=1e-10)
    
    # Verify the cubic part has changed
    assert not np.array_equal(transformed_H[3], H_coeffs[3])
    
    # Verify that degree 4 now has some non-zero terms
    assert np.any(transformed_H[4] != 0)
    
    # Verify that the transformation involves adding a Poisson bracket term
    # {H_2, G_3} affects H_3
    pb_result = poisson(H_coeffs[2], 2, G_n, 3, PSI, CLMO)
    
    # The transformed H_3 should be H_3 + {H_2, G_3}
    expected_H3 = H_coeffs[3].copy()
    expected_H3 += pb_result
    
    np.testing.assert_allclose(transformed_H[3], expected_H3, rtol=1e-10, atol=1e-10)

class MockLibrationPoint:
    """Mock class to simulate a LibrationPoint with linear_modes method."""
    def __init__(self, lam, om1, om2):
        self.lam = lam
        self.om1 = om1
        self.om2 = om2
        
    def linear_modes(self):
        return self.lam, self.om1, self.om2

def test_lie_transform():
    """Test the full lie_transform function."""
    # Create a mock libration point with eigenvalues
    lam = 0.5
    om1 = 1.2
    om2 = 0.8
    mock_point = MockLibrationPoint(lam, om1, om2)
    
    # Create an initial Hamiltonian with terms up to degree 4
    # H = q1*p1 + (q1^2 + q2^2 + p1^2 + p2^2)/2 + q1^3 + q1*q2*p3
    H_coeffs = []
    
    # Degree 0: constant term (none)
    H_coeffs.append(np.zeros(PSI[N_VARS, 0], dtype=np.complex128))
    
    # Degree 1: linear term (none)
    H_coeffs.append(np.zeros(PSI[N_VARS, 1], dtype=np.complex128))
    
    # Degree 2: quadratic term - q1*p1 + (q1^2 + q2^2 + p1^2 + p2^2)/2
    degree2 = np.zeros(PSI[N_VARS, 2], dtype=np.complex128)
    
    # q1*p1 (exponents: [1,0,0,1,0,0])
    k = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = lam
    
    # q2*p2 (exponents: [0,1,0,0,1,0])
    k = np.array([0, 1, 0, 0, 1, 0], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = complex(0, om1)
    
    # q3*p3 (exponents: [0,0,1,0,0,1])
    k = np.array([0, 0, 1, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, 2, PSI, CLMO)
    degree2[idx] = complex(0, om2)
    
    H_coeffs.append(degree2)
    
    # Degree 3: cubic term with a term to eliminate
    degree3 = np.zeros(PSI[N_VARS, 3], dtype=np.complex128)
    
    # q1^3 (exponents: [3,0,0,0,0,0])
    k = np.array([3, 0, 0, 0, 0, 0], dtype=np.int64)
    idx = encode_multiindex(k, 3, PSI, CLMO)
    degree3[idx] = 1.0
    
    # q1*q2*p3 (exponents: [1,1,0,0,0,1]) - this term should be eliminated
    k = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, 3, PSI, CLMO)
    degree3[idx] = 2.0
    
    H_coeffs.append(degree3)
    
    # Degree 4 (empty, but will be filled by the transform)
    H_coeffs.append(np.zeros(PSI[N_VARS, 4], dtype=np.complex128))
    
    # Convert to a List for Numba compatibility
    H_init_coeffs = List()
    for h in H_coeffs:
        H_init_coeffs.append(h.copy())
    
    # Apply the Lie transform
    max_degree = 4
    H_trans, G_total = lie_transform(mock_point, H_init_coeffs, PSI, CLMO, max_degree)
    
    # Verify results have the correct length
    assert len(H_trans) == len(H_init_coeffs)
    assert len(G_total) == max_degree + 1
    
    # The quadratic part should remain unchanged
    np.testing.assert_allclose(H_trans[2], H_init_coeffs[2], rtol=1e-10, atol=1e-10)
    
    # The cubic part should be changed - specifically, q1*q2*p3 should be eliminated
    assert not np.array_equal(H_trans[3], H_init_coeffs[3])
    
    # Check that the term to eliminate is gone or significantly reduced
    k = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
    idx = encode_multiindex(k, 3, PSI, CLMO)
    assert abs(H_trans[3][idx]) < 1e-10
    
    # The generating function G_3 should be non-zero
    assert np.any(G_total[3] != 0)
    
    # Check that G_3 contains a term for q1*q2*p3
    # The coefficient should match what we expect from _solve_homological_equation
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)
    denom = (-1)*lam + (-1)*(1j*om1) + (1)*(1j*om2)
    expected_coeff = -2.0 / denom
    
    # Compare the coefficient in G_3 with our expected value
    assert abs(G_total[3][idx] - expected_coeff) < 1e-10
