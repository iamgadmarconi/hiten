import numpy as np
import pytest
from numba import types
from numba.typed import Dict, List

from hiten.algorithms.bifurcation.transforms import (
    _actionangle2realcenter, _elliptic_monomial_to_series,
    _hyperbolic_monomial, _realcenter2actionangle)
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.polynomial.fourier import (_encode_fourier_index,
                                                 _init_fourier_tables,
                                                 _make_fourier_poly)
from hiten.algorithms.polynomial.operations import (_polynomial_add_inplace,
                                                    _polynomial_multiply,
                                                    _polynomial_power,
                                                    _polynomial_variables_list,
                                                    _polynomial_zero_list)


def _build_sample_polynomial(max_deg_real: int):
    """Return polynomial H = q1*p1 + q2**2 + p2**2 and its lookup tables."""
    psi, clmo = _init_index_tables(max_deg_real)
    encode = _create_encode_dict_from_clmo(clmo)

    var_polys = _polynomial_variables_list(max_deg_real, psi, clmo, encode)
    q1_poly, q2_poly, p1_poly, p2_poly = (
        var_polys[0],
        var_polys[1],
        var_polys[3],
        var_polys[4],
    )

    term1 = _polynomial_multiply(q1_poly, p1_poly, max_deg_real, psi, clmo, encode)
    term2 = _polynomial_power(q2_poly, 2, max_deg_real, psi, clmo, encode)
    term3 = _polynomial_power(p2_poly, 2, max_deg_real, psi, clmo, encode)

    poly = _polynomial_zero_list(max_deg_real, psi)
    _polynomial_add_inplace(poly, term1)
    _polynomial_add_inplace(poly, term2)
    _polynomial_add_inplace(poly, term3)
    return poly, psi, clmo


def _polys_allclose(poly_a, poly_b, atol=1e-12):
    """Check element-wise closeness of two coefficient lists."""
    max_deg = min(len(poly_a), len(poly_b))
    for d in range(max_deg):
        if not np.allclose(poly_a[d], poly_b[d], atol=atol):
            return False
    return True


def test_hyperbolic_monomial():
    """Test the hyperbolic monomial transformation."""
    # Test q
    I_pow, k, coeff = _hyperbolic_monomial(1, 0)
    assert I_pow == 1
    assert k == 1
    assert np.isclose(coeff, 1.0)

    # Test p
    I_pow, k, coeff = _hyperbolic_monomial(0, 1)
    assert I_pow == 1
    assert k == -1
    assert np.isclose(coeff, 1.0)

    # Test q*p
    I_pow, k, coeff = _hyperbolic_monomial(1, 1)
    assert I_pow == 2
    assert k == 0
    assert np.isclose(coeff, 1.0)

    # Test q^2*p^3
    I_pow, k, coeff = _hyperbolic_monomial(2, 3)
    assert I_pow == 5
    assert k == -1
    assert np.isclose(coeff, 1.0)

    # Test constant term
    I_pow, k, coeff = _hyperbolic_monomial(0, 0)
    assert I_pow == 0
    assert k == 0
    assert np.isclose(coeff, 1.0)


def test_elliptic_monomial_to_series_q():
    """Test the elliptic series for a single q."""
    I_pow, coeff_dict = _elliptic_monomial_to_series(q_pow=1, p_pow=0)
    assert I_pow == 0
    # Expected: sqrt(2)*sin(theta) = -i/sqrt(2) * (exp(i*theta) - exp(-i*theta))
    # k=1: -i/sqrt(2), k=-1: i/sqrt(2)
    expected = Dict.empty(key_type=types.int64, value_type=types.complex128)
    expected[1] = -1j / np.sqrt(2)
    expected[-1] = 1j / np.sqrt(2)
    assert coeff_dict.keys() == expected.keys()
    for k in expected:
        assert np.isclose(coeff_dict[k], expected[k])


def test_elliptic_monomial_to_series_p():
    """Test the elliptic series for a single p."""
    I_pow, coeff_dict = _elliptic_monomial_to_series(q_pow=0, p_pow=1)
    assert I_pow == 0
    # Expected: sqrt(2)*cos(theta) = 1/sqrt(2) * (exp(i*theta) + exp(-i*theta))
    # k=1: 1/sqrt(2), k=-1: 1/sqrt(2)
    expected = Dict.empty(key_type=types.int64, value_type=types.complex128)
    expected[1] = 1 / np.sqrt(2)
    expected[-1] = 1 / np.sqrt(2)
    assert coeff_dict.keys() == expected.keys()
    for k in expected:
        assert np.isclose(coeff_dict[k], expected[k])


def test_elliptic_monomial_to_series_q2():
    """Test the elliptic series for q^2."""
    I_pow, coeff_dict = _elliptic_monomial_to_series(q_pow=2, p_pow=0)
    assert I_pow == 1
    # Expected: 2*I*sin^2(theta) = I * (1 - 0.5*(exp(2it) + exp(-2it)))
    # k=0: 1, k=2: -0.5, k=-2: -0.5
    expected = Dict.empty(key_type=types.int64, value_type=types.complex128)
    expected[0] = 1.0
    expected[2] = -0.5
    expected[-2] = -0.5
    assert coeff_dict.keys() == expected.keys()
    for k in expected:
        assert np.isclose(coeff_dict[k], expected[k])


def test_elliptic_monomial_to_series_p2():
    """Test the elliptic series for p^2."""
    I_pow, coeff_dict = _elliptic_monomial_to_series(q_pow=0, p_pow=2)
    assert I_pow == 1
    # Expected: 2*I*cos^2(theta) = I * (1 + 0.5*(exp(2it) + exp(-2it)))
    # k=0: 1, k=2: 0.5, k=-2: 0.5
    expected = Dict.empty(key_type=types.int64, value_type=types.complex128)
    expected[0] = 1.0
    expected[2] = 0.5
    expected[-2] = 0.5
    assert coeff_dict.keys() == expected.keys()
    for k in expected:
        assert np.isclose(coeff_dict[k], expected[k])


def test_elliptic_monomial_to_series_qp():
    """Test the elliptic series for q*p."""
    I_pow, coeff_dict = _elliptic_monomial_to_series(q_pow=1, p_pow=1)
    assert I_pow == 1
    # Expected: 2*I*sin(t)cos(t) = I*sin(2t) = I*(-i/2)*(exp(2it)-exp(-2it))
    # k=2: -i/2, k=-2: i/2
    expected = Dict.empty(key_type=types.int64, value_type=types.complex128)
    expected[2] = -0.5j
    expected[-2] = 0.5j
    assert coeff_dict.keys() == expected.keys()
    for k in expected:
        assert np.isclose(coeff_dict[k], expected[k])


def test_realcenter2actionangle():

    poly_nf_real, psi, clmo = _build_sample_polynomial(2)
    # 3. Perform the transformation
    fourier_coeffs, psiF, clmoF, encodeF = _realcenter2actionangle(poly_nf_real, clmo)

    # 4. Check the results. Expect H_fourier = I₁² + 2*I₂
    # This corresponds to two non-zero terms.

    # Term 1: 2*I₂ (degree 1 in actions)
    # n=(0,1,0), k=(0,0,0) -> coeff = 2.0
    deg1_coeffs = fourier_coeffs[1]
    assert np.count_nonzero(deg1_coeffs) == 1
    posF_I2 = _encode_fourier_index((0, 1, 0, 0, 0, 0), 1, encodeF)
    assert posF_I2 != -1
    assert np.isclose(deg1_coeffs[posF_I2], 2.0)

    # Term 2: I₁² (degree 2 in actions)
    # n=(2,0,0), k=(0,0,0) -> coeff = 1.0
    deg2_coeffs = fourier_coeffs[2]
    assert np.count_nonzero(deg2_coeffs) == 1
    posF_I1_sq = _encode_fourier_index((2, 0, 0, 0, 0, 0), 2, encodeF)
    assert posF_I1_sq != -1
    assert np.isclose(deg2_coeffs[posF_I1_sq], 1.0)

    # Check that all other coefficient blocks are zero
    assert np.count_nonzero(fourier_coeffs[0]) == 0


def test_realcenter2actionangle_odd_power_rejection():
    max_deg_real = 1
    psi, clmo = _init_index_tables(max_deg_real)
    encode_list = _create_encode_dict_from_clmo(clmo)

    # 2. Construct H = p'₂ (degree 1, odd power in elliptic pair 2)
    var_polys = _polynomial_variables_list(max_deg_real, psi, clmo, encode_list)
    poly_nf_real = var_polys[4]  # p'₂

    # 3. Perform the transformation
    fourier_coeffs, _, _, _ = _realcenter2actionangle(poly_nf_real, clmo)

    # 4. Check that the output is zero, as the term should be skipped
    for d in range(len(fourier_coeffs)):
        assert np.count_nonzero(fourier_coeffs[d]) == 0


def test_realcenter2actionangle_advanced_coupling():
    max_deg_real = 5
    psi, clmo = _init_index_tables(max_deg_real)
    encode_list = _create_encode_dict_from_clmo(clmo)

    # 2. Construct H = q'₁ * (q'₂)² * (p'₃)²
    var_polys = _polynomial_variables_list(max_deg_real, psi, clmo, encode_list)
    q1_poly, q2_poly, p3_poly = var_polys[0], var_polys[1], var_polys[5]

    q2_sq = _polynomial_power(q2_poly, 2, max_deg_real, psi, clmo, encode_list)
    p3_sq = _polynomial_power(p3_poly, 2, max_deg_real, psi, clmo, encode_list)
    term1 = _polynomial_multiply(q1_poly, q2_sq, max_deg_real, psi, clmo, encode_list)
    poly_nf_real = _polynomial_multiply(term1, p3_sq, max_deg_real, psi, clmo, encode_list)

    # 3. Perform the transformation
    fourier_coeffs, _, _, encodeF = _realcenter2actionangle(poly_nf_real, clmo)

    # 4. Check the results
    # The result should only have terms of action degree 3 (1+1+1)
    # and should have 9 non-zero terms in total.
    assert np.count_nonzero(fourier_coeffs[0]) == 0
    assert np.count_nonzero(fourier_coeffs[1]) == 0
    assert np.count_nonzero(fourier_coeffs[2]) == 0
    assert np.count_nonzero(fourier_coeffs[3]) == 9
    assert np.count_nonzero(fourier_coeffs[4]) == 0
    assert np.count_nonzero(fourier_coeffs[5]) == 0

    # Check all 9 coefficients explicitly
    deg3_coeffs = fourier_coeffs[3]
    n_vec = (1, 1, 1)  # I1_pow, I2_pow, I3_pow

    # k=(1,0,0): coeff = 1 * 1 * 1 = 1
    pos = _encode_fourier_index(n_vec + (1, 0, 0), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], 1.0)
    # k=(1,0,2): coeff = 1 * 1 * 0.5 = 0.5
    pos = _encode_fourier_index(n_vec + (1, 0, 2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], 0.5)
    # k=(1,0,-2): coeff = 1 * 1 * 0.5 = 0.5
    pos = _encode_fourier_index(n_vec + (1, 0, -2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], 0.5)
    # k=(1,2,0): coeff = 1 * -0.5 * 1 = -0.5
    pos = _encode_fourier_index(n_vec + (1, 2, 0), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.5)
    # k=(1,-2,0): coeff = 1 * -0.5 * 1 = -0.5
    pos = _encode_fourier_index(n_vec + (1, -2, 0), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.5)
    # k=(1,2,2): coeff = 1 * -0.5 * 0.5 = -0.25
    pos = _encode_fourier_index(n_vec + (1, 2, 2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.25)
    # k=(1,2,-2): coeff = 1 * -0.5 * 0.5 = -0.25
    pos = _encode_fourier_index(n_vec + (1, 2, -2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.25)
    # k=(1,-2,2): coeff = 1 * -0.5 * 0.5 = -0.25
    pos = _encode_fourier_index(n_vec + (1, -2, 2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.25)
    # k=(1,-2,-2): coeff = 1 * -0.5 * 0.5 = -0.25
    pos = _encode_fourier_index(n_vec + (1, -2, -2), 3, encodeF)
    assert np.isclose(deg3_coeffs[pos], -0.25)


def test_actionangle2realcenter_roundtrip():
    """Forward then inverse transform reproduces original polynomial."""
    max_deg_real = 2
    poly_orig, _, clmo = _build_sample_polynomial(max_deg_real)

    fourier_coeffs, _, clmoF, _ = _realcenter2actionangle(poly_orig, clmo)
    poly_back, _, _, _ = _actionangle2realcenter(fourier_coeffs, clmoF)

    assert _polys_allclose(poly_orig, poly_back)


def test_actionangle2realcenter_zero_series():
    """Zero Fourier series maps to zero real polynomial."""
    max_deg_F = 3
    k_max = 0
    psiF, clmoF = _init_fourier_tables(max_deg_F, k_max)

    fourier_coeffs = List()
    for d in range(max_deg_F + 1):
        fourier_coeffs.append(_make_fourier_poly(d, psiF))

    poly_back, _, _, _ = _actionangle2realcenter(fourier_coeffs, clmoF)

    for d in range(len(poly_back)):
        assert np.count_nonzero(poly_back[d]) == 0


def test_actionangle2realcenter_advanced_coupling_roundtrip():
    """Round-trip for H = q1 * (q2)^2 * (p3)^2."""
    max_deg_real = 5
    psi, clmo = _init_index_tables(max_deg_real)
    encode = _create_encode_dict_from_clmo(clmo)

    vars_polys = _polynomial_variables_list(max_deg_real, psi, clmo, encode)
    q1_poly, q2_poly, p3_poly = vars_polys[0], vars_polys[1], vars_polys[5]

    q2_sq = _polynomial_power(q2_poly, 2, max_deg_real, psi, clmo, encode)
    p3_sq = _polynomial_power(p3_poly, 2, max_deg_real, psi, clmo, encode)
    term1 = _polynomial_multiply(q1_poly, q2_sq, max_deg_real, psi, clmo, encode)
    poly_orig = _polynomial_multiply(term1, p3_sq, max_deg_real, psi, clmo, encode)

    fourier_coeffs, _, clmoF, _ = _realcenter2actionangle(poly_orig, clmo)
    poly_back, _, _, _ = _actionangle2realcenter(fourier_coeffs, clmoF)

    assert _polys_allclose(poly_orig, poly_back, atol=1e-9)
