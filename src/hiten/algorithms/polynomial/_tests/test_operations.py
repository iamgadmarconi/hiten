import numpy as np
import pytest
from numba.typed import List

from hiten.algorithms.polynomial.base import (CLMO_GLOBAL, ENCODE_DICT_GLOBAL,
                                               PSI_GLOBAL,
                                               _create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from hiten.algorithms.polynomial.operations import (
    polynomial_add_inplace, polynomial_clean, polynomial_degree,
    polynomial_differentiate, polynomial_evaluate, polynomial_jacobian,
    polynomial_multiply, polynomial_poisson_bracket, polynomial_power,
    polynomial_total_degree, polynomial_variable, polynomial_variables_list,
    polynomial_zero_list)
from hiten.utils.config import N_VARS

TEST_MAX_DEG = 5
PSI, CLMO = init_index_tables(TEST_MAX_DEG)
ENCODE_DICT = _create_encode_dict_from_clmo(CLMO)


def _assert_poly_lists_almost_equal(list1, list2, decimal=7, msg=""):
    assert len(list1) == len(list2), f"Polynomial lists have different lengths. {msg}"
    for d in range(len(list1)):
        np.testing.assert_array_almost_equal(
            list1[d], list2[d], decimal=decimal,
            err_msg=f"Mismatch at degree {d}. {msg}"
        )


def test_polynomial_zero_list():
    poly_list = polynomial_zero_list(PSI_GLOBAL.shape[1]-1, PSI_GLOBAL)
    
    assert len(poly_list) == PSI_GLOBAL.shape[1]
    
    for d in range(PSI_GLOBAL.shape[1]-1):
        assert poly_list[d].shape[0] == PSI_GLOBAL[N_VARS, d]
        assert np.all(poly_list[d] == 0.0)
    
    complex_poly_list = polynomial_zero_list(PSI_GLOBAL.shape[1]-1, PSI_GLOBAL)
    
    for d in range(PSI_GLOBAL.shape[1]-1):
        assert complex_poly_list[d].shape[0] == PSI_GLOBAL[N_VARS, d]
        assert np.all(complex_poly_list[d] == 0.0)
        assert complex_poly_list[d].dtype == np.complex128


def test_polynomial_variable():
    max_deg_local = PSI_GLOBAL.shape[1]-1

    for var_idx in range(N_VARS):
        poly = polynomial_variable(var_idx, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
        
        assert len(poly) == max_deg_local + 1
        
        for d in range(max_deg_local + 1):
            if d == 1:
                k = np.zeros(N_VARS, dtype=np.int64)
                k[var_idx] = 1
                idx = encode_multiindex(k, 1, ENCODE_DICT_GLOBAL)
                assert poly[1][idx] == 1.0
                
                tmp = poly[1].copy()
                tmp[idx] = 0.0
                assert np.all(tmp == 0.0)
            else:
                # All other degrees should be zero
                assert np.all(poly[d] == 0.0)


def test_polynomial_variables_list():
    max_deg_local = PSI_GLOBAL.shape[1]-1
    var_polys = polynomial_variables_list(max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    assert len(var_polys) == N_VARS
    
    for i, poly in enumerate(var_polys):
        assert len(poly) == PSI_GLOBAL.shape[1]
        
        k = np.zeros(N_VARS, dtype=np.int64)
        k[i] = 1
        idx = encode_multiindex(k, 1, ENCODE_DICT_GLOBAL)
        assert poly[1][idx] == 1.0


def test_polynomial_add_inplace():
    a = polynomial_zero_list(TEST_MAX_DEG, PSI)
    b = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    for d in range(1, TEST_MAX_DEG + 1):
        for i in range(min(PSI[N_VARS, d], 3)):
            a[d][i] = d + i
            b[d][i] = 2 * d + i
    
    a_copy = [arr.copy() for arr in a]
    
    polynomial_add_inplace(a, b)

    for d in range(TEST_MAX_DEG + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] + b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]
    
    current_a_python_list = [arr.copy() for arr in a_copy]
    a = List()
    for arr in current_a_python_list:
        a.append(arr)
    
    scale = 2.0
    polynomial_add_inplace(a, b, scale)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] + scale * b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]
    
    current_a_python_list_2 = [arr.copy() for arr in a_copy]
    a = List()
    for arr in current_a_python_list_2:
        a.append(arr)
    
    scale = -1.0
    polynomial_add_inplace(a, b, scale)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(PSI[N_VARS, d]):
            if i < 3 and d > 0:
                assert a[d][i] == a_copy[d][i] - b[d][i]
            else:
                assert a[d][i] == a_copy[d][i]


def test_polynomial_multiply():
    max_deg_local = PSI_GLOBAL.shape[1]-1
    x_poly = polynomial_variable(0, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)  # x
    y_poly = polynomial_variable(1, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)  # y
    
    result = polynomial_multiply(x_poly, y_poly, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    assert len(result) == max_deg_local + 1
    
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 1  # x
    k[1] = 1  # y
    idx = encode_multiindex(k, 2, ENCODE_DICT_GLOBAL)
    assert result[2][idx] == 1.0
    
    result[2][idx] = 0.0
    for d in range(max_deg_local + 1):
        assert np.all(result[d] == 0.0)
    
    trunc_deg = 3
    if (PSI_GLOBAL.shape[1]-1) < 2:
        pytest.skip(f"Global TEST_MAX_DEG ({PSI_GLOBAL.shape[1]-1}) too small for base of x_squared/y_squared in truncation test")

    x_squared = polynomial_multiply(x_poly, x_poly, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    y_squared = polynomial_multiply(y_poly, y_poly, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    truncated = polynomial_multiply(x_squared, y_squared, trunc_deg, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    assert len(truncated) == trunc_deg + 1
    
    for d in range(trunc_deg + 1):
        assert np.all(truncated[d] == 0.0)
    
    xy = polynomial_multiply(x_poly, y_poly, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    yx = polynomial_multiply(y_poly, x_poly, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    for d in range(max_deg_local + 1):
        np.testing.assert_array_almost_equal(xy[d], yx[d])


def test_polynomial_power():
    max_deg_local = PSI_GLOBAL.shape[1]-1
    x_poly = polynomial_variable(0, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)  # x
    
    x_pow_0 = polynomial_power(x_poly, 0, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    assert x_pow_0[0][0] == 1.0
    
    x_pow_0[0][0] = 0.0
    for d in range(TEST_MAX_DEG + 1):
        assert np.all(x_pow_0[d] == 0.0)
    
    x_pow_1 = polynomial_power(x_poly, 1, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    for d in range(TEST_MAX_DEG + 1):
        np.testing.assert_array_almost_equal(x_pow_1[d], x_poly[d])
    
    x_pow_2 = polynomial_power(x_poly, 2, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 2
    idx = encode_multiindex(k, 2, ENCODE_DICT_GLOBAL)
    assert x_pow_2[2][idx] == 1.0
    
    x_pow_2[2][idx] = 0.0
    for d in range(TEST_MAX_DEG + 1):
        assert np.all(x_pow_2[d] == 0.0)
    
    trunc_deg = 3
    if (PSI_GLOBAL.shape[1]-1) < 1:
        pytest.skip(f"Global TEST_MAX_DEG ({PSI_GLOBAL.shape[1]-1}) too small for base x_poly of power test")
    
    x_pow_4_trunc = polynomial_power(x_poly, 4, trunc_deg, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
    
    assert len(x_pow_4_trunc) == trunc_deg + 1
    
    for d in range(trunc_deg + 1):
        assert np.all(x_pow_4_trunc[d] == 0.0)
    
    LOCAL_MAX_DEGREE_8 = 8
    PSI_8, CLMO_8 = init_index_tables(LOCAL_MAX_DEGREE_8)

    xy_poly = polynomial_zero_list(LOCAL_MAX_DEGREE_8, PSI_8)

    k_x = np.zeros(N_VARS, dtype=np.int64)
    k_y = np.zeros(N_VARS, dtype=np.int64)
    k_x[0] = 1
    k_y[1] = 1
    idx_x = encode_multiindex(k_x, 1, ENCODE_DICT_GLOBAL)
    idx_y = encode_multiindex(k_y, 1, ENCODE_DICT_GLOBAL)
    xy_poly[1][idx_x] = 1.0
    xy_poly[1][idx_y] = 1.0

    # Compute (x+y)^8
    xy_pow_8 = polynomial_power(xy_poly, 8, LOCAL_MAX_DEGREE_8, PSI_8, CLMO_8, ENCODE_DICT_GLOBAL)

    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 4
    k[1] = 4
    idx = encode_multiindex(k, 8, ENCODE_DICT_GLOBAL)
    assert xy_pow_8[8][idx] == 70.0

    k = np.zeros(N_VARS, dtype=np.int64)
    k[1] = 8
    idx = encode_multiindex(k, 8, ENCODE_DICT_GLOBAL)
    assert xy_pow_8[8][idx] == 1.0

    k = np.zeros(N_VARS, dtype=np.int64)
    k[0] = 8
    idx = encode_multiindex(k, 8, ENCODE_DICT_GLOBAL)
    assert xy_pow_8[8][idx] == 1.0


def test_complex_polynomials():
    """Test operations with complex polynomials"""
    max_deg_local = PSI_GLOBAL.shape[1]-1

    x_plus_iy = polynomial_zero_list(TEST_MAX_DEG, PSI)

    k_x0_var = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0_var, 1, ENCODE_DICT_GLOBAL)
    x_plus_iy[1][idx_x0] = complex(1.0, 1.0)
    
    y_minus_iz = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    k_y_var = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_y = encode_multiindex(k_y_var, 1, ENCODE_DICT_GLOBAL)
    y_minus_iz[1][idx_y] = 1.0
    
    k_z_var = np.array([0,0,1,0,0,0], dtype=np.int64)
    idx_z = encode_multiindex(k_z_var, 1, ENCODE_DICT_GLOBAL)
    y_minus_iz[1][idx_z] = complex(0.0, -1.0)
    
    result_add = polynomial_zero_list(TEST_MAX_DEG, PSI)
    x_plus_iy_copy_for_add_pylist = [arr.copy() for arr in x_plus_iy]
    x_plus_iy_copy_for_add = List()
    for arr in x_plus_iy_copy_for_add_pylist:
        x_plus_iy_copy_for_add.append(arr)

    y_minus_iz_copy_for_add_pylist = [arr.copy() for arr in y_minus_iz]
    y_minus_iz_copy_for_add = List()
    for arr in y_minus_iz_copy_for_add_pylist:
        y_minus_iz_copy_for_add.append(arr)

    polynomial_add_inplace(result_add, x_plus_iy_copy_for_add)
    polynomial_add_inplace(result_add, y_minus_iz_copy_for_add)
    
    assert result_add[1][idx_x0] == complex(1.0, 1.0)
    
    assert result_add[1][idx_y] == 1.0
    
    assert result_add[1][idx_z] == complex(0.0, -1.0)
    
    product = polynomial_multiply(x_plus_iy, y_minus_iz, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL) # Pass ENCODE_DICT_GLOBAL
    
    k_x0y = np.array([1,1,0,0,0,0], dtype=np.int64)
    idx_prod_x0y = encode_multiindex(k_x0y, 2, ENCODE_DICT_GLOBAL)
    assert product[2][idx_prod_x0y] == complex(1.0, 1.0)
    
    k_x0z = np.array([1,0,1,0,0,0], dtype=np.int64)
    idx_prod_x0z = encode_multiindex(k_x0z, 2, ENCODE_DICT_GLOBAL)
    assert product[2][idx_prod_x0z] == complex(1.0, -1.0)

    product[2][idx_prod_x0y] = 0.0
    product[2][idx_prod_x0z] = 0.0
    assert np.all(product[2] == 0.0)

    for d_idx in range(TEST_MAX_DEG + 1):
        if d_idx != 2:
            assert np.all(product[d_idx] == 0.0)


def test_polynomial_multiply_complex_inputs():
    max_deg_local = PSI_GLOBAL.shape[1]-1

    p1 = polynomial_zero_list(TEST_MAX_DEG, PSI)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    k_x1 = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_x0_p1 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    idx_x1_p1 = encode_multiindex(k_x1, 1, ENCODE_DICT_GLOBAL)
    p1[1][idx_x0_p1] = 1.0
    p1[1][idx_x1_p1] = 1.0

    p2 = polynomial_zero_list(TEST_MAX_DEG, PSI)
    k_x2 = np.array([0,0,1,0,0,0], dtype=np.int64)
    idx_x0_p2 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL) # same x0 as above
    idx_x2_p2 = encode_multiindex(k_x2, 1, ENCODE_DICT_GLOBAL)
    p2[1][idx_x0_p2] = 1.0
    p2[1][idx_x2_p2] = -1.0

    result = polynomial_multiply(p1, p2, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL) # Pass ENCODE_DICT_GLOBAL

    assert len(result) == max_deg_local + 1

    k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0_sq = encode_multiindex(k_x0_sq, 2, ENCODE_DICT_GLOBAL)
    assert abs(result[2][idx_x0_sq] - 1.0) < 1e-9

    k_x0x2 = np.array([1,0,1,0,0,0], dtype=np.int64)
    idx_x0x2 = encode_multiindex(k_x0x2, 2, ENCODE_DICT_GLOBAL)
    assert abs(result[2][idx_x0x2] - (-1.0)) < 1e-9
    
    k_x1x0 = np.array([1,1,0,0,0,0], dtype=np.int64) # Standard order is x0 then x1
    idx_x1x0 = encode_multiindex(k_x1x0, 2, ENCODE_DICT_GLOBAL)
    assert abs(result[2][idx_x1x0] - 1.0) < 1e-9

    k_x1x2 = np.array([0,1,1,0,0,0], dtype=np.int64)
    idx_x1x2 = encode_multiindex(k_x1x2, 2, ENCODE_DICT_GLOBAL)
    assert abs(result[2][idx_x1x2] - (-1.0)) < 1e-9
    
    result[2][idx_x0_sq] = 0.0
    result[2][idx_x0x2] = 0.0
    result[2][idx_x1x0] = 0.0
    result[2][idx_x1x2] = 0.0
    assert np.allclose(result[2], 0.0)

    for d in range(max_deg_local + 1):
        if d != 2:
            assert np.allclose(result[d], 0.0)


def test_polynomial_multiply_with_zero_components():
    max_deg_local = PSI_GLOBAL.shape[1]-1
    poly_A = polynomial_variable(0, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL) # Pass ENCODE_DICT_GLOBAL

    poly_B = polynomial_zero_list(max_deg_local, PSI_GLOBAL)
    poly_B[0][0] = 2.0
    k_x1_sq_corrected = np.array([0,1,0,0,0,0], dtype=np.int64)
    k_x1_sq_corrected[1] = 2
    idx_x1_sq = encode_multiindex(k_x1_sq_corrected, 2, ENCODE_DICT_GLOBAL)
    poly_B[2][idx_x1_sq] = 1.0

    result = polynomial_multiply(poly_A, poly_B, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)

    assert len(result) == max_deg_local + 1

    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    assert abs(result[1][idx_x0] - 2.0) < 1e-9
    result[1][idx_x0] = 0.0
    assert np.allclose(result[1], 0.0)

    k_x0_x1sq = np.array([1,2,0,0,0,0], dtype=np.int64)
    idx_x0_x1sq = encode_multiindex(k_x0_x1sq, 3, ENCODE_DICT_GLOBAL)
    assert abs(result[3][idx_x0_x1sq] - 1.0) < 1e-9
    result[3][idx_x0_x1sq] = 0.0
    assert np.allclose(result[3], 0.0)
    
    for d in range(max_deg_local + 1):
        if d not in [1, 3]:
            assert np.allclose(result[d], 0.0)


def test_polynomial_power_complex_base():
    max_deg_local = PSI_GLOBAL.shape[1]-1
    base_poly_complex = polynomial_zero_list(TEST_MAX_DEG, PSI)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    base_poly_complex[1][idx_x0] = complex(1.0, 2.0)

    pow2_result = polynomial_power(base_poly_complex, 2, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL) # Pass ENCODE_DICT_GLOBAL
    
    assert len(pow2_result) == max_deg_local + 1
    k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0_sq = encode_multiindex(k_x0_sq, 2, ENCODE_DICT_GLOBAL)
    
    expected_coeff_pow2 = complex(-3.0, 4.0)
    assert np.isclose(pow2_result[2][idx_x0_sq].real, expected_coeff_pow2.real)
    assert np.isclose(pow2_result[2][idx_x0_sq].imag, expected_coeff_pow2.imag)

    pow2_result[2][idx_x0_sq] = 0j
    assert np.allclose(pow2_result[2], 0j)
    for d in range(max_deg_local + 1):
        if d != 2:
            assert np.allclose(pow2_result[d], 0j)

    pow3_result = polynomial_power(base_poly_complex, 3, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL) # Pass ENCODE_DICT_GLOBAL
    assert len(pow3_result) == max_deg_local + 1
    k_x0_cb = np.array([3,0,0,0,0,0], dtype=np.int64)
    idx_x0_cb = encode_multiindex(k_x0_cb, 3, ENCODE_DICT_GLOBAL)

    expected_coeff_pow3 = complex(-11.0, -2.0)
    assert np.isclose(pow3_result[3][idx_x0_cb].real, expected_coeff_pow3.real)
    assert np.isclose(pow3_result[3][idx_x0_cb].imag, expected_coeff_pow3.imag)

    pow3_result[3][idx_x0_cb] = 0j
    assert np.allclose(pow3_result[3], 0j)
    for d in range(max_deg_local + 1):
        if d != 3:
             assert np.allclose(pow3_result[d], 0j)


def test_polynomial_poisson_antisymmetry():
    """Test antisymmetry: {P,Q} = -{Q,P}"""
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # P = x0
    Q = polynomial_variable(3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Q = p0 (x3)

    PQ = polynomial_poisson_bracket(P, Q, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    QP = polynomial_poisson_bracket(Q, P, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # Compute -{Q, P}
    neg_QP = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(neg_QP, QP, scale=-1.0)

    _assert_poly_lists_almost_equal(PQ, neg_QP, msg="{P,Q} != -{Q,P}")

def test_polynomial_poisson_linearity_first_arg():
    """Test linearity in first argument: {aP+bQ, R} = a{P,R} + b{Q,R}"""
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x0
    Q = polynomial_variable(1, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x1
    R = polynomial_variable(3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # p0 (x3)

    a_scalar, b_scalar = 2.0, 3.0

    # aP
    aP = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aP, P, scale=a_scalar)
    # bQ
    bQ = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(bQ, Q, scale=b_scalar)
    # aP + bQ
    aPbQ = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aPbQ, aP)
    polynomial_add_inplace(aPbQ, bQ)

    # LHS: {aP+bQ, R}
    lhs = polynomial_poisson_bracket(aPbQ, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # RHS: a{P,R} + b{Q,R}
    PR = polynomial_poisson_bracket(P, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    QR = polynomial_poisson_bracket(Q, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    
    aPR = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aPR, PR, scale=a_scalar)
    bQR = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(bQR, QR, scale=b_scalar)

    rhs = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(rhs, aPR)
    polynomial_add_inplace(rhs, bQR)

    _assert_poly_lists_almost_equal(lhs, rhs, msg="{aP+bQ, R} != a{P,R} + b{Q,R}")

def test_polynomial_poisson_linearity_second_arg():
    """Test linearity in second argument: {P, aQ+bR} = a{P,Q} + b{P,R}"""
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x0
    Q = polynomial_variable(3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # p0 (x3)
    R = polynomial_variable(4, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # p1 (x4)
    
    a_scalar, b_scalar = 2.0, 3.0

    # aQ
    aQ = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aQ, Q, scale=a_scalar)
    # bR
    bR = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(bR, R, scale=b_scalar)
    # aQ + bR
    aQbR = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aQbR, aQ)
    polynomial_add_inplace(aQbR, bR)

    # LHS: {P, aQ+bR}
    lhs = polynomial_poisson_bracket(P, aQbR, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # RHS: a{P,Q} + b{P,R}
    PQ = polynomial_poisson_bracket(P, Q, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    PR = polynomial_poisson_bracket(P, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    aPQ = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(aPQ, PQ, scale=a_scalar)
    bPR = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(bPR, PR, scale=b_scalar)

    rhs = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(rhs, aPQ)
    polynomial_add_inplace(rhs, bPR)

    _assert_poly_lists_almost_equal(lhs, rhs, msg="{P, aQ+bR} != a{P,Q} + b{P,R}")

def test_polynomial_poisson_jacobi_identity():
    """Test Jacobi identity: {P,{Q,R}} + {Q,{R,P}} + {R,{P,Q}} = 0"""
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x0
    Q = polynomial_variable(3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # p0 (x3)
    R = polynomial_variable(1, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x1

    # {Q,R}
    QR = polynomial_poisson_bracket(Q, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # {P,{Q,R}}
    P_QR = polynomial_poisson_bracket(P, QR, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # {R,P}
    RP = polynomial_poisson_bracket(R, P, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # {Q,{R,P}}
    Q_RP = polynomial_poisson_bracket(Q, RP, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # {P,Q}
    PQ = polynomial_poisson_bracket(P, Q, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # {R,{P,Q}}
    R_PQ = polynomial_poisson_bracket(R, PQ, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    sum_jacobi = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(sum_jacobi, P_QR)
    polynomial_add_inplace(sum_jacobi, Q_RP)
    polynomial_add_inplace(sum_jacobi, R_PQ)

    # Expected result is the zero polynomial
    zero_poly_list = polynomial_zero_list(TEST_MAX_DEG, PSI)
    _assert_poly_lists_almost_equal(sum_jacobi, zero_poly_list, msg="Jacobi identity failed")

def test_polynomial_poisson_leibniz_rule():
    """Test Leibniz rule: {P, Q*R} = {P,Q}*R + Q*{P,R}"""
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x0
    Q = polynomial_variable(1, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x1
    R = polynomial_variable(3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # p0 (x3)

    # Q*R
    QR_prod = polynomial_multiply(Q, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # LHS: {P, Q*R}
    lhs = polynomial_poisson_bracket(P, QR_prod, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # {P,Q}
    PQ_br = polynomial_poisson_bracket(P, Q, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # {P,Q}*R
    PQ_br_R = polynomial_multiply(PQ_br, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    
    # {P,R}
    PR_br = polynomial_poisson_bracket(P, R, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    # Q*{P,R}
    Q_PR_br = polynomial_multiply(Q, PR_br, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT

    # RHS: {P,Q}*R + Q*{P,R}
    rhs = polynomial_zero_list(TEST_MAX_DEG, PSI)
    polynomial_add_inplace(rhs, PQ_br_R)
    polynomial_add_inplace(rhs, Q_PR_br)

    _assert_poly_lists_almost_equal(lhs, rhs, msg="Leibniz rule {P,QR} failed")

def test_polynomial_poisson_constant_property():
    """Test {C, P} = 0 where C is a constant."""
    C_poly = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(C_poly) > 0 and C_poly[0].size > 0:
        C_poly[0][0] = 5.0
    
    P = polynomial_variable(0, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # x0

    # {C,P}
    CP_br = polynomial_poisson_bracket(C_poly, P, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    zero_poly_list = polynomial_zero_list(TEST_MAX_DEG, PSI)
    _assert_poly_lists_almost_equal(CP_br, zero_poly_list, msg="{C,P} != 0 failed")

    # {P,C}
    PC_br = polynomial_poisson_bracket(P, C_poly, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
    _assert_poly_lists_almost_equal(PC_br, zero_poly_list, msg="{P,C} != 0 failed")

def test_polynomial_canonical_relations():
    """Test {q_i,q_j}=0, {p_i,p_j}=0, {q_i,p_j}=delta_ij"""
    q_vars = [polynomial_variable(i, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) for i in range(3)] # x0,x1,x2
    p_vars = [polynomial_variable(i+3, TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) for i in range(3)] # x3,x4,x5 (p0,p1,p2)

    zero_poly_list = polynomial_zero_list(TEST_MAX_DEG, PSI)
    one_poly_list = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(one_poly_list) > 0 and one_poly_list[0].size > 0:
        one_poly_list[0][0] = 1.0 # Constant 1 polynomial

    for i in range(3):
        for j in range(3):
            # {q_i, q_j}
            qi_qj_br = polynomial_poisson_bracket(q_vars[i], q_vars[j], TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
            _assert_poly_lists_almost_equal(qi_qj_br, zero_poly_list, msg=f"{{q{i},q{j}}} != 0")

            # {p_i, p_j}
            pi_pj_br = polynomial_poisson_bracket(p_vars[i], p_vars[j], TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
            _assert_poly_lists_almost_equal(pi_pj_br, zero_poly_list, msg=f"{{p{i},p{j}}} != 0")

            # {q_i, p_j}
            qi_pj_br = polynomial_poisson_bracket(q_vars[i], p_vars[j], TEST_MAX_DEG, PSI, CLMO, ENCODE_DICT) # Pass ENCODE_DICT
            if i == j:
                _assert_poly_lists_almost_equal(qi_pj_br, one_poly_list, msg=f"{{q{i},p{j}}} != 1 (i=j)")
            else:
                _assert_poly_lists_almost_equal(qi_pj_br, zero_poly_list, msg=f"{{q{i},p{j}}} != 0 (i!=j)")

def test_polynomial_clean_basic():
    p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(min(p[d].shape[0], 5)):
            if i % 3 == 0:
                p[d][i] = 1.0  # Normal value
            elif i % 3 == 1:
                p[d][i] = 1e-16  # Very small value (noise)
            else:
                p[d][i] = 1e-8  # Small but significant value
    
    p_copy = [arr.copy() for arr in p]
    
    cleaned = polynomial_clean(p, 1e-10)
    
    for d in range(TEST_MAX_DEG + 1):
        np.testing.assert_array_equal(p[d], p_copy[d])
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(min(cleaned[d].shape[0], 5)):
            if i % 3 == 0:
                assert cleaned[d][i] == 1.0  # Normal values unchanged
            elif i % 3 == 1:
                assert cleaned[d][i] == 0.0  # Very small values zeroed
            else:
                assert cleaned[d][i] == 1e-8  # Small but significant values unchanged

def test_polynomial_clean_complex():
    p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(min(p[d].shape[0], 8)):
            if i % 4 == 0:
                p[d][i] = complex(1.0, 1.0)  # Normal value
            elif i % 4 == 1:
                p[d][i] = complex(1e-16, 0.0)  # Small real part
            elif i % 4 == 2:
                p[d][i] = complex(0.0, 1e-16)  # Small imaginary part
            else:
                p[d][i] = complex(1e-16, 1e-16)  # Both parts small
    
    cleaned = polynomial_clean(p, 1e-10)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(min(cleaned[d].shape[0], 8)):
            if i % 4 == 0:
                assert cleaned[d][i] == complex(1.0, 1.0)  # Normal values unchanged
            else:
                assert cleaned[d][i] == 0.0  # All small values zeroed

def test_polynomial_clean_tolerances():
    """Test polynomial_clean with different tolerance levels"""
    p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    for d in range(TEST_MAX_DEG + 1):
        for i in range(min(p[d].shape[0], 10)):
            p[d][i] = 10**(-i)  # 1, 0.1, 0.01, 0.001, ...
    
    tol_tests = [0.0, 1e-10, 1e-5, 1e-3, 1e-1]
    
    for tol in tol_tests:
        cleaned = polynomial_clean(p, tol)
        
        for d in range(TEST_MAX_DEG + 1):
            for i in range(min(cleaned[d].shape[0], 10)):
                if 10**(-i) <= tol:
                    assert cleaned[d][i] == 0.0
                else:
                    assert cleaned[d][i] == 10**(-i)

def test_polynomial_degree():
    zero_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    assert polynomial_degree(zero_p) == -1, "Test Case 1 Failed: Zero polynomial"

    const_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(const_p) > 0 and const_p[0].size > 0:
        const_p[0][0] = 5.0
    assert polynomial_degree(const_p) == 0, "Test Case 2 Failed: Constant polynomial"

    linear_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if TEST_MAX_DEG >= 1 and len(linear_p) > 1 and linear_p[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
        if idx_x0 != -1 and idx_x0 < linear_p[1].shape[0]:
             linear_p[1][idx_x0] = 2.0
    assert polynomial_degree(linear_p) == 1, "Test Case 3 Failed: Linear polynomial"

    quad_p = polynomial_zero_list(3, PSI) # Max degree 3
    if len(quad_p) > 2 and quad_p[2].size > 0:
        k_x1_sq = np.array([0,2,0,0,0,0], dtype=np.int64)
        idx_x1_sq = encode_multiindex(k_x1_sq, 2, ENCODE_DICT_GLOBAL)
        if idx_x1_sq != -1 and idx_x1_sq < quad_p[2].shape[0]:
            quad_p[2][idx_x1_sq] = 3.0 
    assert polynomial_degree(quad_p) == 2, "Test Case 4 Failed: Quadratic with leading zeros"

    high_deg_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if TEST_MAX_DEG > 0 and len(high_deg_p) > TEST_MAX_DEG and high_deg_p[TEST_MAX_DEG].size > 0:
        high_deg_p[TEST_MAX_DEG][0] = 1.0 
    if TEST_MAX_DEG > 0: # Only assert if we actually set a high degree term
      assert polynomial_degree(high_deg_p) == TEST_MAX_DEG, "Test Case 5 Failed: Highest degree non-zero"
    elif TEST_MAX_DEG == 0 and np.any(high_deg_p[0]):
      assert polynomial_degree(high_deg_p) == 0, "Test Case 5 Failed: Highest degree (0) non-zero"
    else: # TEST_MAX_DEG == 0 and it's zero
      assert polynomial_degree(high_deg_p) == -1, "Test Case 5 Failed: Highest degree (0) zero"

    noisy_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if TEST_MAX_DEG >= 2 and len(noisy_p) > TEST_MAX_DEG and noisy_p[TEST_MAX_DEG].size > 0:
        noisy_p[TEST_MAX_DEG][0] = 1e-18 # Small non-zero noise
    if TEST_MAX_DEG >= 1 and len(noisy_p) > 1 and noisy_p[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
        if idx_x0 != -1 and idx_x0 < noisy_p[1].shape[0]:
            noisy_p[1][idx_x0] = 1.0 # Actual term at degree 1
            
    if TEST_MAX_DEG >=2:
        assert polynomial_degree(noisy_p) == TEST_MAX_DEG, "Test Case 7 Failed: Noisy high degree"
    elif TEST_MAX_DEG == 1:
        assert polynomial_degree(noisy_p) == 1, "Test Case 7 Failed: Noisy high degree (deg 1)"
    elif TEST_MAX_DEG == 0 and np.any(noisy_p[0]):
        assert polynomial_degree(noisy_p) == 0, "Test Case 7 Failed: Noisy high degree (deg 0)"
    else: # TEST_MAX_DEG == 0 and it's zero
        assert polynomial_degree(noisy_p) == -1, "Test Case 7 Failed: Noisy high degree (deg 0 zero)"

    const_in_long_list = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(const_in_long_list) > 0 and const_in_long_list[0].size > 0:
        const_in_long_list[0][0] = 1.0
    assert polynomial_degree(const_in_long_list) == 0, "Test Case 8 Failed: Constant in longer list"

def test_polynomial_differentiate_simple_monomial():
    original_max_deg = 2
    psi_local, clmo_local = init_index_tables(original_max_deg)
    
    p_coeffs = polynomial_zero_list(original_max_deg, psi_local)
    k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0_sq = encode_multiindex(k_x0_sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x0_sq != -1: p_coeffs[2][idx_x0_sq] = 2.0

    var_idx = 0
    expected_deriv_max_deg = original_max_deg - 1 if original_max_deg > 0 else 0
    deriv_psi, deriv_clmo = init_index_tables(expected_deriv_max_deg)

    deriv_coeffs, returned_deriv_max_deg = \
        polynomial_differentiate(p_coeffs, var_idx, original_max_deg, psi_local, clmo_local, deriv_psi, deriv_clmo, ENCODE_DICT_GLOBAL)

    assert returned_deriv_max_deg == expected_deriv_max_deg
    assert len(deriv_coeffs) == expected_deriv_max_deg + 1

    expected_coeffs = polynomial_zero_list(expected_deriv_max_deg, deriv_psi)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL) # Use ENCODE_DICT_GLOBAL for encoding
    if idx_x0 != -1: expected_coeffs[1][idx_x0] = 4.0
    
    _assert_poly_lists_almost_equal(deriv_coeffs, expected_coeffs, msg="dP/dx0 of 2x0^2")

    var_idx_x1 = 1
    expected_deriv_max_deg_x1 = original_max_deg - 1 if original_max_deg > 0 else 0
    deriv_psi_x1, deriv_clmo_x1 = init_index_tables(expected_deriv_max_deg_x1)
    deriv_coeffs_x1, returned_deriv_max_deg_x1 = \
        polynomial_differentiate(p_coeffs, var_idx_x1, original_max_deg, psi_local, clmo_local, deriv_psi_x1, deriv_clmo_x1, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg_x1 == expected_deriv_max_deg_x1
    expected_zero_coeffs = polynomial_zero_list(expected_deriv_max_deg_x1, deriv_psi_x1)
    _assert_poly_lists_almost_equal(deriv_coeffs_x1, expected_zero_coeffs, msg="dP/dx1 of 2x0^2")


def test_polynomial_differentiate_mixed_terms():
    original_max_deg = 2
    psi_local, clmo_local = init_index_tables(original_max_deg)

    p_coeffs = polynomial_zero_list(original_max_deg, psi_local)
    k_x0x1 = np.array([1,1,0,0,0,0], dtype=np.int64)
    idx_x0x1 = encode_multiindex(k_x0x1, 2, ENCODE_DICT_GLOBAL)
    k_x1_sq = np.array([0,2,0,0,0,0], dtype=np.int64)
    idx_x1_sq = encode_multiindex(k_x1_sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x0x1 != -1: p_coeffs[2][idx_x0x1] = 1.0
    if idx_x1_sq != -1: p_coeffs[2][idx_x1_sq] = 3.0

    var_idx_x0 = 0
    expected_deriv_max_deg_x0 = original_max_deg - 1 if original_max_deg > 0 else 0
    deriv_psi_x0, deriv_clmo_x0 = init_index_tables(expected_deriv_max_deg_x0)
    deriv_coeffs_x0, returned_deriv_max_deg_x0 = \
        polynomial_differentiate(p_coeffs, var_idx_x0, original_max_deg, psi_local, clmo_local, deriv_psi_x0, deriv_clmo_x0, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg_x0 == expected_deriv_max_deg_x0
    expected_coeffs_x0 = polynomial_zero_list(expected_deriv_max_deg_x0, deriv_psi_x0)
    k_x1 = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_x1 = encode_multiindex(k_x1, 1, ENCODE_DICT_GLOBAL)
    if idx_x1 != -1: expected_coeffs_x0[1][idx_x1] = 1.0
    _assert_poly_lists_almost_equal(deriv_coeffs_x0, expected_coeffs_x0, msg="dP/dx0 of x0x1 + 3x1^2")

    var_idx_x1 = 1
    expected_deriv_max_deg_x1 = original_max_deg - 1 if original_max_deg > 0 else 0
    deriv_psi_x1, deriv_clmo_x1 = init_index_tables(expected_deriv_max_deg_x1)
    deriv_coeffs_x1, returned_deriv_max_deg_x1 = \
        polynomial_differentiate(p_coeffs, var_idx_x1, original_max_deg, psi_local, clmo_local, deriv_psi_x1, deriv_clmo_x1, ENCODE_DICT_GLOBAL)

    assert returned_deriv_max_deg_x1 == expected_deriv_max_deg_x1
    expected_coeffs_x1 = polynomial_zero_list(expected_deriv_max_deg_x1, deriv_psi_x1)
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    idx_x1 = encode_multiindex(k_x1, 1, ENCODE_DICT_GLOBAL) # k_x1 defined above
    if idx_x0 != -1: expected_coeffs_x1[1][idx_x0] = 1.0
    if idx_x1 != -1: expected_coeffs_x1[1][idx_x1] = 6.0
    _assert_poly_lists_almost_equal(deriv_coeffs_x1, expected_coeffs_x1, msg="dP/dx1 of x0x1 + 3x1^2")


def test_polynomial_differentiate_constant():
    original_max_deg = 0
    psi_local, clmo_local = init_index_tables(original_max_deg)
    p_coeffs = polynomial_zero_list(original_max_deg, psi_local)
    if p_coeffs[0].size > 0: p_coeffs[0][0] = 5.0

    var_idx = 0
    expected_deriv_max_deg = 0 # Derivative of constant is constant (deg 0)
    deriv_psi, deriv_clmo = init_index_tables(expected_deriv_max_deg)
    deriv_coeffs, returned_deriv_max_deg = \
        polynomial_differentiate(p_coeffs, var_idx, original_max_deg, psi_local, clmo_local, deriv_psi, deriv_clmo, ENCODE_DICT_GLOBAL)

    assert returned_deriv_max_deg == expected_deriv_max_deg
    expected_coeffs = polynomial_zero_list(expected_deriv_max_deg, deriv_psi)
    _assert_poly_lists_almost_equal(deriv_coeffs, expected_coeffs, msg="dP/dx of constant 5")


def test_polynomial_differentiate_zero_polynomial():
    original_max_deg = 3
    psi_local, clmo_local = init_index_tables(original_max_deg)
    p_coeffs = polynomial_zero_list(original_max_deg, psi_local)

    var_idx = 0
    expected_deriv_max_deg = original_max_deg - 1
    deriv_psi, deriv_clmo = init_index_tables(expected_deriv_max_deg)
    deriv_coeffs, returned_deriv_max_deg = \
        polynomial_differentiate(p_coeffs, var_idx, original_max_deg, psi_local, clmo_local, deriv_psi, deriv_clmo, ENCODE_DICT_GLOBAL)

    assert returned_deriv_max_deg == expected_deriv_max_deg
    expected_coeffs = polynomial_zero_list(expected_deriv_max_deg, deriv_psi)
    _assert_poly_lists_almost_equal(deriv_coeffs, expected_coeffs, msg="dP/dx of zero polynomial")


def test_polynomial_differentiate_to_zero_constant():
    original_max_deg = 1
    psi_local, clmo_local = init_index_tables(original_max_deg)
    p_coeffs = polynomial_variable(0, original_max_deg, psi_local, clmo_local, ENCODE_DICT_GLOBAL)
    polynomial_add_inplace(p_coeffs, p_coeffs, 1.0) # p_coeffs is now 2*x0

    var_idx = 0
    expected_deriv_max_deg = 0 # 2*x0 -> 2 (deg 0)
    deriv_psi, deriv_clmo = init_index_tables(expected_deriv_max_deg)
    deriv_coeffs, returned_deriv_max_deg = \
        polynomial_differentiate(p_coeffs, var_idx, original_max_deg, psi_local, clmo_local, deriv_psi, deriv_clmo, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg == expected_deriv_max_deg
    expected_coeffs = polynomial_zero_list(expected_deriv_max_deg, deriv_psi)
    if expected_coeffs[0].size > 0: expected_coeffs[0][0] = 2.0
    _assert_poly_lists_almost_equal(deriv_coeffs, expected_coeffs, msg="dP/dx0 of 2x0")


def test_polynomial_differentiate_multiple_vars_complex():
    original_max_deg = 3
    psi_local, clmo_local = init_index_tables(original_max_deg)
    p_coeffs = polynomial_zero_list(original_max_deg, psi_local)
    k_x0sq_x1 = np.array([2,1,0,0,0,0], dtype=np.int64)
    idx_x0sq_x1 = encode_multiindex(k_x0sq_x1, 3, ENCODE_DICT_GLOBAL)
    if idx_x0sq_x1 != -1: p_coeffs[3][idx_x0sq_x1] = complex(1.0, 1.0)
    k_x1_x2sq = np.array([0,1,2,0,0,0], dtype=np.int64)
    idx_x1_x2sq = encode_multiindex(k_x1_x2sq, 3, ENCODE_DICT_GLOBAL)
    if idx_x1_x2sq != -1: p_coeffs[3][idx_x1_x2sq] = complex(2.0, -1.0)

    var_idx_x0 = 0
    expected_deriv_max_deg_x0 = original_max_deg - 1
    deriv_psi_x0, deriv_clmo_x0 = init_index_tables(expected_deriv_max_deg_x0)
    deriv_coeffs_x0, returned_deriv_max_deg_x0 = \
        polynomial_differentiate(p_coeffs, var_idx_x0, original_max_deg, psi_local, clmo_local, deriv_psi_x0, deriv_clmo_x0, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg_x0 == expected_deriv_max_deg_x0
    expected_coeffs_x0 = polynomial_zero_list(expected_deriv_max_deg_x0, deriv_psi_x0)
    k_x0x1 = np.array([1,1,0,0,0,0], dtype=np.int64)
    idx_x0x1_deriv = encode_multiindex(k_x0x1, 2, ENCODE_DICT_GLOBAL)
    if idx_x0x1_deriv != -1: expected_coeffs_x0[2][idx_x0x1_deriv] = complex(2.0, 2.0)
    _assert_poly_lists_almost_equal(deriv_coeffs_x0, expected_coeffs_x0, msg="Complex dP/dx0")

    var_idx_x1 = 1
    expected_deriv_max_deg_x1 = original_max_deg - 1
    deriv_psi_x1, deriv_clmo_x1 = init_index_tables(expected_deriv_max_deg_x1)
    deriv_coeffs_x1, returned_deriv_max_deg_x1 = \
        polynomial_differentiate(p_coeffs, var_idx_x1, original_max_deg, psi_local, clmo_local, deriv_psi_x1, deriv_clmo_x1, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg_x1 == expected_deriv_max_deg_x1
    expected_coeffs_x1 = polynomial_zero_list(expected_deriv_max_deg_x1, deriv_psi_x1)
    k_x0sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0sq_deriv = encode_multiindex(k_x0sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x0sq_deriv != -1: expected_coeffs_x1[2][idx_x0sq_deriv] = complex(1.0, 1.0)
    k_x2sq = np.array([0,0,2,0,0,0], dtype=np.int64)
    idx_x2sq_deriv = encode_multiindex(k_x2sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x2sq_deriv != -1: expected_coeffs_x1[2][idx_x2sq_deriv] = complex(2.0, -1.0)
    _assert_poly_lists_almost_equal(deriv_coeffs_x1, expected_coeffs_x1, msg="Complex dP/dx1")

    var_idx_x2 = 2
    expected_deriv_max_deg_x2 = original_max_deg - 1
    deriv_psi_x2, deriv_clmo_x2 = init_index_tables(expected_deriv_max_deg_x2)
    deriv_coeffs_x2, returned_deriv_max_deg_x2 = \
        polynomial_differentiate(p_coeffs, var_idx_x2, original_max_deg, psi_local, clmo_local, deriv_psi_x2, deriv_clmo_x2, ENCODE_DICT_GLOBAL)
    
    assert returned_deriv_max_deg_x2 == expected_deriv_max_deg_x2
    expected_coeffs_x2 = polynomial_zero_list(expected_deriv_max_deg_x2, deriv_psi_x2)
    k_x1x2 = np.array([0,1,1,0,0,0], dtype=np.int64)
    idx_x1x2_deriv = encode_multiindex(k_x1x2, 2, ENCODE_DICT_GLOBAL)
    if idx_x1x2_deriv != -1: expected_coeffs_x2[2][idx_x1x2_deriv] = complex(4.0, -2.0)
    _assert_poly_lists_almost_equal(deriv_coeffs_x2, expected_coeffs_x2, msg="Complex dP/dx2")

@pytest.mark.parametrize("max_poly_deg", range(TEST_MAX_DEG + 1))
def test_polynomial_evaluate_zero_poly_list(max_poly_deg):
    """Test evaluation of a zero polynomial list."""
    psi_local, clmo_local = init_index_tables(max_poly_deg)
    zero_p_list = polynomial_zero_list(max_poly_deg, psi_local)
    point = np.random.rand(N_VARS) + 1j * np.random.rand(N_VARS)
    
    result = polynomial_evaluate(zero_p_list, point, clmo_local)
    assert np.isclose(result, 0.0 + 0.0j)

def test_polynomial_evaluate_constant_poly():
    """Test evaluation of a constant polynomial: P(x) = 5.0 - 2.0j."""
    max_poly_deg = 2 # Does not matter much for constant
    psi_local, clmo_local = init_index_tables(max_poly_deg)
    const_val = 5.0 - 2.0j
    
    p_list = polynomial_zero_list(max_poly_deg, psi_local)
    if len(p_list) > 0 and p_list[0].size > 0:
        p_list[0][0] = const_val
    
    point = np.random.rand(N_VARS) * 10 # Random point
    point = point.astype(np.complex128)  # Ensure point is complex
    numeric_eval = polynomial_evaluate(p_list, point, clmo_local)
    assert np.isclose(numeric_eval, const_val)

def test_polynomial_evaluate_linear_poly():
    max_poly_deg = 3 
    psi_local, clmo_local = init_index_tables(max_poly_deg)

    p_list = polynomial_zero_list(max_poly_deg, psi_local)
    coeff_x0 = 1.0 + 1.0j
    coeff_x1 = 2.0 - 0.5j

    k_x0 = np.zeros(N_VARS, dtype=np.int64)
    k_x0[0] = 1
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    if idx_x0 != -1 and len(p_list) > 1 and idx_x0 < p_list[1].shape[0]:
        p_list[1][idx_x0] = coeff_x0

    k_x1 = np.zeros(N_VARS, dtype=np.int64)
    k_x1[1] = 1
    idx_x1 = encode_multiindex(k_x1, 1, ENCODE_DICT_GLOBAL)
    if idx_x1 != -1 and len(p_list) > 1 and idx_x1 < p_list[1].shape[0]:
        p_list[1][idx_x1] = coeff_x1

    point = np.array([0.5 - 0.1j, 1.0 + 0.2j] + [0.0]*(N_VARS-2), dtype=np.complex128)
    
    numeric_eval = polynomial_evaluate(p_list, point, clmo_local)
    
    expected_val = coeff_x0 * point[0] + coeff_x1 * point[1]
    assert np.isclose(numeric_eval, expected_val)

def test_polynomial_evaluate_mixed_degree_poly():
    max_poly_deg = 3
    psi_local, clmo_local = init_index_tables(max_poly_deg)

    p_list = polynomial_zero_list(max_poly_deg, psi_local)

    if len(p_list) > 0 and p_list[0].size > 0: p_list[0][0] = 2.0
    coeff_x0 = 1.0 + 1.0j
    k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
    idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
    if idx_x0 != -1 and len(p_list) > 1: p_list[1][idx_x0] = coeff_x0

    coeff_x1sq = 0.5
    k_x1sq = np.array([0,2,0,0,0,0], dtype=np.int64)
    idx_x1sq = encode_multiindex(k_x1sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x1sq != -1 and len(p_list) > 2: p_list[2][idx_x1sq] = coeff_x1sq

    coeff_x0x1x2 = -(1.0 - 0.5j)
    k_x0x1x2 = np.array([1,1,1,0,0,0], dtype=np.int64)
    idx_x0x1x2 = encode_multiindex(k_x0x1x2, 3, ENCODE_DICT_GLOBAL)
    if idx_x0x1x2 != -1 and len(p_list) > 3: p_list[3][idx_x0x1x2] = coeff_x0x1x2

    point = np.array([0.5, -0.2, 1.0] + [0.0]*(N_VARS-3), dtype=np.complex128)
    point += 1j * np.array([0.1, 0.3, -0.1] + [0.0]*(N_VARS-3), dtype=np.complex128)

    numeric_eval = polynomial_evaluate(p_list, point, clmo_local)

def test_polynomial_evaluate_at_origin_list():
    max_poly_deg = TEST_MAX_DEG
    psi_local, clmo_local = init_index_tables(max_poly_deg)
    p_list = polynomial_zero_list(max_poly_deg, psi_local)

    const_term = 3.14 - 2.71j
    if len(p_list) > 0 and p_list[0].size > 0:
        p_list[0][0] = const_term # Set a constant term
    
    if max_poly_deg >= 1 and len(p_list) > 1 and p_list[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0,1,ENCODE_DICT_GLOBAL)
        if idx_x0 !=-1: p_list[1][idx_x0] = 1.0

    point_at_origin = np.zeros(N_VARS, dtype=np.complex128)
    numeric_eval = polynomial_evaluate(p_list, point_at_origin, clmo_local)

    assert np.isclose(numeric_eval, const_term) # Only const term should remain

def test_polynomial_evaluate_empty_parts():
    max_poly_deg = 3
    psi_local, clmo_local = init_index_tables(max_poly_deg)
    
    p_list = polynomial_zero_list(max_poly_deg, psi_local)

    const_val = 5.0
    if len(p_list) > 0 and p_list[0].size > 0: p_list[0][0] = const_val
    
    coeff_x0sq = 2.0
    k_x0sq = np.array([2,0,0,0,0,0], dtype=np.int64)
    idx_x0sq = encode_multiindex(k_x0sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x0sq != -1 and len(p_list) > 2: p_list[2][idx_x0sq] = coeff_x0sq

    point = np.array([0.5+0.1j] + [0.0]*(N_VARS-1), dtype=np.complex128)
    numeric_eval = polynomial_evaluate(p_list, point, clmo_local)

    expected_val = const_val + coeff_x0sq * (point[0]**2)
    assert np.isclose(numeric_eval, expected_val)

def test_polynomial_jacobian():
    original_max_deg_main = 3

    p_coeffs_main = polynomial_zero_list(original_max_deg_main, PSI)

    k_x0sq_x1 = np.array([2, 1, 0, 0, 0, 0], dtype=np.int64)
    idx_x0sq_x1 = encode_multiindex(k_x0sq_x1, 3, ENCODE_DICT_GLOBAL)
    if idx_x0sq_x1 != -1: p_coeffs_main[3][idx_x0sq_x1] = 2.0

    k_x1_x2sq = np.array([0, 1, 2, 0, 0, 0], dtype=np.int64)
    idx_x1_x2sq = encode_multiindex(k_x1_x2sq, 3, ENCODE_DICT_GLOBAL)
    if idx_x1_x2sq != -1: p_coeffs_main[3][idx_x1_x2sq] = complex(1.0, 1.0)

    jacobian_P_main = polynomial_jacobian(p_coeffs_main, original_max_deg_main, PSI, CLMO, ENCODE_DICT_GLOBAL)

    assert len(jacobian_P_main) == N_VARS, "Jacobian should have N_VARS components"

    deriv_max_deg_main = original_max_deg_main - 1 # Should be 2

    expected_dP_dx0 = polynomial_zero_list(deriv_max_deg_main, PSI)
    k_x0_x1 = np.array([1, 1, 0, 0, 0, 0], dtype=np.int64)
    idx_x0_x1 = encode_multiindex(k_x0_x1, 2, ENCODE_DICT_GLOBAL)
    if idx_x0_x1 != -1: expected_dP_dx0[2][idx_x0_x1] = 4.0
    _assert_poly_lists_almost_equal(jacobian_P_main[0], expected_dP_dx0, msg="dP/dx0 mismatch")

    expected_dP_dx1 = polynomial_zero_list(deriv_max_deg_main, PSI)
    k_x0sq = np.array([2, 0, 0, 0, 0, 0], dtype=np.int64)
    idx_x0sq = encode_multiindex(k_x0sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x0sq != -1: expected_dP_dx1[2][idx_x0sq] = 2.0
    k_x2sq = np.array([0, 0, 2, 0, 0, 0], dtype=np.int64)
    idx_x2sq = encode_multiindex(k_x2sq, 2, ENCODE_DICT_GLOBAL)
    if idx_x2sq != -1: expected_dP_dx1[2][idx_x2sq] = complex(1.0, 1.0)
    _assert_poly_lists_almost_equal(jacobian_P_main[1], expected_dP_dx1, msg="dP/dx1 mismatch")

    expected_dP_dx2 = polynomial_zero_list(deriv_max_deg_main, PSI)
    k_x1_x2 = np.array([0, 1, 1, 0, 0, 0], dtype=np.int64)
    idx_x1_x2 = encode_multiindex(k_x1_x2, 2, ENCODE_DICT_GLOBAL)
    if idx_x1_x2 != -1: expected_dP_dx2[2][idx_x1_x2] = complex(2.0, 2.0)
    _assert_poly_lists_almost_equal(jacobian_P_main[2], expected_dP_dx2, msg="dP/dx2 mismatch")

    expected_zero_deriv_main = polynomial_zero_list(deriv_max_deg_main, PSI)
    for i in range(3, N_VARS):
        _assert_poly_lists_almost_equal(jacobian_P_main[i], expected_zero_deriv_main, msg=f"dP/dx{i} mismatch, should be zero")

    original_max_deg_const = 0
    p_coeffs_const = polynomial_zero_list(original_max_deg_const, PSI)
    if p_coeffs_const[0].size > 0: p_coeffs_const[0][0] = 5.0

    jacobian_P_const = polynomial_jacobian(p_coeffs_const, original_max_deg_const, PSI, CLMO, ENCODE_DICT_GLOBAL)
    assert len(jacobian_P_const) == N_VARS
    deriv_max_deg_const = 0 # Max degree of derivative of constant is 0

    expected_zero_deriv_const = polynomial_zero_list(deriv_max_deg_const, PSI)
    for i in range(N_VARS):
        assert len(jacobian_P_const[i]) == deriv_max_deg_const + 1
        _assert_poly_lists_almost_equal(jacobian_P_const[i], expected_zero_deriv_const, msg=f"dP/dx{i} for constant P mismatch")

    original_max_deg_zero = 2
    p_coeffs_zero = polynomial_zero_list(original_max_deg_zero, PSI)

    jacobian_P_zero = polynomial_jacobian(p_coeffs_zero, original_max_deg_zero, PSI, CLMO, ENCODE_DICT_GLOBAL)
    assert len(jacobian_P_zero) == N_VARS
    deriv_max_deg_zero = original_max_deg_zero -1 if original_max_deg_zero >0 else 0 

    expected_zero_deriv_for_zero_poly = polynomial_zero_list(deriv_max_deg_zero, PSI)
    for i in range(N_VARS):
        assert len(jacobian_P_zero[i]) == deriv_max_deg_zero + 1
        _assert_poly_lists_almost_equal(jacobian_P_zero[i], expected_zero_deriv_for_zero_poly, msg=f"dP/dx{i} for zero P mismatch")

def test_polynomial_total_degree():
    zero_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    assert polynomial_total_degree(zero_p, PSI) == -1, "Test Case 1 Failed: Zero polynomial"

    assert polynomial_total_degree(zero_p, PSI) == polynomial_degree(zero_p), \
        "polynomial_total_degree and polynomial_degree should agree for zero polynomial"

    const_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(const_p) > 0 and const_p[0].size > 0:
        const_p[0][0] = 5.0
    assert polynomial_total_degree(const_p, PSI) == 0, "Test Case 2 Failed: Constant polynomial"
    
    assert polynomial_total_degree(const_p, PSI) == polynomial_degree(const_p), \
        "polynomial_total_degree and polynomial_degree should agree for constant polynomial"

    linear_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if TEST_MAX_DEG >= 1 and len(linear_p) > 1 and linear_p[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
        if idx_x0 != -1 and idx_x0 < linear_p[1].shape[0]:
             linear_p[1][idx_x0] = 2.0
    assert polynomial_total_degree(linear_p, PSI) == 1, "Test Case 3 Failed: Linear polynomial"
    
    assert polynomial_total_degree(linear_p, PSI) == polynomial_degree(linear_p), \
        "polynomial_total_degree and polynomial_degree should agree for linear polynomial"

    quad_p = polynomial_zero_list(3, PSI) # Max degree 3
    if len(quad_p) > 2 and quad_p[2].size > 0:
        k_x1_sq = np.array([0,2,0,0,0,0], dtype=np.int64)
        idx_x1_sq = encode_multiindex(k_x1_sq, 2, ENCODE_DICT_GLOBAL)
        if idx_x1_sq != -1 and idx_x1_sq < quad_p[2].shape[0]:
            quad_p[2][idx_x1_sq] = 3.0 
    assert polynomial_total_degree(quad_p, PSI) == 2, "Test Case 4 Failed: Quadratic with leading zeros"
    
    assert polynomial_total_degree(quad_p, PSI) == polynomial_degree(quad_p), \
        "polynomial_total_degree and polynomial_degree should agree for quadratic polynomial"

    multi_deg_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    
    if len(multi_deg_p) > 0 and multi_deg_p[0].size > 0:
        multi_deg_p[0][0] = 1.0
    
    if TEST_MAX_DEG >= 1 and len(multi_deg_p) > 1 and multi_deg_p[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
        if idx_x0 != -1 and idx_x0 < multi_deg_p[1].shape[0]:
            multi_deg_p[1][idx_x0] = 2.0
    
    if TEST_MAX_DEG >= 3 and len(multi_deg_p) > 3 and multi_deg_p[3].size > 0:
        k_x0_x1_x2 = np.array([1,1,1,0,0,0], dtype=np.int64)
        idx_x0_x1_x2 = encode_multiindex(k_x0_x1_x2, 3, ENCODE_DICT_GLOBAL)
        if idx_x0_x1_x2 != -1 and idx_x0_x1_x2 < multi_deg_p[3].shape[0]:
            multi_deg_p[3][idx_x0_x1_x2] = 4.0
    
    expected_degree = min(3, TEST_MAX_DEG) if TEST_MAX_DEG >= 1 else 0
    assert polynomial_total_degree(multi_deg_p, PSI) == expected_degree, \
        "Test Case 5 Failed: Multi-degree polynomial"
    
    assert polynomial_total_degree(multi_deg_p, PSI) == polynomial_degree(multi_deg_p), \
        "polynomial_total_degree and polynomial_degree should agree for multi-degree polynomial"

    if TEST_MAX_DEG > 0:
        high_deg_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
        if len(high_deg_p) > TEST_MAX_DEG and high_deg_p[TEST_MAX_DEG].size > 0:
            high_deg_p[TEST_MAX_DEG][0] = 1.0 
        assert polynomial_total_degree(high_deg_p, PSI) == TEST_MAX_DEG, \
            "Test Case 6 Failed: Highest degree non-zero"
        
        assert polynomial_total_degree(high_deg_p, PSI) == polynomial_degree(high_deg_p), \
            "polynomial_total_degree and polynomial_degree should agree for highest degree polynomial"

    max_deg_local = PSI_GLOBAL.shape[1]-1
    for var_idx in range(min(N_VARS, 3)):
        var_poly = polynomial_variable(var_idx, max_deg_local, PSI_GLOBAL, CLMO_GLOBAL, ENCODE_DICT_GLOBAL)
        assert polynomial_total_degree(var_poly, PSI_GLOBAL) == 1, \
            f"Test Case 7 Failed: Variable x{var_idx} polynomial should have degree 1"
        
        assert polynomial_total_degree(var_poly, PSI_GLOBAL) == polynomial_degree(var_poly), \
            f"polynomial_total_degree and polynomial_degree should agree for variable x{var_idx}"

    complex_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if len(complex_p) > 0 and complex_p[0].size > 0:
        complex_p[0][0] = complex(1.0, 2.0)  # Complex constant
    
    if TEST_MAX_DEG >= 2 and len(complex_p) > 2 and complex_p[2].size > 0:
        k_x0_sq = np.array([2,0,0,0,0,0], dtype=np.int64)
        idx_x0_sq = encode_multiindex(k_x0_sq, 2, ENCODE_DICT_GLOBAL)
        if idx_x0_sq != -1 and idx_x0_sq < complex_p[2].shape[0]:
            complex_p[2][idx_x0_sq] = complex(3.0, -1.0)  # Complex quadratic term
    
    expected_complex_degree = min(2, TEST_MAX_DEG) if TEST_MAX_DEG >= 2 else 0
    assert polynomial_total_degree(complex_p, PSI) == expected_complex_degree, \
        "Test Case 8 Failed: Complex polynomial"
    
    # Should match existing polynomial_degree function
    assert polynomial_total_degree(complex_p, PSI) == polynomial_degree(complex_p), \
        "polynomial_total_degree and polynomial_degree should agree for complex polynomial"

    # Test case 9: Small coefficient polynomial (should still count if non-zero)
    small_coeff_p = polynomial_zero_list(TEST_MAX_DEG, PSI)
    if TEST_MAX_DEG >= 1 and len(small_coeff_p) > 1 and small_coeff_p[1].size > 0:
        k_x0 = np.array([1,0,0,0,0,0], dtype=np.int64)
        idx_x0 = encode_multiindex(k_x0, 1, ENCODE_DICT_GLOBAL)
        if idx_x0 != -1 and idx_x0 < small_coeff_p[1].shape[0]:
            small_coeff_p[1][idx_x0] = 1e-15  # Very small but non-zero
    
    if TEST_MAX_DEG >= 1:
        assert polynomial_total_degree(small_coeff_p, PSI) == 1, \
            "Test Case 9 Failed: Small coefficient polynomial"
        
        # Should match existing polynomial_degree function  
        assert polynomial_total_degree(small_coeff_p, PSI) == polynomial_degree(small_coeff_p), \
            "polynomial_total_degree and polynomial_degree should agree for small coefficient polynomial"
    else:
        assert polynomial_total_degree(small_coeff_p, PSI) == -1, \
            "Test Case 9 Failed: Zero polynomial when TEST_MAX_DEG < 1"
