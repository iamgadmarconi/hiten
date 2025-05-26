import numpy as np
import pytest
from numba.typed import List as NumbaList

# Polynomial tools
from algorithms.center.polynomial.base import (
    init_index_tables, make_poly, encode_multiindex, _create_encode_dict_from_clmo
)
from algorithms.variables import N_VARS # Expected to be 6 for 3 DOF

# CPU Poincare step implementation (from map.py)
from algorithms.center.poincare.map import _poincare_step_jit, N_SYMPLECTIC_DOF

# CUDA Poincare step implementation (from cuda/step.py)
from algorithms.center.poincare.cuda.step import PoincareMapCUDA

# Constants for the test
N_DOF_TEST = 3 
TEST_MAX_DEGREE_JACOBIAN_COMPONENT = 2 # Max degree of dH/dx_i polynomials

# Initialize PSI and CLMO tables for the test polynomials
PSI_TEST, CLMO_TEST = init_index_tables(TEST_MAX_DEGREE_JACOBIAN_COMPONENT)
ENCODE_DICT_TEST = _create_encode_dict_from_clmo(CLMO_TEST)


def _create_example_jac_h_for_poincare_test():
    """
    Creates a sample jac_H for testing Poincare step.
    (Identical to the one in test_rk4.py for consistency)
    """
    if N_VARS != 6:
        raise ValueError(f"This test expects N_VARS=6, but got {N_VARS}")

    jac_H_list = []
    max_deg = TEST_MAX_DEGREE_JACOBIAN_COMPONENT

    # dH/dq0 = x3^2
    poly_dH_dq0 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_p0sq = np.array([0,0,0,2,0,0], dtype=np.int64)
    idx_p0sq = encode_multiindex(k_p0sq, 2, ENCODE_DICT_TEST)
    if idx_p0sq != -1: poly_dH_dq0[2][idx_p0sq] = 1.0
    jac_H_list.append(poly_dH_dq0)

    # dH/dq1 = 2*x1
    poly_dH_dq1 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_2x1 = np.array([0,1,0,0,0,0], dtype=np.int64)
    idx_2x1 = encode_multiindex(k_2x1, 1, ENCODE_DICT_TEST)
    if idx_2x1 != -1: poly_dH_dq1[1][idx_2x1] = 2.0
    jac_H_list.append(poly_dH_dq1)

    # dH/dq2 = 0
    poly_dH_dq2 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dq2)

    # dH/dp0 = 2*x0*x3
    poly_dH_dp0 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_2x0x3 = np.array([1,0,0,1,0,0], dtype=np.int64)
    idx_2x0x3 = encode_multiindex(k_2x0x3, 2, ENCODE_DICT_TEST)
    if idx_2x0x3 != -1: poly_dH_dp0[2][idx_2x0x3] = 2.0
    jac_H_list.append(poly_dH_dp0)

    # dH/dp1 = 0
    poly_dH_dp1 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dp1)

    # dH/dp2 = 0
    poly_dH_dp2 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dp2)
    
    return jac_H_list


def test_poincare_step_cuda_vs_cpu():
    """
    Tests a single Poincare step (crossing finding) CUDA vs CPU.
    """
    jac_H_py_lists = _create_example_jac_h_for_poincare_test()

    # Initial conditions for Poincare section q3=0
    q2_initial = 0.1
    p2_initial = 0.05
    p3_initial = 0.2 # Needs to be > 0 for a valid start
    
    dt_test = 0.001
    max_steps_test = 10000
    integrator_order_test = 4 # For CPU symplectic integrator if used
    use_symplectic_cpu = False # For _poincare_step_jit, use RK4 for closer comparison to CUDA

    # Prepare inputs for Numba JITted CPU function _poincare_step_jit
    jac_H_numba = NumbaList()
    for poly_comp_list in jac_H_py_lists:
        inner_list_typed = NumbaList()
        for coeff_array in poly_comp_list:
            inner_list_typed.append(coeff_array)
        jac_H_numba.append(inner_list_typed)

    clmo_numba = NumbaList()
    for arr in CLMO_TEST:
        clmo_numba.append(arr)

    # CPU Poincare Step (using RK4 internally as use_symplectic_cpu is False)
    # _poincare_step_jit uses N_SYMPLECTIC_DOF internally, ensure it matches N_DOF_TEST
    if N_SYMPLECTIC_DOF != N_DOF_TEST:
        pytest.skip(f"N_SYMPLECTIC_DOF ({N_SYMPLECTIC_DOF}) != N_DOF_TEST ({N_DOF_TEST}). CPU test may not match.")

    cpu_flag, cpu_q2p, cpu_p2p, cpu_p3p = _poincare_step_jit(
        q2_initial, p2_initial, p3_initial, dt_test, 
        jac_H_numba, clmo_numba, 
        integrator_order_test, max_steps_test, 
        use_symplectic_cpu, N_DOF_TEST 
    )

    # CUDA Poincare Step
    cuda_poincare_map = PoincareMapCUDA(jac_H_py_lists, CLMO_TEST)
    initial_conditions_cuda = np.array([[q2_initial, p2_initial, p3_initial]], dtype=np.float64)
    
    cuda_flags, cuda_crossings = cuda_poincare_map.find_crossings(
        initial_conditions_cuda, dt=dt_test, max_steps=max_steps_test
    )
    
    cuda_flag = cuda_flags[0]
    cuda_q2p, cuda_p2p, cuda_p3p = cuda_crossings[0]

    # Compare results
    assert cpu_flag == cuda_flag, f"Success flags differ: CPU={cpu_flag}, CUDA={cuda_flag}"
    
    if cpu_flag == 1: # Only compare values if a crossing was found by both
        np.testing.assert_allclose(cpu_q2p, cuda_q2p, atol=1e-7, rtol=1e-7, err_msg="q2' values differ")
        np.testing.assert_allclose(cpu_p2p, cuda_p2p, atol=1e-7, rtol=1e-7, err_msg="p2' values differ")
        np.testing.assert_allclose(cpu_p3p, cuda_p3p, atol=1e-7, rtol=1e-7, err_msg="p3' values differ")
