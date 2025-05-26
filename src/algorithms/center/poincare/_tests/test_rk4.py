import numpy as np
import pytest
from numba.typed import List as NumbaList

# Polynomial tools
from algorithms.center.polynomial.base import (
    init_index_tables, make_poly, encode_multiindex, _create_encode_dict_from_clmo
)
from algorithms.variables import N_VARS # Expected to be 6 for 3 DOF

# CPU RK4 implementation (from map.py)
from algorithms.center.poincare.map import _rk4_step 

# CUDA RK4 implementation (from cuda/rk4.py)
from algorithms.center.poincare.cuda.rk4 import RK4IntegratorCUDA

# Constants for the test
N_DOF_TEST = 3 
# Ensure N_VARS from algorithms.variables is consistent (N_VARS == 2 * N_DOF_TEST)
# For polynomial creation, N_VARS is used by init_index_tables, etc.
# For dynamics, _rk4_step takes n_dof, RK4IntegratorCUDA infers it.

TEST_MAX_DEGREE_JACOBIAN_COMPONENT = 2 # Max degree of dH/dx_i polynomials

# Initialize PSI and CLMO tables for the test polynomials
# These functions use N_VARS internally from algorithms.variables
PSI_TEST, CLMO_TEST = init_index_tables(TEST_MAX_DEGREE_JACOBIAN_COMPONENT)
ENCODE_DICT_TEST = _create_encode_dict_from_clmo(CLMO_TEST)


def _create_example_jac_h_for_rk4_test():
    """
    Creates a sample jac_H for testing RK4.
    Based on H = q0*p0^2 + q1^2. (N_DOF_TEST = 3, N_VARS = 6)
    Variables are mapped as: q0,q1,q2 -> x0,x1,x2 and p0,p1,p2 -> x3,x4,x5.
    
    dH/dq0 = p0^2 = x3^2
    dH/dq1 = 2*q1 = 2*x1
    dH/dq2 = 0
    dH/dp0 = 2*q0*p0 = 2*x0*x3
    dH/dp1 = 0
    dH/dp2 = 0

    Each component of jac_H is a list of coefficient arrays, 
    one for each degree up to TEST_MAX_DEGREE_JACOBIAN_COMPONENT.
    """
    if N_VARS != 6:
        raise ValueError(f"This test's _create_example_jac_h expects N_VARS=6, but got {N_VARS}")

    jac_H_list = []
    max_deg = TEST_MAX_DEGREE_JACOBIAN_COMPONENT

    # dH/dq0 = x3^2 (Degree 2)
    poly_dH_dq0 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_p0sq = np.array([0,0,0,2,0,0], dtype=np.int64) # x3^2
    idx_p0sq = encode_multiindex(k_p0sq, 2, ENCODE_DICT_TEST)
    if idx_p0sq != -1:
         poly_dH_dq0[2][idx_p0sq] = 1.0
    jac_H_list.append(poly_dH_dq0)

    # dH/dq1 = 2*x1 (Degree 1)
    poly_dH_dq1 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_2x1 = np.array([0,1,0,0,0,0], dtype=np.int64) # x1
    idx_2x1 = encode_multiindex(k_2x1, 1, ENCODE_DICT_TEST)
    if idx_2x1 != -1:
        poly_dH_dq1[1][idx_2x1] = 2.0
    jac_H_list.append(poly_dH_dq1)

    # dH/dq2 = 0
    poly_dH_dq2 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dq2)

    # dH/dp0 = 2*x0*x3 (Degree 2)
    poly_dH_dp0 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    k_2x0x3 = np.array([1,0,0,1,0,0], dtype=np.int64) # x0*x3
    idx_2x0x3 = encode_multiindex(k_2x0x3, 2, ENCODE_DICT_TEST)
    if idx_2x0x3 != -1:
        poly_dH_dp0[2][idx_2x0x3] = 2.0
    jac_H_list.append(poly_dH_dp0)

    # dH/dp1 = 0
    poly_dH_dp1 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dp1)

    # dH/dp2 = 0
    poly_dH_dp2 = [make_poly(d, PSI_TEST) for d in range(max_deg + 1)]
    jac_H_list.append(poly_dH_dp2)
    
    return jac_H_list


def test_rk4_cuda_vs_cpu_single_step():
    """
    Tests a single RK4 step CUDA implementation against the CPU equivalent.
    """
    jac_H_test_py_lists = _create_example_jac_h_for_rk4_test()
    
    # Sample state vector [q0,q1,q2,p0,p1,p2] -> [x0,x1,x2,x3,x4,x5]
    initial_state_cpu = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.15], dtype=np.float64)
    dt_test = 0.01

    # Prepare inputs for Numba JITted CPU function _rk4_step
    # Convert Python lists of lists/arrays to Numba typed lists
    jac_H_numba = NumbaList()
    for poly_comp_list in jac_H_test_py_lists:
        inner_list_typed = NumbaList()
        for coeff_array in poly_comp_list:
            inner_list_typed.append(coeff_array)
        jac_H_numba.append(inner_list_typed)

    clmo_numba = NumbaList()
    for arr in CLMO_TEST: # CLMO_TEST is already a list of np.ndarrays
        clmo_numba.append(arr)

    # CPU RK4 step
    cpu_next_state = _rk4_step(initial_state_cpu, dt_test, jac_H_numba, clmo_numba, N_DOF_TEST)

    # CUDA RK4 step
    # RK4IntegratorCUDA expects jac_H as Python list of lists of np.ndarrays, 
    # and clmo as Python list of np.ndarrays.
    cuda_integrator = RK4IntegratorCUDA(jac_H_test_py_lists, CLMO_TEST)
    
    # The 'step' method expects a 2D array of states (n_states, n_vars)
    initial_state_cuda = initial_state_cpu.reshape(1, -1) 
    cuda_next_state_batch = cuda_integrator.step(initial_state_cuda, dt_test)
    cuda_next_state = cuda_next_state_batch[0] # Get the first (and only) state result

    # Compare CPU and CUDA results
    np.testing.assert_allclose(cpu_next_state, cuda_next_state, atol=1e-9, rtol=1e-9)
