import numpy as np
import pytest

from algorithms.center.polynomial.base import (
    init_index_tables, make_poly, encode_multiindex, _create_encode_dict_from_clmo
)
from algorithms.center.polynomial.operations import polynomial_evaluate
from algorithms.variables import N_VARS

from algorithms.center.poincare.map_cuda import HamiltonianRHSEvaluatorCUDA, N_DOF

# Max degree for the partial derivative polynomials of H used in these tests.
# If H is cubic, its partials are quadratic.
TEST_MAX_DEGREE_JACOBIAN_COMPONENT = 2 
PSI, CLMO = init_index_tables(TEST_MAX_DEGREE_JACOBIAN_COMPONENT)
# Create an ENCODE_DICT specific to the CLMO table used in these tests
ENCODE_DICT = _create_encode_dict_from_clmo(CLMO)


def _create_example_jac_h():
    """
    Creates a sample jac_H for testing based on H = q0*p0^2 + q1^2.
    Variables are mapped as: q0,q1,q2 -> x0,x1,x2 and p0,p1,p2 -> x3,x4,x5.
    
    dH/dq0 = p0^2 = x3^2
    dH/dq1 = 2*q1 = 2*x1
    dH/dq2 = 0
    dH/dp0 = 2*q0*p0 = 2*x0*x3
    dH/dp1 = 0
    dH/dp2 = 0

    Each component of jac_H is a list of coefficient arrays, one for each degree.
    """
    jac_H_list = []

    # dH/dq0 = x3^2 (Degree 2)
    poly_dH_dq0 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    k_p0sq = np.array([0,0,0,2,0,0], dtype=np.int64) # x3^2
    idx_p0sq = encode_multiindex(k_p0sq, 2, ENCODE_DICT)
    if idx_p0sq != -1 and 2 < len(poly_dH_dq0) and idx_p0sq < poly_dH_dq0[2].shape[0]:
         poly_dH_dq0[2][idx_p0sq] = 1.0
    jac_H_list.append(poly_dH_dq0)

    # dH/dq1 = 2*x1 (Degree 1)
    poly_dH_dq1 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    k_2x1 = np.array([0,1,0,0,0,0], dtype=np.int64) # x1
    idx_2x1 = encode_multiindex(k_2x1, 1, ENCODE_DICT)
    if idx_2x1 != -1 and 1 < len(poly_dH_dq1) and idx_2x1 < poly_dH_dq1[1].shape[0]:
        poly_dH_dq1[1][idx_2x1] = 2.0
    jac_H_list.append(poly_dH_dq1)

    # dH/dq2 = 0 (All degrees are zero polynomials)
    poly_dH_dq2 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    jac_H_list.append(poly_dH_dq2)

    # dH/dp0 = 2*x0*x3 (Degree 2)
    poly_dH_dp0 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    k_2x0x3 = np.array([1,0,0,1,0,0], dtype=np.int64) # x0*x3
    idx_2x0x3 = encode_multiindex(k_2x0x3, 2, ENCODE_DICT)
    if idx_2x0x3 != -1 and 2 < len(poly_dH_dp0) and idx_2x0x3 < poly_dH_dp0[2].shape[0]:
        poly_dH_dp0[2][idx_2x0x3] = 2.0
    jac_H_list.append(poly_dH_dp0)

    # dH/dp1 = 0
    poly_dH_dp1 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    jac_H_list.append(poly_dH_dp1)

    # dH/dp2 = 0
    poly_dH_dp2 = [make_poly(d, PSI) for d in range(TEST_MAX_DEGREE_JACOBIAN_COMPONENT + 1)]
    jac_H_list.append(poly_dH_dp2)
    
    return jac_H_list

def test_hamiltonian_rhs_cuda_vs_cpu():
    """
    Tests the HamiltonianRHS CUDA implementation against a CPU equivalent.
    """
    jac_H = _create_example_jac_h()
    
    # Sample state vector [q0,q1,q2,p0,p1,p2]
    # Corresponds to [x0,x1,x2,x3,x4,x5]
    state_real = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.15], dtype=np.float64)
    state_complex = state_real.astype(np.complex128)

    # CPU Calculation of Hamiltonian RHS
    cpu_rhs = np.zeros(2 * N_DOF, dtype=np.float64)
    
    # dH/dqi polynomials are jac_H[0]...jac_H[N_DOF-1]
    # dH/dpi polynomials are jac_H[N_DOF]...jac_H[2*N_DOF-1]
    
    for i in range(N_DOF):
        # dp_i/dt = -dH/dq_i
        # jac_H[i] is the polynomial for dH/dq_i
        poly_dH_dqi = jac_H[i]
        val_dH_dqi = polynomial_evaluate(poly_dH_dqi, state_complex, CLMO)
        cpu_rhs[N_DOF + i] = -val_dH_dqi.real
        
        # dq_i/dt = dH/dp_i
        # jac_H[N_DOF + i] is the polynomial for dH/dp_i
        poly_dH_dpi = jac_H[N_DOF + i]
        val_dH_dpi = polynomial_evaluate(poly_dH_dpi, state_complex, CLMO)
        cpu_rhs[i] = val_dH_dpi.real

    # CUDA Calculation of Hamiltonian RHS
    # The CLMO table passed to HamiltonianRHSEvaluatorCUDA must be the same
    # one used for encoding the polynomials in jac_H and for CPU evaluation.
    cuda_evaluator = HamiltonianRHSEvaluatorCUDA(jac_H, CLMO) 
    cuda_rhs = cuda_evaluator.evaluate_single(state_real)

    # Compare CPU and CUDA results
    np.testing.assert_allclose(cpu_rhs, cuda_rhs, atol=1e-9, rtol=1e-9)
