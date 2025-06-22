import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba.typed import List
from scipy.integrate import solve_ivp

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from hiten.algorithms.polynomial.operations import polynomial_evaluate
from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF, N_VARS_POLY,
                                               P_POLY_INDICES, Q_POLY_INDICES,
                                               TaoSymplectic)

TEST_MAX_DEG = 6


def evaluate_hamiltonian(
    H_poly_list: List[np.ndarray],
    state_6d: np.ndarray, # Expects [q1,q2,q3,p1,p2,p3]
    clmo_tables: List[np.ndarray]
    ) -> float:
    """
    Evaluates the Hamiltonian for a given 6D state.
    The Hamiltonian polynomial itself is defined over 6 variables.
    """
    if state_6d.shape[0] != 2 * N_SYMPLECTIC_DOF:
        raise ValueError(f"State dimension {state_6d.shape[0]} not compatible with N_SYMPLECTIC_DOF {N_SYMPLECTIC_DOF}")

    eval_point_6d = np.zeros(N_VARS_POLY, dtype=np.complex128)
    eval_point_6d[Q_POLY_INDICES] = state_6d[0:N_SYMPLECTIC_DOF] # q1,q2,q3
    eval_point_6d[P_POLY_INDICES] = state_6d[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF] # p1,p2,p3
    
    return polynomial_evaluate(H_poly_list, eval_point_6d, clmo_tables).real


@pytest.fixture(scope="module")
def symplectic_test_data():
    psi_tables, clmo_tables_numba = init_index_tables(TEST_MAX_DEG)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_tables_numba)

    H_poly = [np.zeros(psi_tables[N_VARS_POLY, d], dtype=np.complex128) for d in range(TEST_MAX_DEG + 1)]

    idx_P_var = P_POLY_INDICES[0]
    idx_Q_var = Q_POLY_INDICES[0]

    # H = P1^2/2 - (1 - Q1^2/2 + Q1^4/24 - Q1^6/720)
    # P1^2/2 term (degree 2)
    k_Psq = np.zeros(N_VARS_POLY, dtype=np.int64); k_Psq[idx_P_var] = 2
    idx_Psq_encoded = encode_multiindex(k_Psq, 2, encode_dict_list)
    if idx_Psq_encoded != -1: H_poly[2][idx_Psq_encoded] = 0.5

    # -1 term (degree 0)
    k_const = np.zeros(N_VARS_POLY, dtype=np.int64)
    idx_const_encoded = encode_multiindex(k_const, 0, encode_dict_list)
    if idx_const_encoded != -1: H_poly[0][idx_const_encoded] = -1.0

    # +Q1^2/2 term (degree 2)
    k_Qsq = np.zeros(N_VARS_POLY, dtype=np.int64); k_Qsq[idx_Q_var] = 2
    idx_Qsq_encoded = encode_multiindex(k_Qsq, 2, encode_dict_list)
    if idx_Qsq_encoded != -1: H_poly[2][idx_Qsq_encoded] += 0.5 # Add to existing P1^2/2 degree 2 array

    # -Q1^4/24 term (degree 4)
    if TEST_MAX_DEG >= 4:
        k_Q4 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q4[idx_Q_var] = 4
        idx_Q4_encoded = encode_multiindex(k_Q4, 4, encode_dict_list)
        if idx_Q4_encoded != -1: H_poly[4][idx_Q4_encoded] = -1.0 / 24.0

    # +Q1^6/720 term (degree 6)
    if TEST_MAX_DEG >= 6:
        k_Q6 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q6[idx_Q_var] = 6
        idx_Q6_encoded = encode_multiindex(k_Q6, 6, encode_dict_list)
        if idx_Q6_encoded != -1: H_poly[6][idx_Q6_encoded] = 1.0 / 720.0
    
    # Convert H_poly to Numba typed list for internal consistency if polynomial_jacobian needs it
    H_poly_numba = List()
    for arr in H_poly:
        H_poly_numba.append(arr.copy())

    # Create the Hamiltonian system using the new API
    hamiltonian_system = create_hamiltonian_system(
        H_blocks=H_poly_numba,
        max_degree=TEST_MAX_DEG,
        psi_table=psi_tables,
        clmo_table=clmo_tables_numba,
        encode_dict_list=encode_dict_list,
        n_dof=N_SYMPLECTIC_DOF,
        name="Test Pendulum System"
    )

    return H_poly_numba, hamiltonian_system, psi_tables, clmo_tables_numba


def test_energy_conservation(symplectic_test_data):
    H_poly, hamiltonian_system, psi, clmo = symplectic_test_data

    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/2
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 20.0
    num_steps = 2000
    times = np.linspace(0, t_final, num_steps, dtype=np.float64)
    order = 6 # Test with 8th order
    omega_tao = 20.0 # Increased from 5.0 for better energy conservation

    # Use the new integrator API
    integrator = TaoSymplectic(order=order, c_omega_heuristic=omega_tao)
    solution = integrator.integrate(hamiltonian_system, initial_state, times)
    trajectory = solution.states

    initial_energy = evaluate_hamiltonian(H_poly, trajectory[0], clmo)
    final_energy = evaluate_hamiltonian(H_poly, trajectory[-1], clmo)
    
    assert np.isclose(initial_energy, final_energy, atol=1e-5), (
        f"Energy not conserved for pendulum. Initial: {initial_energy}, Final: {final_energy}"
    )


def test_reversibility(symplectic_test_data):
    _, hamiltonian_system, _, _ = symplectic_test_data

    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = 0.5
    initial_p1 = 0.3
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 1.5
    num_steps = 150
    times_forward = np.linspace(0, t_final, num_steps, dtype=np.float64)
    times_backward = np.linspace(t_final, 0, num_steps, dtype=np.float64)
    order = 4
    omega_tao = 5.0  # Increased from 0.1 for better energy conservation

    # Use the new integrator API
    integrator = TaoSymplectic(order=order, c_omega_heuristic=omega_tao)

    # Forward integration
    solution_fwd = integrator.integrate(hamiltonian_system, initial_state, times_forward)
    state_at_t_final = solution_fwd.states[-1].copy()

    # Backward integration
    solution_bwd = integrator.integrate(hamiltonian_system, state_at_t_final, times_backward)
    final_state_reversed = solution_bwd.states[-1]

    assert np.allclose(initial_state, final_state_reversed, atol=1e-6), (
        f"Reversibility failed. Initial: {initial_state}, Reversed: {final_state_reversed}"
    )


def test_final_state_error(symplectic_test_data):
    _, hamiltonian_system, _, _ = symplectic_test_data
    
    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/4
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = np.pi # Integrate for roughly half a period for Q0=pi/4, P0=0 (small angle)
    order = 6
    omega_tao = 5.0  # Increased from 0.05 for better energy conservation

    # Use the new integrator API
    integrator = TaoSymplectic(order=order, c_omega_heuristic=omega_tao)

    # Run with a moderate number of steps
    num_steps1 = 200
    times1 = np.linspace(0, t_final, num_steps1, dtype=np.float64)
    solution1 = integrator.integrate(hamiltonian_system, initial_state, times1)
    final_state1 = solution1.states[-1]

    # Run with many more steps (reference)
    num_steps2 = 800 
    times2 = np.linspace(0, t_final, num_steps2, dtype=np.float64)
    solution2 = integrator.integrate(hamiltonian_system, initial_state, times2)
    final_state_ref = solution2.states[-1]

    assert np.allclose(final_state1, final_state_ref, atol=1e-5, rtol=1e-4), (
        f"Final state error too large. Coarse: {final_state1}, Ref: {final_state_ref}"
    )


def test_vs_solve_ivp(symplectic_test_data):
    H_poly, hamiltonian_system, psi, clmo = symplectic_test_data

    initial_q1 = 0.1
    initial_p1 = 0.0
    initial_state_scipy = np.array([initial_q1, initial_p1], dtype=np.float64)
    initial_state_symplectic = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 100.0  
    num_points = int(t_final * 100.0)
    t_eval = np.linspace(0, t_final, num_points)

    def taylor_pendulum_ode(t, y):
        Q, P = y[0], y[1]
        taylor_sin_Q = Q - (Q**3)/6.0 + (Q**5)/120.0 # 5th order Taylor expansion for sin(Q)
        return [P, -taylor_sin_Q]  # [dQ/dt, dP/dt = -sin(Q)]
    
    scipy_solution = solve_ivp(
        taylor_pendulum_ode,
        [0, t_final],
        initial_state_scipy,
        method='RK45',
        rtol=1e-13,
        atol=1e-13,
        t_eval=t_eval
    )
    
    actual_times = t_eval
    
    order = 6  # Higher order for better accuracy
    omega_tao = 20.0
    
    # Use the new integrator API
    integrator = TaoSymplectic(order=order, c_omega_heuristic=omega_tao)
    solution = integrator.integrate(hamiltonian_system, initial_state_symplectic, actual_times)
    symplectic_traj = solution.states
    
    symplectic_Q = symplectic_traj[:, Q_POLY_INDICES[0]]
    symplectic_P = symplectic_traj[:, P_POLY_INDICES[0]]
    
    scipy_Q = scipy_solution.y[0]
    scipy_P = scipy_solution.y[1]
    
    analytical_Q = initial_state_scipy[0] * np.cos(actual_times)
    analytical_P = -initial_state_scipy[0] * np.sin(actual_times)
    
    scipy_energy = []
    analytical_energy = []
    
    for i in range(len(actual_times)):
        current_scipy_state_6d = np.array([scipy_Q[i], 0.0, 0.0, scipy_P[i], 0.0, 0.0])
        scipy_energy.append(evaluate_hamiltonian(H_poly, current_scipy_state_6d, clmo))

        current_analytical_state_6d = np.array([analytical_Q[i], 0.0, 0.0, analytical_P[i], 0.0, 0.0])
        analytical_energy.append(evaluate_hamiltonian(H_poly, current_analytical_state_6d, clmo))
    
    symplectic_energy = []
    for i in range(len(actual_times)):
        state_6d = symplectic_traj[i]
        symplectic_energy.append(evaluate_hamiltonian(H_poly, state_6d, clmo))
    
    scipy_energy = np.array(scipy_energy)
    symplectic_energy = np.array(symplectic_energy)
    analytical_energy = np.array(analytical_energy)
    
    q_rms_error_symplectic = np.sqrt(np.mean((symplectic_Q - analytical_Q)**2))
    q_rms_error_scipy = np.sqrt(np.mean((scipy_Q - analytical_Q)**2))
    
    print(f"\nRMS Q error vs analytical: Symplectic: {q_rms_error_symplectic}, solve_ivp: {q_rms_error_scipy}")
    
    max_rms_error = 0.01
    assert q_rms_error_symplectic < max_rms_error, f"Symplectic Q error too large: {q_rms_error_symplectic}"
    assert q_rms_error_scipy < max_rms_error, f"solve_ivp Q error too large: {q_rms_error_scipy}"
    
    scipy_energy_drift = np.max(np.abs(scipy_energy - scipy_energy[0]))
    symplectic_energy_drift = np.max(np.abs(symplectic_energy - symplectic_energy[0]))
    analytical_energy_drift = np.max(np.abs(analytical_energy - analytical_energy[0]))
    
    print(f"Energy drift: Symplectic: {symplectic_energy_drift}, solve_ivp: {scipy_energy_drift}, analytical: {analytical_energy_drift}")
    
    assert symplectic_energy_drift < 1e-4, f"Symplectic energy drift too large: {symplectic_energy_drift}"
    
    if scipy_energy_drift > 1e-10:
        assert symplectic_energy_drift < scipy_energy_drift, (
            f"Symplectic integrator should have less energy drift. "
            f"Symplectic: {symplectic_energy_drift}, solve_ivp: {scipy_energy_drift}"
        )
    

    plt.figure(figsize=(15, 10))
    
    # Plot Q trajectories
    plt.subplot(2, 2, 1)
    plt.plot(actual_times, symplectic_Q, 'b-', label='Symplectic')
    plt.plot(actual_times, scipy_Q, 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_Q, 'g-.', label='Analytical')
    plt.title('Position (Q)')
    plt.legend()
    
    # Plot P trajectories
    plt.subplot(2, 2, 2)
    plt.plot(actual_times, symplectic_P, 'b-', label='Symplectic')
    plt.plot(actual_times, scipy_P, 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_P, 'g-.', label='Analytical')
    plt.title('Momentum (P)')
    plt.legend()
    
    # Plot phase space
    plt.subplot(2, 2, 3)
    plt.plot(symplectic_Q, symplectic_P, 'b-', label='Symplectic')
    plt.plot(scipy_Q, scipy_P, 'r--', label='solve_ivp')
    plt.plot(analytical_Q, analytical_P, 'g-.', label='Analytical')
    plt.title('Phase Space')
    plt.xlabel('Q')
    plt.ylabel('P')
    plt.legend()
    
    # Plot energy error
    plt.subplot(2, 2, 4)
    plt.plot(actual_times, symplectic_energy - symplectic_energy[0], 'b-', label='Symplectic')
    plt.plot(actual_times, scipy_energy - scipy_energy[0], 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_energy - analytical_energy[0], 'g-.', label='Analytical')
    plt.title('Energy Error')
    plt.yscale('symlog', linthresh=1e-15)  # Log scale to see small differences
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'symplectic_vs_scipy.png'))
    plt.close()
    print("Saved comparison plot to '{}' in test directory".format(os.path.join(os.path.dirname(__file__), 'symplectic_vs_scipy.png')))
