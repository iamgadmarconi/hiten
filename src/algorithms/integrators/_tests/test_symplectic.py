import os

import numpy as np
import pytest
from numba.typed import List

from algorithms.center.polynomial.base import (encode_multiindex,
                                               init_index_tables,
                                               _create_encode_dict_from_clmo)
from algorithms.center.polynomial.operations import (polynomial_evaluate,
                                                     polynomial_jacobian)
from algorithms.integrators.symplectic import (P_POLY_INDICES,
                                               Q_POLY_INDICES)
from algorithms.integrators.symplectic import N_SYMPLECTIC_DOF
from algorithms.integrators.symplectic import N_VARS_POLY, integrate_symplectic

# --- Test Configuration ---
MAX_DEG_TEST_HAM = 6  # Max degree for Taylor expansion of test Hamiltonians

# Helper to create Numba typed list of lists for Jacobian
def _numbafy_jacobian(jac_py_list_of_lists: list[list[np.ndarray]]) -> List[List[np.ndarray]]:
    jac_numba_typed = List()
    for poly_deriv_py_list in jac_py_list_of_lists:
        inner_numba_list = List()
        for coeff_array_np in poly_deriv_py_list:
            inner_numba_list.append(coeff_array_np.copy())
        jac_numba_typed.append(inner_numba_list)
    return jac_numba_typed

# Helper to evaluate H(Q,P) given its polynomial representation
def evaluate_hamiltonian_test_system(
    H_poly_list: List[np.ndarray],
    state_6d: np.ndarray, # Expects [q1,q2,q3,p1,p2,p3]
    psi_tables: np.ndarray, # This argument might be unused if clmo is sufficient
    clmo_tables: List[np.ndarray]
    ) -> float:
    """
    Evaluates the Hamiltonian for a given 6D state.
    The Hamiltonian polynomial itself is defined over 6 variables.
    """
    if state_6d.shape[0] != 2 * N_SYMPLECTIC_DOF:
        raise ValueError(f"State dimension {state_6d.shape[0]} not compatible with N_SYMPLECTIC_DOF {N_SYMPLECTIC_DOF}")

    # The polynomial_evaluate function expects a 6D complex vector where variables
    # are ordered according to their definition (q1,q2,q3,p1,p2,p3)
    # Q_POLY_INDICES are [0,1,2], P_POLY_INDICES are [3,4,5]
    eval_point_6d = np.zeros(N_VARS_POLY, dtype=np.complex128)
    eval_point_6d[Q_POLY_INDICES] = state_6d[0:N_SYMPLECTIC_DOF] # q1,q2,q3
    eval_point_6d[P_POLY_INDICES] = state_6d[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF] # p1,p2,p3
    
    return polynomial_evaluate(H_poly_list, eval_point_6d, clmo_tables).real


@pytest.fixture(scope="module")
def pendulum_hamiltonian_data():
    """
    Prepares data for H = P1^2/2 - cos(Q1) ~ P1^2/2 - (1 - Q1^2/2! + Q1^4/4! - Q1^6/6!)
    This is a 1-DOF system, now embedded in the 3-DOF integrator framework.
    Q1 corresponds to q1 (poly_idx 0), P1 to p1 (poly_idx 3).
    Other DOFs (Q2,P2, Q3,P3) will have zero coefficients in this Hamiltonian.
    """
    # N_SYMPLECTIC_DOF is now 3, this test adapts to it by focusing on the first DOF.

    psi_tables, clmo_tables_numba = init_index_tables(MAX_DEG_TEST_HAM)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_tables_numba)

    H_poly = [np.zeros(psi_tables[N_VARS_POLY, d], dtype=np.complex128) for d in range(MAX_DEG_TEST_HAM + 1)]

    # P1 corresponds to poly_var index Q_POLY_INDICES[0] for Q part, P_POLY_INDICES[0] for P part
    # P1 is p1, maps to polynomial variable index 3
    # Q1 is q1, maps to polynomial variable index 0
    idx_P_var = P_POLY_INDICES[0] # Should be 3
    idx_Q_var = Q_POLY_INDICES[0] # Should be 0

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
    if MAX_DEG_TEST_HAM >= 4:
        k_Q4 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q4[idx_Q_var] = 4
        idx_Q4_encoded = encode_multiindex(k_Q4, 4, encode_dict_list)
        if idx_Q4_encoded != -1: H_poly[4][idx_Q4_encoded] = -1.0 / 24.0

    # +Q1^6/720 term (degree 6)
    if MAX_DEG_TEST_HAM >= 6:
        k_Q6 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q6[idx_Q_var] = 6
        idx_Q6_encoded = encode_multiindex(k_Q6, 6, encode_dict_list)
        if idx_Q6_encoded != -1: H_poly[6][idx_Q6_encoded] = 1.0 / 720.0
    
    # Convert H_poly to Numba typed list for internal consistency if polynomial_jacobian needs it
    H_poly_numba = List()
    for arr in H_poly:
        H_poly_numba.append(arr.copy())

    jac_H_py = polynomial_jacobian(
        H_poly_numba, 
        MAX_DEG_TEST_HAM, 
        psi_tables, 
        clmo_tables_numba, 
        encode_dict_list
    )
    jac_H_numba = _numbafy_jacobian(jac_H_py)

    return H_poly_numba, jac_H_numba, psi_tables, clmo_tables_numba


def test_energy_conservation_pendulum(pendulum_hamiltonian_data):
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data

    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/2
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 20.0
    num_steps = 20000
    times = np.linspace(0, t_final, num_steps, dtype=np.float64)
    order = 8 # Test with 6th order
    omega_tao = 200.0 # Increased from 5.0 for better energy conservation

    trajectory = integrate_symplectic(
        initial_state_6d=initial_state, # Changed name
        t_values=times,
        jac_H_rn_typed=jac_H, # Changed name
        clmo_H_typed=clmo,
        order=order,
        c_omega_heuristic=omega_tao
    )

    initial_energy = evaluate_hamiltonian_test_system(H_poly, trajectory[0], psi, clmo)
    final_energy = evaluate_hamiltonian_test_system(H_poly, trajectory[-1], psi, clmo)
    
    # Energy error for symplectic integrators should ideally not grow linearly
    # For a 4th order method, expect good conservation. The tolerance here is indicative.
    # The polynomial approx of cos(Q) also introduces error.
    assert np.isclose(initial_energy, final_energy, atol=1e-5), (
        f"Energy not conserved for pendulum. Initial: {initial_energy}, Final: {final_energy}"
    )


def test_reversibility_pendulum(pendulum_hamiltonian_data):
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data

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

    # Forward integration
    traj_fwd = integrate_symplectic(
        initial_state, times_forward, jac_H, clmo, order, c_omega_heuristic=omega_tao
    )
    state_at_t_final = traj_fwd[-1].copy()

    # Backward integration
    # For Hamiltonian systems, reversing time implies P -> -P if H(q,-p) = H(q,p)
    # Tao's method is time-reversible directly by using negative timesteps.
    traj_bwd = integrate_symplectic(
        state_at_t_final, times_backward, jac_H, clmo, order, c_omega_heuristic=omega_tao
    )
    final_state_reversed = traj_bwd[-1]

    assert np.allclose(initial_state, final_state_reversed, atol=1e-6), (
        f"Reversibility failed. Initial: {initial_state}, Reversed: {final_state_reversed}"
    )


def test_final_state_error_pendulum(pendulum_hamiltonian_data):
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data
    
    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/4
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = np.pi # Integrate for roughly half a period for Q0=pi/4, P0=0 (small angle)
    order = 6
    omega_tao = 5.0  # Increased from 0.05 for better energy conservation

    # Run with a moderate number of steps
    num_steps1 = 200
    times1 = np.linspace(0, t_final, num_steps1, dtype=np.float64)
    traj1 = integrate_symplectic(initial_state, times1, jac_H, clmo, order, c_omega_heuristic=omega_tao)
    final_state1 = traj1[-1]

    # Run with many more steps (reference)
    num_steps2 = 800 
    times2 = np.linspace(0, t_final, num_steps2, dtype=np.float64)
    traj2 = integrate_symplectic(initial_state, times2, jac_H, clmo, order, c_omega_heuristic=omega_tao)
    final_state_ref = traj2[-1]

    # Check that final_state1 is reasonably close to final_state_ref
    # The error should scale with (dt)^order. (dt1/dt2)^order ~ (4)^6 error reduction ideally.
    # This is a basic convergence check, not a strict error bound test.
    assert np.allclose(final_state1, final_state_ref, atol=1e-5, rtol=1e-4), (
        f"Final state error too large. Coarse: {final_state1}, Ref: {final_state_ref}"
    )


def test_comparison_with_solve_ivp(pendulum_hamiltonian_data):
    """
    Compare the symplectic integrator with scipy's solve_ivp for a pendulum system.
    Both use the same Taylor approximation of sin(Q) to ensure a fair comparison.
    
    For small amplitude oscillations and short time periods, both integrators 
    should produce very similar results. Over longer periods, the symplectic 
    integrator should maintain better energy conservation.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        pytest.skip("scipy not available for comparison test")
    
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data

    # Pendulum is 1-DOF (q1, p1). Initial state for solve_ivp is [q1, p1]
    # Symplectic integrator needs the full 6D state.
    initial_q1 = 0.1
    initial_p1 = 0.0
    initial_state_scipy = np.array([initial_q1, initial_p1], dtype=np.float64)
    initial_state_symplectic = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    # Use shorter integration time and more points for accurate comparison
    t_final = 100.0  
    num_points = int(t_final * 10000.0)
    t_eval = np.linspace(0, t_final, num_points)
    
    # For small oscillations, the period is approximately 2π
    # So t_final = 2.0 should cover about 1/π periods ≈ 0.64 periods
    
    # Define pendulum ODE system for solve_ivp using the SAME Taylor expansion of sin(Q)
    # as in our polynomial Hamiltonian
    def taylor_pendulum_ode(t, y):
        """
        The pendulum ODE system with Taylor expanded sin(Q)
        y[0] is q1, y[1] is p1
        """
        Q, P = y[0], y[1]  # Extract from state vector
        # Use the same order of expansion as in the pendulum_hamiltonian_data fixture
        # H_poly has cos(Q) up to Q^6, so sin(Q) = -dH/dQ should be up to Q^5
        taylor_sin_Q = Q - (Q**3)/6.0 + (Q**5)/120.0 # 5th order Taylor expansion for sin(Q)
        return [P, -taylor_sin_Q]  # [dQ/dt, dP/dt = -sin(Q)]
    
    # Integrate with solve_ivp (high accuracy)
    scipy_solution = solve_ivp(
        taylor_pendulum_ode,
        [0, t_final],
        initial_state_scipy,  # Use 2D state for solve_ivp on 1-DOF ODE
        method='RK45',
        rtol=1e-13,  # Very tight tolerance
        atol=1e-13,
        t_eval=t_eval
    )
    
    # Get the actual time values returned by solve_ivp
    actual_times = t_eval  # Use requested times for consistency
    
    # Integrate with our symplectic integrator
    order = 6  # Higher order for better accuracy
    omega_tao = 20.0
    
    symplectic_traj = integrate_symplectic(
        initial_state_symplectic, # Use 6D state
        actual_times,
        jac_H,
        clmo,
        order=order,
        c_omega_heuristic=omega_tao
    )
    
    # Extract relevant DOF for comparison (q1, p1)
    symplectic_Q = symplectic_traj[:, Q_POLY_INDICES[0]] # q1 is at index 0
    symplectic_P = symplectic_traj[:, P_POLY_INDICES[0]] # p1 is at index N_SYMPLECTIC_DOF (which is 3)
    
    # Get reference solutions
    scipy_Q = scipy_solution.y[0]
    scipy_P = scipy_solution.y[1]
    
    # For small oscillations, we can also compute an analytical solution
    # For a harmonic oscillator (which approximates pendulum for small angles):
    # Q(t) = Q₀cos(t) + P₀sin(t)
    # P(t) = P₀cos(t) - Q₀sin(t)
    analytical_Q = initial_state_scipy[0] * np.cos(actual_times)
    analytical_P = -initial_state_scipy[0] * np.sin(actual_times)
    
    # Calculate energies using the same Taylor expanded Hamiltonian
    # H = P^2/2 - (1 - Q^2/2 + Q^4/24 - Q^6/720)
    scipy_energy = []
    analytical_energy = []
    
    for i in range(len(actual_times)):
        # Calculate energy for scipy solution using H_poly
        # Construct 6D state for evaluate_hamiltonian_test_system: [q1_scipy, 0,0, p1_scipy,0,0]
        current_scipy_state_6d = np.array([scipy_Q[i], 0.0, 0.0, scipy_P[i], 0.0, 0.0])
        scipy_energy.append(evaluate_hamiltonian_test_system(H_poly, current_scipy_state_6d, psi, clmo))

        # Calculate energy for analytical solution using H_poly
        # Construct 6D state: [q1_analytical, 0,0, p1_analytical,0,0]
        current_analytical_state_6d = np.array([analytical_Q[i], 0.0, 0.0, analytical_P[i], 0.0, 0.0])
        analytical_energy.append(evaluate_hamiltonian_test_system(H_poly, current_analytical_state_6d, psi, clmo))
    
    # For symplectic, the trajectory is already 6D
    symplectic_energy = []
    for i in range(len(actual_times)):
        state_6d = symplectic_traj[i]
        symplectic_energy.append(evaluate_hamiltonian_test_system(H_poly, state_6d, psi, clmo))
    
    # Convert to numpy arrays
    scipy_energy = np.array(scipy_energy)
    symplectic_energy = np.array(symplectic_energy)
    analytical_energy = np.array(analytical_energy)
    
    # --- Basic trajectory comparison ---
    # For short periods and small oscillations, all methods should give similar results
    # Calculate RMS error against analytical solution
    q_rms_error_symplectic = np.sqrt(np.mean((symplectic_Q - analytical_Q)**2))
    q_rms_error_scipy = np.sqrt(np.mean((scipy_Q - analytical_Q)**2))
    
    print(f"\nRMS Q error vs analytical: Symplectic: {q_rms_error_symplectic}, solve_ivp: {q_rms_error_scipy}")
    
    # Both methods should have small errors for this short integration
    max_rms_error = 0.01  # 1% error tolerance
    assert q_rms_error_symplectic < max_rms_error, f"Symplectic Q error too large: {q_rms_error_symplectic}"
    assert q_rms_error_scipy < max_rms_error, f"solve_ivp Q error too large: {q_rms_error_scipy}"
    
    # --- Energy conservation test ---
    # Calculate energy drift as max deviation from initial energy
    scipy_energy_drift = np.max(np.abs(scipy_energy - scipy_energy[0]))
    symplectic_energy_drift = np.max(np.abs(symplectic_energy - symplectic_energy[0]))
    analytical_energy_drift = np.max(np.abs(analytical_energy - analytical_energy[0]))
    
    print(f"Energy drift: Symplectic: {symplectic_energy_drift}, solve_ivp: {scipy_energy_drift}, analytical: {analytical_energy_drift}")
    
    # For this short integration, both should maintain reasonable energy conservation
    # But symplectic should be better
    assert symplectic_energy_drift < 1e-4, f"Symplectic energy drift too large: {symplectic_energy_drift}"
    
    # The key advantage of symplectic integrators is better energy conservation
    # This advantage may be subtle for short integrations but should still be visible
    if scipy_energy_drift > 1e-10:  # Only compare if solve_ivp has some drift
        assert symplectic_energy_drift < scipy_energy_drift, (
            f"Symplectic integrator should have less energy drift. "
            f"Symplectic: {symplectic_energy_drift}, solve_ivp: {scipy_energy_drift}"
        )
    
    # Optional: Plot the trajectories for visual inspection
    try:
        import matplotlib.pyplot as plt
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
    except ImportError:
        pass
