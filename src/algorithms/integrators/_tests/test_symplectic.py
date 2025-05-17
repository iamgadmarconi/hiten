import numpy as np
import pytest
import os
from numba.typed import List

from algorithms.center.polynomial.base import init_index_tables, encode_multiindex
from algorithms.center.polynomial.operations import polynomial_evaluate, polynomial_jacobian
from algorithms.integrators.symplectic import (
    integrate_symplectic,
    N_CM_DOF as INTEGRATOR_N_CM_DOF, # Get N_CM_DOF used by integrator
    N_VARS_POLY, CM_Q_POLY_INDICES, CM_P_POLY_INDICES
)

# --- Test Configuration ---
MAX_DEG_TEST_HAM = 6  # Max degree for Taylor expansion of test Hamiltonians
# For 1 DOF system, N_CM_DOF in integrator must be 1
# We need to ensure the integrator's N_CM_DOF is consistent with our 1-DOF test case.
# This might require a mechanism to set N_CM_DOF for the test, or the test adapts.
# For now, let's assume the integrator is compiled with N_CM_DOF=1 for these tests,
# or we focus on one pair of Q,P from a 2-DOF system and ensure other parts are zero.

# For a 1-DOF system (Q, P):
# Q_cm1 maps to poly_idx CM_Q_POLY_INDICES[0]
# P_cm1 maps to poly_idx CM_P_POLY_INDICES[0]

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
def evaluate_hamiltonian_cm(
    H_poly_list: List[np.ndarray],
    cm_state_2dof: np.ndarray, # [Q1, P1] or [Q1,Q2,P1,P2] depending on N_CM_DOF
    psi_tables: np.ndarray,
    clmo_tables: List[np.ndarray]
    ) -> float:
    """
    Evaluates the Hamiltonian for a given CM state.
    Adapts to N_CM_DOF used by the integrator module.
    """
    if cm_state_2dof.shape[0] == 2: # Assuming 1-DOF for this specific helper context
        q_cm_actual = np.array([cm_state_2dof[0]])
        p_cm_actual = np.array([cm_state_2dof[1]])
    elif cm_state_2dof.shape[0] == 4 and INTEGRATOR_N_CM_DOF == 2:
        q_cm_actual = cm_state_2dof[0:INTEGRATOR_N_CM_DOF]
        p_cm_actual = cm_state_2dof[INTEGRATOR_N_CM_DOF : 2*INTEGRATOR_N_CM_DOF]
    else:
        # Fallback or error for mismatched dimensions
        # This part needs to be robust based on how N_CM_DOF is handled
        raise ValueError(f"State dimension {cm_state_2dof.shape[0]} not compatible with N_CM_DOF {INTEGRATOR_N_CM_DOF}")

    point_6d = np.zeros(N_VARS_POLY, dtype=np.complex128)
    for i in range(INTEGRATOR_N_CM_DOF):
        point_6d[CM_Q_POLY_INDICES[i]] = q_cm_actual[i]
        point_6d[CM_P_POLY_INDICES[i]] = p_cm_actual[i]

    return polynomial_evaluate(H_poly_list, point_6d, clmo_tables).real


@pytest.fixture(scope="module")
def pendulum_hamiltonian_data():
    """
    Prepares data for H = P^2/2 - cos(Q) ~ P^2/2 - (1 - Q^2/2! + Q^4/4! - Q^6/6!)
    This is a 1-DOF system. We'll use the first CM DoF (Q_cm1, P_cm1).
    Assumes INTEGRATOR_N_CM_DOF is 1 or adaptable.
    If INTEGRATOR_N_CM_DOF = 2, we set Q2,P2 terms to zero.
    """
    if INTEGRATOR_N_CM_DOF != 1 and INTEGRATOR_N_CM_DOF != 2:
        pytest.skip("Pendulum test requires integrator N_CM_DOF to be 1 or adaptable to 1-DOF from 2-DOF.")

    psi_tables, clmo_tables_numba = init_index_tables(MAX_DEG_TEST_HAM)
    H_poly = [np.zeros(psi_tables[N_VARS_POLY, d], dtype=np.complex128) for d in range(MAX_DEG_TEST_HAM + 1)]

    # P_cm1 corresponds to poly_var index CM_P_POLY_INDICES[0]
    # Q_cm1 corresponds to poly_var index CM_Q_POLY_INDICES[0]
    idx_P_var = CM_P_POLY_INDICES[0]
    idx_Q_var = CM_Q_POLY_INDICES[0]

    # H = P^2/2 - (1 - Q^2/2 + Q^4/24 - Q^6/720)
    # P^2/2 term (degree 2)
    k_Psq = np.zeros(N_VARS_POLY, dtype=np.int64); k_Psq[idx_P_var] = 2
    idx_Psq_encoded = encode_multiindex(k_Psq, 2, psi_tables, clmo_tables_numba)
    if idx_Psq_encoded != -1: H_poly[2][idx_Psq_encoded] = 0.5

    # -1 term (degree 0)
    k_const = np.zeros(N_VARS_POLY, dtype=np.int64)
    idx_const_encoded = encode_multiindex(k_const, 0, psi_tables, clmo_tables_numba)
    if idx_const_encoded != -1: H_poly[0][idx_const_encoded] = -1.0

    # +Q^2/2 term (degree 2)
    k_Qsq = np.zeros(N_VARS_POLY, dtype=np.int64); k_Qsq[idx_Q_var] = 2
    idx_Qsq_encoded = encode_multiindex(k_Qsq, 2, psi_tables, clmo_tables_numba)
    if idx_Qsq_encoded != -1: H_poly[2][idx_Qsq_encoded] += 0.5 # Add to existing P^2/2 degree 2 array

    # -Q^4/24 term (degree 4)
    if MAX_DEG_TEST_HAM >= 4:
        k_Q4 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q4[idx_Q_var] = 4
        idx_Q4_encoded = encode_multiindex(k_Q4, 4, psi_tables, clmo_tables_numba)
        if idx_Q4_encoded != -1: H_poly[4][idx_Q4_encoded] = -1.0 / 24.0

    # +Q^6/720 term (degree 6)
    if MAX_DEG_TEST_HAM >= 6:
        k_Q6 = np.zeros(N_VARS_POLY, dtype=np.int64); k_Q6[idx_Q_var] = 6
        idx_Q6_encoded = encode_multiindex(k_Q6, 6, psi_tables, clmo_tables_numba)
        if idx_Q6_encoded != -1: H_poly[6][idx_Q6_encoded] = 1.0 / 720.0
    
    # Convert H_poly to Numba typed list for internal consistency if polynomial_jacobian needs it
    H_poly_numba = List()
    for arr in H_poly:
        H_poly_numba.append(arr.copy())

    jac_H_py = polynomial_jacobian(H_poly_numba, MAX_DEG_TEST_HAM, psi_tables, clmo_tables_numba)
    jac_H_numba = _numbafy_jacobian(jac_H_py)

    return H_poly_numba, jac_H_numba, psi_tables, clmo_tables_numba


def test_energy_conservation_pendulum(pendulum_hamiltonian_data):
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data

    # Ensure we use 1-DOF slice of CM variables for a 1-DOF Hamiltonian
    if INTEGRATOR_N_CM_DOF == 1:
        initial_state = np.array([np.pi/2, 0.0], dtype=np.float64) # Q0, P0
    elif INTEGRATOR_N_CM_DOF == 2:
        # Use Q1,P1 for pendulum, Q2,P2 are zero and should remain so
        initial_state = np.array([np.pi/2, 0.0, 0.0, 0.0], dtype=np.float64) # Q1,Q2,P1,P2
    else:
        pytest.skip(f"Integrator N_CM_DOF {INTEGRATOR_N_CM_DOF} not directly testable with 1-DOF pendulum like this.")

    t_final = 20.0
    num_steps = 2000
    times = np.linspace(0, t_final, num_steps, dtype=np.float64)
    order = 6 # Test with 6th order
    omega_tao = 20.0 # Increased from 5.0 for better energy conservation

    trajectory = integrate_symplectic(
        initial_cm_state_4d=initial_state, # Adapt based on INTEGRATOR_N_CM_DOF
        t_values=times,
        jac_H_cm_rn_typed=jac_H,
        clmo_H_typed=clmo,
        order=order,
        c_omega_heuristic=omega_tao
    )

    initial_energy = evaluate_hamiltonian_cm(H_poly, trajectory[0], psi, clmo)
    final_energy = evaluate_hamiltonian_cm(H_poly, trajectory[-1], psi, clmo)
    
    # Energy error for symplectic integrators should ideally not grow linearly
    # For a 4th order method, expect good conservation. The tolerance here is indicative.
    # The polynomial approx of cos(Q) also introduces error.
    assert np.isclose(initial_energy, final_energy, atol=1e-5), (
        f"Energy not conserved for pendulum. Initial: {initial_energy}, Final: {final_energy}"
    )


def test_reversibility_pendulum(pendulum_hamiltonian_data):
    H_poly, jac_H, psi, clmo = pendulum_hamiltonian_data

    if INTEGRATOR_N_CM_DOF == 1:
        initial_state = np.array([0.5, 0.3], dtype=np.float64)
    elif INTEGRATOR_N_CM_DOF == 2:
        initial_state = np.array([0.5, 0.0, 0.3, 0.0], dtype=np.float64)
    else:
        pytest.skip("Integrator N_CM_DOF not suitable for this 1-DOF pendulum reversibility test.")

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

    if INTEGRATOR_N_CM_DOF == 1:
        initial_state = np.array([np.pi/4, 0.0], dtype=np.float64)
        # For H = P^2/2 - cos(Q), period is approx 2*pi for small Q0, P0=0.
        # For Q0=pi/4, period is T = 4K(sin^2(Q0/2)) where K is elliptic integral.
        # Let's use a time where we expect a certain state for a simpler reference if possible,
        # or compare against a very high-accuracy solve_ivp.
        # For simplicity, let's just check against a smaller number of steps to see convergence.
    elif INTEGRATOR_N_CM_DOF == 2:
        initial_state = np.array([np.pi/4, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        pytest.skip("Integrator N_CM_DOF not suitable for this 1-DOF pendulum final state test.")

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

    # Use very small initial displacement to stay firmly in region where Taylor series is accurate
    if INTEGRATOR_N_CM_DOF == 1:
        initial_state = np.array([0.1, 0.0], dtype=np.float64)  # Q0, P0
    elif INTEGRATOR_N_CM_DOF == 2:
        initial_state = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float64)  # Q1, Q2, P1, P2
    else:
        pytest.skip("Integrator N_CM_DOF not suitable for this solve_ivp comparison test")

    # Use shorter integration time and more points for accurate comparison
    t_final = 100.0  # Just a couple of periods for a small oscillation
    num_points = int(t_final * 1000.0)
    t_eval = np.linspace(0, t_final, num_points)
    
    # For small oscillations, the period is approximately 2π
    # So t_final = 2.0 should cover about 1/π periods ≈ 0.64 periods
    
    # Define pendulum ODE system for solve_ivp using the SAME Taylor expansion of sin(Q)
    # as in our polynomial Hamiltonian
    def taylor_pendulum_ode(t, y):
        """
        The pendulum ODE system with Taylor expanded sin(Q)
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
        initial_state[:2],  # Only use first 2 components for 1-DOF pendulum
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
        initial_state,
        actual_times,
        jac_H,
        clmo,
        order=order,
        c_omega_heuristic=omega_tao
    )
    
    # Extract relevant DOF for comparison
    if INTEGRATOR_N_CM_DOF == 1:
        symplectic_Q = symplectic_traj[:, 0]
        symplectic_P = symplectic_traj[:, 1]
    else:  # 2-DOF case, use first DOF (assuming pendulum is in first DOF)
        symplectic_Q = symplectic_traj[:, 0]
        symplectic_P = symplectic_traj[:, INTEGRATOR_N_CM_DOF]
    
    # Get reference solutions
    scipy_Q = scipy_solution.y[0]
    scipy_P = scipy_solution.y[1]
    
    # For small oscillations, we can also compute an analytical solution
    # For a harmonic oscillator (which approximates pendulum for small angles):
    # Q(t) = Q₀cos(t) + P₀sin(t)
    # P(t) = P₀cos(t) - Q₀sin(t)
    analytical_Q = initial_state[0] * np.cos(actual_times)
    analytical_P = -initial_state[0] * np.sin(actual_times)
    
    # Calculate energies using the same Taylor expanded Hamiltonian
    # H = P^2/2 - (1 - Q^2/2 + Q^4/24 - Q^6/720)
    scipy_energy = []
    analytical_energy = []
    
    for i in range(len(actual_times)):
        # Calculate energy for scipy solution using H_poly
        if INTEGRATOR_N_CM_DOF == 1:
            current_scipy_state = np.array([scipy_Q[i], scipy_P[i]])
        else: # INTEGRATOR_N_CM_DOF == 2
            # Pendulum motion is on Q1, P1. Scipy evolved only these.
            current_scipy_state = np.array([scipy_Q[i], 0.0, scipy_P[i], 0.0])
        scipy_energy.append(evaluate_hamiltonian_cm(H_poly, current_scipy_state, psi, clmo))

        # Calculate energy for analytical solution using H_poly
        if INTEGRATOR_N_CM_DOF == 1:
            current_analytical_state = np.array([analytical_Q[i], analytical_P[i]])
        else: # INTEGRATOR_N_CM_DOF == 2
            current_analytical_state = np.array([analytical_Q[i], 0.0, analytical_P[i], 0.0])
        analytical_energy.append(evaluate_hamiltonian_cm(H_poly, current_analytical_state, psi, clmo))
    
    # For symplectic, use our polynomial approximation evaluator
    symplectic_energy = []
    for i in range(len(actual_times)):
        if INTEGRATOR_N_CM_DOF == 1:
            state = np.array([symplectic_Q[i], symplectic_P[i]])
        else:
            state = symplectic_traj[i]
        symplectic_energy.append(evaluate_hamiltonian_cm(H_poly, state, psi, clmo))
    
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
