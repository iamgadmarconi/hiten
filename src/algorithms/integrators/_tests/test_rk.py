import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba.typed import List
from scipy.integrate import solve_ivp

from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import polynomial_evaluate
from algorithms.dynamics.rhs import create_rhs_system
from algorithms.dynamics.rtbp import create_rtbp_system
from algorithms.dynamics.hamiltonian import create_hamiltonian_system
from algorithms.integrators.rk import RungeKutta
from algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF, N_VARS_POLY,
                                               P_POLY_INDICES, Q_POLY_INDICES)

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
def rk_test_data():
    """Create test data for RK integrator tests."""
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
    
    # Convert H_poly to Numba typed list for internal consistency
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


def test_energy_conservation(rk_test_data):
    """Test energy conservation for RK integrator (should show energy drift)."""
    H_poly, hamiltonian_system, psi, clmo = rk_test_data

    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/6  # Smaller amplitude for better RK performance
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 10.0  # Shorter time for RK test
    num_steps = 10000
    times = np.linspace(0, t_final, num_steps, dtype=np.float64)
    order = 8  # Test with 8th order RK

    # Use the new integrator API
    integrator = RungeKutta(order=order)
    solution = integrator.integrate(hamiltonian_system, initial_state, times)
    trajectory = solution.states

    initial_energy = evaluate_hamiltonian(H_poly, trajectory[0], clmo)
    final_energy = evaluate_hamiltonian(H_poly, trajectory[-1], clmo)
    
    # RK methods don't conserve energy exactly, so we expect some drift
    energy_drift = abs(final_energy - initial_energy)
    relative_drift = energy_drift / abs(initial_energy)
    
    # For RK8 with fine timestep, energy drift should be reasonable but not perfect
    assert relative_drift < 1e-6, (
        f"Energy drift too large for RK integrator. Initial: {initial_energy}, "
        f"Final: {final_energy}, Relative drift: {relative_drift}"
    )


def test_reversibility(rk_test_data):
    """Test reversibility for RK integrator (should show some error)."""
    _, hamiltonian_system, _, _ = rk_test_data

    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = 0.3  # Smaller amplitude for better accuracy
    initial_p1 = 0.2
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 1.0  # Shorter time for reversibility test
    num_steps = 1000
    times_forward = np.linspace(0, t_final, num_steps, dtype=np.float64)
    times_backward = np.linspace(t_final, 0, num_steps, dtype=np.float64)
    order = 8  # High order for better accuracy

    # Use the new integrator API
    integrator = RungeKutta(order=order)

    # Forward integration
    solution_fwd = integrator.integrate(hamiltonian_system, initial_state, times_forward)
    state_at_t_final = solution_fwd.states[-1].copy()

    # Backward integration
    solution_bwd = integrator.integrate(hamiltonian_system, state_at_t_final, times_backward)
    final_state_reversed = solution_bwd.states[-1]

    # RK methods are not exactly reversible, so we allow larger tolerance
    assert np.allclose(initial_state, final_state_reversed, atol=1e-8, rtol=1e-6), (
        f"Reversibility error too large for RK integrator. Initial: {initial_state}, "
        f"Reversed: {final_state_reversed}"
    )


def test_final_state_error(rk_test_data):
    """Test convergence of RK integrator with different step sizes."""
    _, hamiltonian_system, _, _ = rk_test_data
    
    # Pendulum is 1-DOF (q1, p1). Initial state is 6D [q1,q2,q3,p1,p2,p3]
    initial_q1 = np.pi/4
    initial_p1 = 0.0
    initial_state = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = np.pi/2  # Quarter period for small angle approximation
    order = 6

    # Use the new integrator API
    integrator = RungeKutta(order=order)

    # Run with a moderate number of steps
    num_steps1 = 100
    times1 = np.linspace(0, t_final, num_steps1, dtype=np.float64)
    solution1 = integrator.integrate(hamiltonian_system, initial_state, times1)
    final_state1 = solution1.states[-1]

    # Run with many more steps (reference)
    num_steps2 = 1600  # 16x more steps
    times2 = np.linspace(0, t_final, num_steps2, dtype=np.float64)
    solution2 = integrator.integrate(hamiltonian_system, initial_state, times2)
    final_state_ref = solution2.states[-1]

    # For RK6, error should scale as O(h^6), so 16x more steps should give much better accuracy
    assert np.allclose(final_state1, final_state_ref, atol=1e-4, rtol=1e-3), (
        f"Final state error too large for RK integrator. Coarse: {final_state1}, "
        f"Ref: {final_state_ref}"
    )


def test_vs_solve_ivp(rk_test_data):
    """Test RK integrator against scipy's solve_ivp."""
    H_poly, hamiltonian_system, psi, clmo = rk_test_data

    initial_q1 = 0.1
    initial_p1 = 0.0
    initial_state_scipy = np.array([initial_q1, initial_p1], dtype=np.float64)
    initial_state_rk = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 50.0  # Moderate time for comparison
    num_points = int(t_final * 1000.0)  # 1000 points per unit time
    t_eval = np.linspace(0, t_final, num_points)

    def taylor_pendulum_ode(t, y):
        Q, P = y[0], y[1]
        taylor_sin_Q = Q - (Q**3)/6.0 + (Q**5)/120.0  # 5th order Taylor expansion for sin(Q)
        return [P, -taylor_sin_Q]  # [dQ/dt, dP/dt = -sin(Q)]
    
    scipy_solution = solve_ivp(
        taylor_pendulum_ode,
        [0, t_final],
        initial_state_scipy,
        method='RK45',
        rtol=1e-10,
        atol=1e-10,
        t_eval=t_eval
    )
    
    actual_times = t_eval
    
    order = 8  # High order for comparison
    
    # Use the new integrator API
    integrator = RungeKutta(order=order)
    solution = integrator.integrate(hamiltonian_system, initial_state_rk, actual_times)
    rk_traj = solution.states
    
    rk_Q = rk_traj[:, Q_POLY_INDICES[0]]
    rk_P = rk_traj[:, P_POLY_INDICES[0]]
    
    scipy_Q = scipy_solution.y[0]
    scipy_P = scipy_solution.y[1]
    
    analytical_Q = initial_state_scipy[0] * np.cos(actual_times)
    analytical_P = -initial_state_scipy[0] * np.sin(actual_times)
    
    # Compute energies
    scipy_energy = []
    analytical_energy = []
    
    for i in range(len(actual_times)):
        current_scipy_state_6d = np.array([scipy_Q[i], 0.0, 0.0, scipy_P[i], 0.0, 0.0])
        scipy_energy.append(evaluate_hamiltonian(H_poly, current_scipy_state_6d, clmo))

        current_analytical_state_6d = np.array([analytical_Q[i], 0.0, 0.0, analytical_P[i], 0.0, 0.0])
        analytical_energy.append(evaluate_hamiltonian(H_poly, current_analytical_state_6d, clmo))
    
    rk_energy = []
    for i in range(len(actual_times)):
        state_6d = rk_traj[i]
        rk_energy.append(evaluate_hamiltonian(H_poly, state_6d, clmo))
    
    scipy_energy = np.array(scipy_energy)
    rk_energy = np.array(rk_energy)
    analytical_energy = np.array(analytical_energy)
    
    # Compare trajectory accuracy
    q_rms_error_rk = np.sqrt(np.mean((rk_Q - analytical_Q)**2))
    q_rms_error_scipy = np.sqrt(np.mean((scipy_Q - analytical_Q)**2))
    
    print(f"\nRMS Q error vs analytical: RK{order}: {q_rms_error_rk}, solve_ivp: {q_rms_error_scipy}")
    
    max_rms_error = 0.01
    assert q_rms_error_rk < max_rms_error, f"RK Q error too large: {q_rms_error_rk}"
    assert q_rms_error_scipy < max_rms_error, f"solve_ivp Q error too large: {q_rms_error_scipy}"
    
    # Compare energy conservation
    scipy_energy_drift = np.max(np.abs(scipy_energy - scipy_energy[0]))
    rk_energy_drift = np.max(np.abs(rk_energy - rk_energy[0]))
    analytical_energy_drift = np.max(np.abs(analytical_energy - analytical_energy[0]))
    
    print(f"Energy drift: RK{order}: {rk_energy_drift}, solve_ivp: {scipy_energy_drift}, analytical: {analytical_energy_drift}")
    
    # Both should have reasonable energy conservation for this test
    assert rk_energy_drift < 1e-2, f"RK energy drift too large: {rk_energy_drift}"
    assert scipy_energy_drift < 1e-2, f"solve_ivp energy drift too large: {scipy_energy_drift}"
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot Q trajectories
    plt.subplot(2, 2, 1)
    plt.plot(actual_times, rk_Q, 'b-', label=f'RK{order}')
    plt.plot(actual_times, scipy_Q, 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_Q, 'g-.', label='Analytical')
    plt.title('Position (Q)')
    plt.legend()
    
    # Plot P trajectories
    plt.subplot(2, 2, 2)
    plt.plot(actual_times, rk_P, 'b-', label=f'RK{order}')
    plt.plot(actual_times, scipy_P, 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_P, 'g-.', label='Analytical')
    plt.title('Momentum (P)')
    plt.legend()
    
    # Plot phase space
    plt.subplot(2, 2, 3)
    plt.plot(rk_Q, rk_P, 'b-', label=f'RK{order}')
    plt.plot(scipy_Q, scipy_P, 'r--', label='solve_ivp')
    plt.plot(analytical_Q, analytical_P, 'g-.', label='Analytical')
    plt.title('Phase Space')
    plt.xlabel('Q')
    plt.ylabel('P')
    plt.legend()
    
    # Plot energy error
    plt.subplot(2, 2, 4)
    plt.plot(actual_times, rk_energy - rk_energy[0], 'b-', label=f'RK{order}')
    plt.plot(actual_times, scipy_energy - scipy_energy[0], 'r--', label='solve_ivp')
    plt.plot(actual_times, analytical_energy - analytical_energy[0], 'g-.', label='Analytical')
    plt.title('Energy Error')
    plt.yscale('symlog', linthresh=1e-15)  # Log scale to see small differences
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'rk_vs_scipy.png'))
    plt.close()
    print("Saved comparison plot to '{}' in test directory".format(os.path.join(os.path.dirname(__file__), 'rk_vs_scipy.png')))


def test_rtbp_system():
    """Test RK integrator with RTBP system."""
    # Earth-Moon system
    mu = 0.012277471
    rtbp_system = create_rtbp_system(mu, name="Earth-Moon RTBP")
    
    # Initial conditions near L1 point (approximate)
    initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=np.float64)
    
    t_final = 2.0  # Short integration time
    num_steps = 2000
    times = np.linspace(0, t_final, num_steps)
    
    # Test different orders
    for order in [4, 6, 8]:
        integrator = RungeKutta(order=order)
        solution = integrator.integrate(rtbp_system, initial_state, times)
        
        # Basic sanity checks
        assert solution.states.shape == (num_steps, 6), f"Wrong trajectory shape for order {order}"
        assert np.allclose(solution.times, times), f"Times mismatch for order {order}"
        assert np.allclose(solution.states[0], initial_state), f"Initial state mismatch for order {order}"


def test_generic_rhs_system():
    """Test RK integrator with generic RHS system."""
    # Simple harmonic oscillator
    def harmonic_oscillator(t, y):
        return np.array([y[1], -y[0]])  # [dx/dt = v, dv/dt = -x]
    
    rhs_system = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic Oscillator")
    
    # Initial conditions
    initial_state = np.array([1.0, 0.0])  # Start at x=1, v=0
    
    t_final = 2*np.pi  # One full period
    num_steps = 1000
    times = np.linspace(0, t_final, num_steps)
    
    # Test with RK4
    integrator = RungeKutta(order=4)
    solution = integrator.integrate(rhs_system, initial_state, times)
    
    # Analytical solution
    analytical_x = np.cos(times)
    analytical_v = -np.sin(times)
    
    # Check accuracy
    x_error = np.max(np.abs(solution.states[:, 0] - analytical_x))
    v_error = np.max(np.abs(solution.states[:, 1] - analytical_v))
    
    assert x_error < 1e-6, f"Position error too large: {x_error}"
    assert v_error < 1e-6, f"Velocity error too large: {v_error}"
    
    # Check that we return to initial state after one period
    final_state = solution.states[-1]
    assert np.allclose(final_state, initial_state, atol=1e-6), (
        f"Harmonic oscillator not periodic. Initial: {initial_state}, Final: {final_state}"
    )
