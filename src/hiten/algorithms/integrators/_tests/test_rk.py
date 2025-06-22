import numpy as np
import pytest
from numba.typed import List
from scipy.integrate import solve_ivp

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from hiten.algorithms.polynomial.operations import polynomial_evaluate
from hiten.algorithms.dynamics.rhs import create_rhs_system
from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.integrators.rk import RungeKutta, AdaptiveRK
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF, N_VARS_POLY,
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
    """Compare every RK implementation (fixed-step and adaptive) with SciPy's reference solvers on a Hamiltonian hiten.system."""

    H_poly, hamiltonian_system, _, clmo = rk_test_data

    # Initial conditions (small amplitude pendulum)
    initial_q1 = 0.1
    initial_p1 = 0.0

    # 2-D state for SciPy, 6-D state for our code (extra zeros)
    state_scipy = np.array([initial_q1, initial_p1], dtype=np.float64)
    state_rk = np.array([initial_q1, 0.0, 0.0, initial_p1, 0.0, 0.0], dtype=np.float64)

    t_final = 20.0
    n_samples = int(t_final * 200.0)  # 200 samples per unit time
    t_eval = np.linspace(0.0, t_final, n_samples, dtype=np.float64)

    # (label, factory, scipy_method)
    rk_variants = [
        ("RK4",  lambda: RungeKutta(order=4), "DOP853"),
        ("RK6",  lambda: RungeKutta(order=6), "DOP853"),
        ("RK8",  lambda: RungeKutta(order=8), "DOP853"),
        ("RK45", lambda: AdaptiveRK(order=5, debug=True), "RK45"),
        ("DOP853", lambda: AdaptiveRK(order=8, debug=True), "DOP853"),
    ]

    def taylor_pendulum_ode(t, y):
        Q, P = y
        sin_taylor = Q - (Q ** 3) / 6.0 + (Q ** 5) / 120.0  # 5th-order expansion
        return [P, -sin_taylor]

    for label, make_integrator, scipy_method in rk_variants:
        integrator = make_integrator()

        # Integrate with our RK implementation
        sol_rk = integrator.integrate(hamiltonian_system, state_rk, t_eval)
        rk_Q = sol_rk.states[:, 0]

        # Reference solution from SciPy
        sol_sp = solve_ivp(
            taylor_pendulum_ode,
            [0.0, t_final],
            state_scipy,
            method=scipy_method,
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-12,
        )
        sp_Q = sol_sp.y[0]

        # Trajectory comparison (RMS error)
        q_rms = np.sqrt(np.mean((rk_Q - sp_Q) ** 2))
        assert q_rms < 5e-3, f"{label}: Q RMS error too large vs SciPy - {q_rms}"

        # Energy drift comparison (sanity check)
        rk_energy = np.array([evaluate_hamiltonian(H_poly, s, clmo) for s in sol_rk.states])
        sp_energy = np.array([
            evaluate_hamiltonian(H_poly, np.array([sp_Q[i], 0.0, 0.0, sol_sp.y[1][i], 0.0, 0.0]), clmo)
            for i in range(len(t_eval))
        ])

        rk_drift = np.max(np.abs(rk_energy - rk_energy[0]))
        sp_drift = np.max(np.abs(sp_energy - sp_energy[0]))

        assert rk_drift < 1e-2, f"{label}: energy drift too large - {rk_drift}"
        assert sp_drift < 1e-2, f"{label}: SciPy energy drift too large - {sp_drift}"


def test_vs_solve_ivp_generic_rhs():
    """Compare every RK implementation with SciPy on a generic RHS (harmonic oscillator)."""

    # Harmonic oscillator: x'' + x = 0  ->  [x', v'; v' = -x]
    def harmonic_oscillator(t, y):
        return np.array([y[1], -y[0]])

    rhs_system = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic Oscillator")

    initial_state = np.array([1.0, 0.0], dtype=np.float64)
    t_final = 2 * np.pi  # one period
    n_samples = 2000
    t_eval = np.linspace(0.0, t_final, n_samples, dtype=np.float64)

    rk_variants = [
        ("RK4",  lambda: RungeKutta(order=4), "DOP853"),
        ("RK6",  lambda: RungeKutta(order=6), "DOP853"),
        ("RK8",  lambda: RungeKutta(order=8), "DOP853"),
        ("RK45", lambda: AdaptiveRK(order=5, debug=True), "RK45"),
        ("DOP853", lambda: AdaptiveRK(order=8, debug=True), "DOP853"),
    ]

    for label, make_integrator, scipy_method in rk_variants:
        integrator = make_integrator()

        sol_rk = integrator.integrate(rhs_system, initial_state, t_eval)
        rk_x = sol_rk.states[:, 0]
        rk_v = sol_rk.states[:, 1]

        sol_sp = solve_ivp(
            harmonic_oscillator,
            [0.0, t_final],
            initial_state,
            method=scipy_method,
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-12,
        )
        sp_x = sol_sp.y[0]
        sp_v = sol_sp.y[1]

        x_rms = np.sqrt(np.mean((rk_x - sp_x) ** 2))
        v_rms = np.sqrt(np.mean((rk_v - sp_v) ** 2))

        tol = 1e-5 if label != "RK4" else 1e-4  # slightly looser for lower order
        assert x_rms < tol, f"{label}: position RMS error too large - {x_rms}"
        assert v_rms < tol, f"{label}: velocity RMS error too large - {v_rms}"

