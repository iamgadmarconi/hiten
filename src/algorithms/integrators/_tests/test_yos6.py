import numpy as np
import pytest
from numba.typed import List

# Functions and objects to be tested
from algorithms.integrators.yos6 import yoshida6_integrate, yoshida6_step
from algorithms.center.polynomial.base import init_index_tables, make_poly, encode_multiindex, decode_multiindex
from algorithms.center.polynomial.operations import polynomial_zero_list, polynomial_differentiate, polynomial_evaluate
from algorithms.center.polynomial.algebra import _poly_diff
from algorithms.variables import N_VARS  # To get number of variables

# Create actual polynomials for a harmonic oscillator system
# H = 0.5 * (q2^2 + p2^2)

def setup_harmonic_oscillator_system(max_deg=4):
    """
    Create polynomial blocks for a simple harmonic oscillator system.
    The variable ordering is (0, q2, q3, 0, p2, p3) matching the CR3BP center manifold reduction.
    Returns H_blocks and precomputed dH_blocks.
    """
    # Initialize tables for the polynomial algebra
    psi, clmo = init_index_tables(max_deg)
    
    # Create zero polynomial list as a base
    H_blocks = polynomial_zero_list(max_deg, psi)
    
    # Set coefficients for H = 0.5 * (q2^2 + p2^2)
    # First, create multiindex vector for q2^2
    q2_squared_idx = np.zeros(N_VARS, dtype=np.int64)
    q2_squared_idx[1] = 2  # q2^2 - variable at index 1 with power 2 (variable order is q1,q2,q3,p1,p2,p3)
    
    # Create multiindex vector for p2^2
    p2_squared_idx = np.zeros(N_VARS, dtype=np.int64)
    p2_squared_idx[4] = 2  # p2^2 - variable at index 4 with power 2 (variable order is q1,q2,q3,p1,p2,p3)
    
    # Encode the multiindex vectors to get the corresponding positions in the coefficient array
    q2_squared_pos = encode_multiindex(q2_squared_idx, 2, psi, clmo)
    p2_squared_pos = encode_multiindex(p2_squared_idx, 2, psi, clmo)
    
    # Set the coefficients at the correct positions
    H_blocks[2][q2_squared_pos] = 0.5  # 0.5 * q2^2
    H_blocks[2][p2_squared_pos] = 0.5  # 0.5 * p2^2
    
    # Compute derivatives with automatic differentiation
    dH_blocks = []
    for var_idx in range(6):
        dH_i, _ = polynomial_differentiate(H_blocks, var_idx, max_deg, psi, clmo, psi, clmo)
        dH_blocks.append(dH_i)
    
    # Convert to a Numba typed list for JIT compatibility
    dH_blocks_typed = List()
    for dH_i in dH_blocks:
        dH_blocks_typed.append(dH_i)
    
    return H_blocks, dH_blocks_typed, clmo

@pytest.fixture
def ho_system():
    """Fixture providing the harmonic oscillator system components"""
    return setup_harmonic_oscillator_system()

def print_polynomial_info(H_blocks, dH_blocks, clmo):
    """Helper function to print information about polynomials and their derivatives"""
    # Print out the first few terms of our Hamiltonian to verify it's correct
    for d in range(len(H_blocks)):
        if np.any(H_blocks[d]):
            print(f"Degree {d} terms in Hamiltonian:")
            for i in range(H_blocks[d].shape[0]):
                if abs(H_blocks[d][i]) > 1e-10:
                    k = decode_multiindex(i, d, clmo)
                    print(f"  Coefficient {H_blocks[d][i]:.6f} for term with powers {k}")
    
    # Check all derivative blocks
    for var_idx in range(6):
        var_name = ['q1', 'q2', 'q3', 'p1', 'p2', 'p3'][var_idx]
        print(f"\ndH/d{var_name} terms:")
        for d in range(len(dH_blocks[var_idx])):
            if np.any(dH_blocks[var_idx][d]):
                print(f"  Degree {d} terms:")
                for i in range(dH_blocks[var_idx][d].shape[0]):
                    if abs(dH_blocks[var_idx][d][i]) > 1e-10:
                        k = decode_multiindex(i, d, clmo)
                        print(f"    Coefficient {dH_blocks[var_idx][d][i]:.6f} for term with powers {k}")

def test_energy_conservation(ho_system):
    """
    Tests that the energy of the harmonic oscillator is conserved during integration.
    """
    H_blocks, dH_blocks, clmo = ho_system
    
    # Generate a test state vector (q1, q2, q3, p1, p2, p3)
    # For H = 0.5 * (q2^2 + p2^2), q1, q3, p1, p3 are passive.
    initial_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # q1=0, q2=1, q3=0, p1=0, p2=0, p3=0
    
    # Directly calculate energy to verify system
    energy_value = polynomial_evaluate(H_blocks, initial_state, clmo).real
    # We expect 0.5 * (1^2 + 0^2) = 0.5
    expected_energy = 0.5
    assert abs(energy_value - expected_energy) < 1e-12, f"Energy calculation is wrong. Expected {expected_energy}, got {energy_value}"
    
    # Run the integration test
    t_end = 20.0  # Integrate for a few periods
    dt = 0.01

    # Use the Hamiltonian and derivatives
    traj, energy = yoshida6_integrate(
        initial_state, t_end, dt,
        H_blocks=H_blocks,
        dH_blocks=dH_blocks,
        clmo=clmo
    )

    assert len(energy) > 1, "Energy array should not be empty"

    # Energy should be conserved at expected_energy = 0.5
    assert np.allclose(energy, expected_energy, atol=1e-9), \
        f"Energy not conserved. Expected ~{expected_energy}, got std: {np.std(energy)}, range: {np.min(energy)}-{np.max(energy)}"

def test_reversibility(ho_system):
    """
    Tests if integrating forward and then backward returns to the initial state.
    """
    H_blocks, dH_blocks, clmo = ho_system
    
    # Arbitrary initial state (q1, q2, q3, p1, p2, p3)
    s0 = np.array([0.1, 0.5, -0.2, -0.1, -0.3, 0.2]) 
    dt = 0.005
    t_duration = 2.0 # Integrate for t_duration forward, then t_duration backward
    
    num_steps = int(round(t_duration / dt))
    assert num_steps > 0, "Number of integration steps must be positive"

    # Integrate forward
    s_current = s0.copy()
    for _ in range(num_steps):
        s_current = yoshida6_step(s_current, dt, dH_blocks, clmo)
    
    s_mid = s_current.copy()

    # Integrate backward
    for _ in range(num_steps):
        s_current = yoshida6_step(s_current, -dt, dH_blocks, clmo)
    
    s_final = s_current

    # The final state should be very close to the initial state
    assert np.allclose(s_final, s0, atol=1e-10), \
        f"Integrator not reversible. Initial: {s0}, Final: {s_final}, Diff: {s_final - s0}"

def test_known_solution_harmonic_oscillator(ho_system):
    """
    Tests the integrator against the analytical solution of a harmonic oscillator.
    q1(t) = q1_0 (passive)
    q2(t) = q2_0 cos(t) + p2_0 sin(t)
    q3(t) = q3_0 (passive)
    p1(t) = p1_0 (passive)
    p2(t) = -q2_0 sin(t) + p2_0 cos(t)
    p3(t) = p3_0 (passive)
    """
    H_blocks, dH_blocks, clmo = ho_system

    q1_0, q2_0, q3_0 = 0.0, 1.0, 0.2
    p1_0, p2_0, p3_0 = 0.0, 0.0, -0.1 
    initial_state = np.array([q1_0, q2_0, q3_0, p1_0, p2_0, p3_0])
    
    # Use a shorter integration time and smaller time step for better accuracy
    t_end = np.pi/2  # Quarter period - reduces accumulated error
    dt = 0.00001  # Smaller step size for better accuracy

    traj, _ = yoshida6_integrate(
        initial_state, t_end, dt,
        H_blocks=H_blocks,
        dH_blocks=dH_blocks,
        clmo=clmo
    )
    
    final_numerical_state = traj[-1]

    # Analytical solution at t_end (π/2)
    # At t = π/2: cos(π/2) = 0, sin(π/2) = 1
    # So q2 = p2_0 = 0, p2 = -q2_0 = -1
    # q1, q3, p1, p3 should remain unchanged as they are passive.
    q1_analytic_final = q1_0
    q2_analytic_final = q2_0 * np.cos(t_end) + p2_0 * np.sin(t_end)
    q3_analytic_final = q3_0
    p1_analytic_final = p1_0
    p2_analytic_final = -q2_0 * np.sin(t_end) + p2_0 * np.cos(t_end)
    p3_analytic_final = p3_0
    
    analytic_final_state = np.array([q1_analytic_final, q2_analytic_final, q3_analytic_final, 
                                     p1_analytic_final, p2_analytic_final, p3_analytic_final])
    
    # Print the actual numerical vs analytical values for diagnosis
    print(f"\nNumerical final state: {final_numerical_state}")
    print(f"Analytical final state: {analytic_final_state}")
    print(f"Difference: {final_numerical_state - analytic_final_state}")
    
    # Yoshida6 should be very accurate with the smaller dt
    assert np.allclose(final_numerical_state, analytic_final_state, atol=1e-8), \
        f"Numerical solution diverges from analytical. Numerical: {final_numerical_state}, Analytical: {analytic_final_state}"

# To run these tests, navigate to the root of your project (cr3bpv2) in the terminal
# and run pytest. Ensure pytest and numpy are installed.
# Example:
# cd /path/to/cr3bpv2
# pytest
