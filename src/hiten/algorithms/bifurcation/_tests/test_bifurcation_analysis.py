import numpy as np
import pytest
from numba.typed import List

from hiten.algorithms.bifurcation.analysis import (_eigenvalues_flow,
                                                   _equilibrium_with_energy,
                                                   _find_equilibrium,
                                                   _linearise_hamiltonian,
                                                   _symplectic_J)
from hiten.algorithms.fourier.base import (_create_encode_dict_fourier,
                                           _encode_fourier_index,
                                           _init_fourier_tables)
from hiten.algorithms.fourier.operations import _make_fourier_poly


@pytest.fixture(scope="module")
def fourier_data():
    """
    Set up a test Hamiltonian and corresponding Fourier data structures.
    H = I_1 + 2*I_2 + I_1^2 + 3*I_2^2 + I_1*I_2*cos(th_1 - 2*th_2)
    """
    max_degree = 2
    k_max = 2
    psiF, clmoF = _init_fourier_tables(max_degree, k_max)
    encodeF = _create_encode_dict_fourier(clmoF)

    coeffs_list = List()
    coeffs_list.append(_make_fourier_poly(0, psiF))
    coeffs_list.append(_make_fourier_poly(1, psiF))
    coeffs_list.append(_make_fourier_poly(2, psiF))

    # Degree 1: I_1 + 2*I_2
    pos = _encode_fourier_index((1, 0, 0, 0, 0, 0), 1, encodeF)
    coeffs_list[1][pos] = 1.0
    pos = _encode_fourier_index((0, 1, 0, 0, 0, 0), 1, encodeF)
    coeffs_list[1][pos] = 2.0

    # Degree 2: I_1^2 + 3*I_2^2 + I_1*I_2*cos(th_1 - 2*th_2)
    # I_1^2
    pos = _encode_fourier_index((2, 0, 0, 0, 0, 0), 2, encodeF)
    coeffs_list[2][pos] = 1.0
    # 3*I_2^2
    pos = _encode_fourier_index((0, 2, 0, 0, 0, 0), 2, encodeF)
    coeffs_list[2][pos] = 3.0
    # I_1*I_2*cos(th_1 - 2*th_2) = 0.5*I_1*I_2*(exp(i*(th_1-2th_2)) + exp(-i*(th_1-2th_2)))
    # Term 1: k = (1, -2, 0)
    pos = _encode_fourier_index((1, 1, 0, 1, -2, 0), 2, encodeF)
    coeffs_list[2][pos] = 0.5
    # Term 2: k = (-1, 2, 0)
    pos = _encode_fourier_index((1, 1, 0, -1, 2, 0), 2, encodeF)
    coeffs_list[2][pos] = 0.5

    return coeffs_list, clmoF


def test_symplectic_J():
    """Test the construction of the symplectic matrix J."""
    J = _symplectic_J()
    assert J.shape == (6, 6)
    assert np.allclose(J.T, -J)
    assert np.allclose(J @ J, -np.identity(6))

    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))
    expected_J = np.block([[Z3, -I3], [I3, Z3]])
    assert np.allclose(J, expected_J)


def test_linearise_hamiltonian(fourier_data):
    """Test the linearization of a sample Hamiltonian."""
    coeffs_list, clmoF = fourier_data
    I_vals = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    theta_vals = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    H_val, grad, Hess, A = _linearise_hamiltonian(
        coeffs_list, I_vals, theta_vals, clmoF
    )

    # Expected values for H = I_1 + 2*I_2 + I_1^2 + 3*I_2^2 + I_1*I_2*cos(th_1 - 2*th_2)
    # at I=(1,1,0), th=(0,0,0)
    expected_H_val = 1 + 2 + 1 + 3 + 1 * 1 * np.cos(0)
    assert np.isclose(H_val, expected_H_val)

    expected_grad = np.array([
        1 + 2 * 1 + 1 * np.cos(0),  # dH/dI1
        2 + 6 * 1 + 1 * np.cos(0),  # dH/dI2
        0.0,                        # dH/dI3
        -1 * 1 * np.sin(0),         # dH/dth1
        2 * 1 * 1 * np.sin(0),      # dH/dth2
        0.0,                        # dH/dth3
    ])
    np.testing.assert_allclose(grad, expected_grad, atol=1e-14)

    expected_Hess = np.zeros((6, 6))
    # d2H/dI^2 block
    expected_Hess[0, 0] = 2.0  # d2H/dI1^2
    expected_Hess[1, 1] = 6.0  # d2H/dI2^2
    expected_Hess[0, 1] = expected_Hess[1, 0] = np.cos(0)  # d2H/dI1dI2

    # d2H/dth^2 block
    expected_Hess[3, 3] = -1 * 1 * np.cos(0)  # d2H/dth1^2
    expected_Hess[4, 4] = -4 * 1 * 1 * np.cos(0)  # d2H/dth2^2
    expected_Hess[3, 4] = expected_Hess[4, 3] = 2 * 1 * 1 * np.cos(0) # d2H/dth1dth2

    # d2H/dIdth block is zero at th=0
    np.testing.assert_allclose(Hess, expected_Hess, atol=1e-14)

    expected_A = _symplectic_J() @ expected_Hess
    np.testing.assert_allclose(A, expected_A, atol=1e-14)


def test_eigenvalues_flow(fourier_data):
    """Test the eigenvalue calculation of the linearized flow."""
    coeffs_list, clmoF = fourier_data
    I_vals = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    theta_vals = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    eigvals = _eigenvalues_flow(coeffs_list, I_vals, theta_vals, clmoF, sort=True)

    _, _, _, A = _linearise_hamiltonian(coeffs_list, I_vals, theta_vals, clmoF)
    expected_eigvals = np.linalg.eigvals(A)
    expected_eigvals = np.sort_complex(expected_eigvals)

    np.testing.assert_allclose(eigvals, expected_eigvals, atol=1e-14)


def test_find_equilibrium_newton(fourier_data):
    """Test finding an equilibrium point using Newton's method."""
    coeffs_list, clmoF = fourier_data

    # Analytical solution for the equilibrium point of the test Hamiltonian
    # H = I_1 + 2*I_2 + I_1^2 + 3*I_2^2 + I_1*I_2*cos(th_1 - 2*th_2)
    # The gradient is zero when sin(th_1 - 2*th_2) = 0, so let's choose th_1=th_2=0.
    # This leads to a linear system for I_1, I_2:
    # 1 + 2*I_1 + I_2 = 0
    # 2 + I_1 + 6*I_2 = 0
    # Solution: I_1 = -4/11, I_2 = -3/11
    I_sol = np.array([-4.0 / 11.0, -3.0 / 11.0, 0.0])
    theta_sol = np.array([0.0, 0.0, 0.0])

    # Initial guess close to the solution
    I_guess = np.array([-0.3, -0.2, 0.0])
    theta_guess = np.array([0.1, 0.05, 0.0])

    I_k, theta_k, converged, num_iter = _find_equilibrium(
        coeffs_list, clmoF, I_guess, theta_guess
    )

    assert converged
    assert num_iter < 20
    np.testing.assert_allclose(I_k, I_sol, atol=1e-10)

    # Verify that the gradient is indeed close to zero at the solution
    _, grad, _, _ = _linearise_hamiltonian(coeffs_list, I_k, theta_k, clmoF)
    assert np.linalg.norm(grad) < 1e-10


def test_equilibrium_with_energy(fourier_data):
    """Test finding an equilibrium point for a target energy."""
    coeffs_list, clmoF = fourier_data

    # From test_find_equilibrium_newton, we know there's an equilibrium at:
    I_sol = np.array([-4.0 / 11.0, -3.0 / 11.0, 0.0])
    theta_sol = np.array([0.0, 0.0, 0.0])
    # The energy at this point is H = -5/11
    target_energy = -5.0 / 11.0

    # Initial guess (perturbed solution)
    I_guess = I_sol * 0.9
    theta_guess = np.array([0.1, 0.1, 0.0])
    E_guess = target_energy * 1.1
    I_theta_E_guess = np.concatenate((I_guess, theta_guess, [E_guess]))

    I_k, theta_k, converged, num_iter = _equilibrium_with_energy(
        coeffs_list, clmoF, I_theta_E_guess, target_energy, tol=1e-12
    )

    assert converged
    assert num_iter < 20
    np.testing.assert_allclose(I_k, I_sol, atol=1e-10)
    # Angles can be tricky due to wrapping, but should be close to 0 or 2*pi
    # For this Hamiltonian, equilibrium is on the line th_1 - 2*th_2 = 0.
    # So we check if the solution lies on this line.
    assert np.isclose(theta_k[0] - 2 * theta_k[1], 0.0, atol=1e-10)

    # Verify that the solution respects the energy constraint
    H_val, grad, _, _ = _linearise_hamiltonian(coeffs_list, I_k, theta_k, clmoF)
    assert np.isclose(H_val, target_energy, atol=1e-12)
    assert np.linalg.norm(grad) < 1e-10
