from __future__ import annotations

import numpy as np

from hiten.algorithms.fourier.operations import (_fourier_evaluate_with_grad,
                                                 _fourier_hessian)


def _symplectic_J() -> np.ndarray:
    J = np.zeros((6, 6), dtype=np.complex128)
    J[0:3, 3:6] = -np.identity(3)
    J[3:6, 0:3] = np.identity(3)
    return J


def _linearise_hamiltonian(
    coeffs_list,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    H_val, gI, gT = _fourier_evaluate_with_grad(coeffs_list, I_vals, theta_vals, clmoF)
    Hess = _fourier_hessian(coeffs_list, I_vals, theta_vals, clmoF)
    grad = np.concatenate((gI, gT))
    A = _symplectic_J() @ Hess
    return H_val, grad, Hess, A


def _eigenvalues_flow(
    coeffs_list,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
    sort: bool = True,
):
    _, _, _, A = _linearise_hamiltonian(coeffs_list, I_vals, theta_vals, clmoF)
    eigvals = np.linalg.eigvals(A)
    if sort:
        eigvals = np.sort_complex(eigvals)
    return eigvals


def _find_equilibrium(
    coeffs_list,
    clmoF,
    I_guess: np.ndarray,
    theta_guess: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20,
):
    I_k = np.copy(I_guess)
    theta_k = np.copy(theta_guess)

    for i in range(max_iter):
        # We need gradient and Hessian. _linearise_hamiltonian gives both.
        _, grad_c, Hess_c, _ = _linearise_hamiltonian(
            coeffs_list, I_k, theta_k, clmoF
        )

        # Work with real parts to avoid complex casting issues (imaginary parts
        # can arise from numerical noise but should be ≈0 for a real-valued Hamiltonian).
        grad = np.real_if_close(grad_c, tol=1e12)
        Hess = np.real_if_close(Hess_c, tol=1e12)

        # Check for convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            return I_k, theta_k, True, i

        # Solve for the Newton step: Hess * step = -grad (use pseudo-inverse for robustness)
        step = np.linalg.pinv(Hess) @ (-grad)

        # Update the state vector (step is real-valued)
        I_k += step[0:3]
        theta_k += step[3:6]

        # Normalize angles to (-pi, pi]
        theta_k = (theta_k + np.pi) % (2 * np.pi) - np.pi

    # Check for convergence after the loop finishes
    _, grad_c, _, _ = _linearise_hamiltonian(coeffs_list, I_k, theta_k, clmoF)
    grad = np.real_if_close(grad_c, tol=1e12)
    converged = np.linalg.norm(grad) < tol

    return I_k, theta_k, converged, max_iter


def _equilibrium_with_energy(
    coeffs_list,
    clmoF,
    I_theta_E_guess: np.ndarray,
    target_energy: float,
    tol: float = 1e-12,
    max_iter: int = 25,
):
    vec = np.copy(I_theta_E_guess)

    for it in range(max_iter):
        I_vals = vec[0:3]
        theta_vals = vec[3:6]
        E_var = vec[6]

        H_val, g_full, Hess, _ = _linearise_hamiltonian(
            coeffs_list, I_vals, theta_vals, clmoF
        )

        g_full = np.real_if_close(g_full, tol=1e12)
        Hess = np.real_if_close(Hess, tol=1e12)
        H_val = float(np.real_if_close(H_val, tol=1e12))

        # Assemble residual vector: 6 grad components + energy constraint
        F = np.empty(7, dtype=float)
        F[0:6] = g_full
        F[6] = H_val - target_energy

        res_norm = np.linalg.norm(F)
        if res_norm < tol:
            return vec[0:3], vec[3:6], True, it

        # Jacobian matrix (7x7)
        Jmat = np.zeros((7, 7), dtype=float)
        Jmat[0:6, 0:6] = Hess
        Jmat[0:6, 6] = -g_full
        Jmat[6, 0:6] = g_full
        Jmat[6, 6] = -1.0

        step = np.linalg.pinv(Jmat) @ (-F)

        vec += step
        # Normalize angles to (-pi, pi]
        vec[3:6] = (vec[3:6] + np.pi) % (2 * np.pi) - np.pi

    return vec[0:3], vec[3:6], False, max_iter
