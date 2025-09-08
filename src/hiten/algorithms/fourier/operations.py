"""High-level operations on Fourier-Taylor polynomials.

Provide construction, evaluation, gradient, and Hessian routines for
lists of homogeneous blocks representing Fourier-Taylor polynomials in
action-angle variables. The low-level kernels live in
:mod:`hiten.algorithms.fourier.algebra`.
"""

import numpy as np
from numba import njit

from hiten.algorithms.fourier.algebra import (_fpoly_block_evaluate,
                                              _fpoly_block_gradient,
                                              _fpoly_block_hessian)
from hiten.algorithms.utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _make_fourier_poly(degree: int, psiF: np.ndarray):  
    """Create a zero-initialized homogeneous block of a given degree.

    Parameters
    ----------
    degree : int
        Total action degree of the block.
    psiF : numpy.ndarray
        Lookup array where ``psiF[d]`` is the size of the degree ``d`` block.

    Returns
    -------
    numpy.ndarray
        Zero-filled coefficient array of length ``psiF[degree]`` with dtype
        ``complex128``.
    """
    size = psiF[degree]
    return np.zeros(size, dtype=np.complex128)


@njit(fastmath=FASTMATH, cache=False)
def _fourier_evaluate(
    coeffs_list,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    """Evaluate a Fourier-Taylor polynomial at ``(I, theta)``.

    Parameters
    ----------
    coeffs_list : numba.typed.List
        List of homogeneous blocks by degree starting at 0.
    I_vals : numpy.ndarray
        Actions ``[I1, I2, I3]``.
    theta_vals : numpy.ndarray
        Angles ``[theta1, theta2, theta3]``.
    clmoF : numba.typed.List
        Packed index arrays per degree.

    Returns
    -------
    complex
        Complex value at the specified point.
    """
    val = 0.0 + 0.0j
    max_deg = len(coeffs_list) - 1
    for d in range(max_deg + 1):
        block = coeffs_list[d]
        if block.shape[0]:
            val += _fpoly_block_evaluate(block, d, I_vals, theta_vals, clmoF)
    return val


@njit(fastmath=FASTMATH, cache=False)
def _fourier_evaluate_with_grad(
    coeffs_list,               # numba.typed.List[np.ndarray]
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    """Evaluate value and gradients of a polynomial at ``(I, theta)``.

    Parameters
    ----------
    coeffs_list : numba.typed.List
        List of homogeneous blocks by degree starting at 0.
    I_vals : numpy.ndarray
        Actions ``[I1, I2, I3]``.
    theta_vals : numpy.ndarray
        Angles ``[theta1, theta2, theta3]``.
    clmoF : numba.typed.List
        Packed index arrays per degree.

    Returns
    -------
    tuple
        ``(val, gI, gT)`` where ``val`` is complex, and ``gI``, ``gT`` are
        arrays of shape ``(3,)`` with dtype ``complex128``.
    """
    val = 0.0 + 0.0j
    gI = np.zeros(3, dtype=np.complex128)
    gT = np.zeros(3, dtype=np.complex128)

    max_deg = len(coeffs_list) - 1
    for d in range(max_deg + 1):
        block = coeffs_list[d]
        if block.shape[0]:
            v_b, gI_b, gT_b = _fpoly_block_gradient(block, d, I_vals, theta_vals, clmoF)
            val += v_b
            for j in range(3):
                gI[j] += gI_b[j]
                gT[j] += gT_b[j]

    return val, gI, gT


@njit(fastmath=FASTMATH, cache=False)
def _fourier_hessian(
    coeffs_list,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    """Compute the 6x6 Hessian at ``(I, theta)`` for a polynomial.

    Variables are ordered ``[I1, I2, I3, theta1, theta2, theta3]``.

    Parameters
    ----------
    coeffs_list : numba.typed.List
        List of homogeneous blocks by degree starting at 0.
    I_vals : numpy.ndarray
        Actions ``[I1, I2, I3]``.
    theta_vals : numpy.ndarray
        Angles ``[theta1, theta2, theta3]``.
    clmoF : numba.typed.List
        Packed index arrays per degree.

    Returns
    -------
    numpy.ndarray
        ``(6, 6)`` complex Hessian matrix.
    """
    H_total = np.zeros((6, 6), dtype=np.complex128)
    max_deg = len(coeffs_list) - 1
    for d in range(max_deg + 1):
        block = coeffs_list[d]
        if block.shape[0]:
            H_block = _fpoly_block_hessian(block, d, I_vals, theta_vals, clmoF)
            # Accumulate
            for i in range(6):
                for j in range(6):
                    H_total[i, j] += H_block[i, j]
    return H_total
