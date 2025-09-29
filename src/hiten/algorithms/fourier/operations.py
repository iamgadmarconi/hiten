import numpy as np
from numba import njit

from hiten.algorithms.fourier.algebra import (_fpoly_block_evaluate,
                                              _fpoly_block_gradient,
                                              _fpoly_block_hessian)
from hiten.algorithms.utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _make_fourier_poly(degree: int, psiF: np.ndarray):  
    """Create a new Fourier polynomial coefficient array of specified degree.
    
    Parameters
    ----------
    degree : int
        Degree of the polynomial.
    psiF : np.ndarray
        Combinatorial table from _init_index_tables.
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
    """Evaluate a Fourier polynomial at specific action and angle values.
    
    Parameters
    ----------
    coeffs_list : List
        List of Fourier polynomial coefficient arrays.
    I_vals : np.ndarray
        Values of the actions.
    theta_vals : np.ndarray
        Values of the angles.
    clmoF : List
        List of arrays containing packed multi-indices.
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
    """Evaluate a Fourier polynomial with gradient at specific action and angle values.
    
    Parameters
    ----------
    coeffs_list : List
        List of Fourier polynomial coefficient arrays.
    I_vals : np.ndarray
        Values of the actions.
    theta_vals : np.ndarray
        Values of the angles.
    clmoF : List
        List of arrays containing packed multi-indices.
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
    """Compute the Hessian matrix of a Fourier polynomial.
    
    Parameters
    ----------
    coeffs_list : List
        List of Fourier polynomial coefficient arrays.
    I_vals : np.ndarray
        Values of the actions.
    theta_vals : np.ndarray
        Values of the angles.
    clmoF : List
        List of arrays containing packed multi-indices.
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
