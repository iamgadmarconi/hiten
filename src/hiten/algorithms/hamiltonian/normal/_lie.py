r"""
hamiltonian.normal._lie
====================

Numba-accelerated helpers for Lie-series based full normalization of polynomial
Hamiltonians, following the algorithm described in Jorba (1999).

This implements the full normal form, eliminating all non-resonant terms
rather than just the partial normal form used for center manifold reduction.

References
----------
Jorba, À. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".
"""

import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.hamiltonian.lie import (_apply_poly_transform,
                                              _solve_homological_equation)
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _decode_multiindex)
from hiten.algorithms.polynomial.operations import (_polynomial_clean,
                                                    _polynomial_zero_list)
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


def _lie_transform(
    point, 
    poly_init: List[np.ndarray], 
    psi: np.ndarray, 
    clmo: np.ndarray, 
    max_degree: int, 
    tol: float = 1e-30,
    resonance_tol: float = 1e-14
) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""
    Perform a full Lie transformation to normalize a Hamiltonian.
    
    This implements the full normal form, eliminating all non-resonant terms
    according to the resonance condition :math:`(k, \omega) = 0`.
    
    Parameters
    ----------
    point : object
        Object containing information about the linearized dynamics
        (eigenvalues and frequencies)
    poly_init : List[np.ndarray]
        Initial polynomial Hamiltonian to normalize
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    max_degree : int
        Maximum degree to include in the normalized Hamiltonian
    tol : float, optional
        Tolerance for cleaning small coefficients, default is 1e-30
        
    Returns
    -------
    tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        A tuple containing:
        - The normalized Hamiltonian (containing only resonant terms)
        - The generating function for the normalization
        - The eliminated terms at each degree (for testing homological equation)
        
    Notes
    -----
    This implements the full normal form algorithm from Jorba (1999), which
    systematically eliminates all non-resonant terms degree by degree.
    
    The main differences from the partial normal form are:
    1. Uses full resonance condition (k,ω) = 0 instead of k[0] = k[3]
    2. Results in a much sparser normal form containing only resonant terms
    3. Requires careful handling of small divisors
    """

    # Extract frequencies - for full normal form we need all frequencies
    lam, om1, om2 = point.linear_modes
    omega = np.array([lam, -lam, 1j*om1, -1j*om1, 1j*om2, -1j*om2], dtype=np.complex128)
    eta = np.array([omega[0], omega[2], omega[4]], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_trans = [h.copy() for h in poly_init]
    poly_G_total = _polynomial_zero_list(max_degree, psi)
    poly_elim_total = _polynomial_zero_list(max_degree, psi)  # Store eliminated terms

    # Track small divisors encountered
    small_divisors_log = []

    for n in range(3, max_degree+1):
        logger.info(f"Full normalization at order: {n}")
        p_n = poly_trans[n]
        if not p_n.any():
            continue
        p_elim = _select_nonresonant_terms(p_n, n, omega, clmo, resonance_tol)
        if not p_elim.any():
            logger.info(f"  No non-resonant terms at degree {n}")
            continue
            
        # Store the eliminated terms for this degree
        if n < len(poly_elim_total):
            poly_elim_total[n] = p_elim.copy()
            
        # Solve homological equation with small divisor handling
        p_G_n = _solve_homological_equation(
            p_elim, n, eta, clmo
        )

        # Clean Gn
        if p_G_n.any():
            temp_G_n_list = List()
            temp_G_n_list.append(p_G_n)
            cleaned_G_n_list = _polynomial_clean(temp_G_n_list, tol)
            p_G_n = cleaned_G_n_list[0]

        # Apply the Lie transform
        poly_trans_typed = List()
        for item_arr in poly_trans:
            poly_trans_typed.append(item_arr)
            
        poly_trans = _apply_poly_transform(
            poly_trans_typed, p_G_n, n, max_degree, psi, clmo, encode_dict_list, tol
        )
        
        if n < len(poly_G_total) and poly_G_total[n].shape == p_G_n.shape:
            poly_G_total[n] += p_G_n
        elif n < len(poly_G_total) and poly_G_total[n].size == p_G_n.size:
            poly_G_total[n] += p_G_n.reshape(poly_G_total[n].shape)

        # Verify that non-resonant terms were eliminated
        p_check = _select_nonresonant_terms(poly_trans[n], n, omega, clmo, resonance_tol)
        if p_check.any():
            logger.warning(f"  Warning: Some non-resonant terms remain at degree {n}")
            
    if small_divisors_log:
        logger.info(f"Total small divisors encountered: {len(small_divisors_log)}")
    
    poly_G_total = _polynomial_clean(poly_G_total, tol)
    poly_elim_total = _polynomial_clean(poly_elim_total, tol)
    
    return poly_trans, poly_G_total, poly_elim_total


@njit(fastmath=FASTMATH, cache=False)
def _select_nonresonant_terms(
    p_n: np.ndarray, 
    n: int, 
    omega: np.ndarray,
    clmo: np.ndarray,
    resonance_tol: float = 1e-14) -> np.ndarray:
    r"""
    Select non-resonant terms to be eliminated by the full normal form.
    
    Parameters
    ----------
    p_n : numpy.ndarray
        Coefficient array for the homogeneous part of degree n
    n : int
        Degree of the homogeneous terms
    omega : numpy.ndarray
        Array of frequencies :math:`[\omega_1, -\omega_1, \omega_2, -\omega_2, \omega_3, -\omega_3]`
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
    resonance_tol : float
        Tolerance for identifying resonant terms
        
    Returns
    -------
    numpy.ndarray
        Coefficient array containing only the non-resonant terms
    """
    p_elim = p_n.copy()
    for i in range(p_n.shape[0]):
        if p_elim[i] != 0.0:
            k = _decode_multiindex(i, n, clmo)
            resonance_value = ((k[3] - k[0]) * omega[0] + 
                             (k[4] - k[1]) * omega[2] + 
                             (k[5] - k[2]) * omega[4])
            if abs(resonance_value) < resonance_tol:
                p_elim[i] = 0.0
    
    return p_elim

