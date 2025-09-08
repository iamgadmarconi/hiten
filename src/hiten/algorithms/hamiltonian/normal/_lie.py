"""Provide full normal form computation using Lie series transformations.

This module provides JIT-compiled routines for computing full normal forms
of polynomial Hamiltonian systems using Lie series transformations. Unlike
partial normal forms used for center manifold reduction, full normalization
eliminates all non-resonant terms according to the complete resonance condition.

The full normal form produces maximally simplified Hamiltonian systems where
only resonant terms remain, providing the sparsest possible representation
while preserving all essential dynamical features.

All functions are optimized with Numba JIT compilation for high-performance
numerical computation of high-order polynomial transformations.

References
----------
Jorba, A. (1999). A methodology for the numerical computation of normal forms,
centre manifolds and first integrals of Hamiltonian systems.
*Experimental Mathematics*, 8(2), 155-195.
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
    degree: int, 
    tol: float = 1e-30,
    resonance_tol: float = 1e-14
) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""Perform full Lie series normalization of polynomial Hamiltonian.

    Eliminates all non-resonant terms using the complete resonance condition
    (k, omega) = 0, producing the maximally simplified canonical form.

    Parameters
    ----------
    point : object
        Linear dynamics object with `linear_modes` attribute containing
        (lambda, omega1, omega2) frequency values.
    poly_init : List[ndarray]
        Initial polynomial Hamiltonian coefficients by degree.
    psi : ndarray
        Combinatorial lookup table from polynomial indexing.
    clmo : ndarray
        Coefficient layout mapping objects.
    degree : int
        Maximum polynomial degree to include.
    tol : float, optional
        Tolerance for cleaning small coefficients. Default is 1e-30.
    resonance_tol : float, optional
        Tolerance for identifying resonant terms. Default is 1e-14.
        
    Returns
    -------
    poly_trans : List[ndarray]
        Fully normalized Hamiltonian containing only resonant terms.
    poly_G_total : List[ndarray]
        Complete generating function for the normalization.
    poly_elim_total : List[ndarray]
        Eliminated terms at each degree.
        
    Notes
    -----
    Uses the full resonance condition (k, omega) = 0 where k is the multi-index
    and omega = [lambda, -lambda, i*omega1, -i*omega1, i*omega2, -i*omega2].
    
    Differs from partial normal form by eliminating all non-resonant terms
    rather than just those with k[0] != k[3]. May encounter small divisor
    problems for nearly resonant terms.
    
    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.normal._lie._select_nonresonant_terms` : Term selection
    :func:`~hiten.algorithms.hamiltonian.center._lie._lie_transform` : Partial version
    """

    # Extract frequencies - for full normal form we need all frequencies
    lam, om1, om2 = point.linear_modes
    omega = np.array([lam, -lam, 1j*om1, -1j*om1, 1j*om2, -1j*om2], dtype=np.complex128)
    eta = np.array([omega[0], omega[2], omega[4]], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_trans = [h.copy() for h in poly_init]
    poly_G_total = _polynomial_zero_list(degree, psi)
    poly_elim_total = _polynomial_zero_list(degree, psi)  # Store eliminated terms

    # Track small divisors encountered
    small_divisors_log = []

    for n in range(3, degree+1):
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
            poly_trans_typed, p_G_n, n, degree, psi, clmo, encode_dict_list, tol
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
    r"""Identify non-resonant terms using full resonance condition.
    
    JIT-compiled function that selects terms for elimination based on
    the complete frequency analysis (k, omega) = 0.
    
    Parameters
    ----------
    p_n : ndarray
        Coefficient array for homogeneous terms of degree n.
    n : int
        Degree of the homogeneous terms.
    omega : ndarray, shape (6,)
        Frequency vector [lambda, -lambda, i*omega1, -i*omega1, 
        i*omega2, -i*omega2].
    clmo : ndarray
        Coefficient layout mapping objects.
    resonance_tol : float, optional
        Tolerance for identifying resonant terms. Default is 1e-14.
        
    Returns
    -------
    ndarray
        Coefficient array with only non-resonant terms.
        
    Notes
    -----
    A term with multi-index k = [k0, k1, k2, k3, k4, k5] is resonant if:
    
    (k, omega) = (k3-k0)*lambda + (k4-k1)*i*omega1 + (k5-k2)*i*omega2 ~ 0
    
    Unlike partial normalization (k[0] = k[3]), this considers all frequency
    combinations for more complete term elimination.
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.base._decode_multiindex` : Multi-index decoding
    :func:`~hiten.algorithms.hamiltonian.center._lie._select_terms_for_elimination` : Partial version
    """
    p_elim = p_n.copy()
    for i in range(p_n.shape[0]):
        if p_elim[i] != 0.0:
            k = _decode_multiindex(i, n, clmo)
            # Compute full resonance condition: (k, omega)
            # = (k3-k0)*lambda + (k4-k1)*i*omega1 + (k5-k2)*i*omega2
            resonance_value = ((k[3] - k[0]) * omega[0] + 
                             (k[4] - k[1]) * omega[2] + 
                             (k[5] - k[2]) * omega[4])
            if abs(resonance_value) < resonance_tol:
                p_elim[i] = 0.0
    
    return p_elim

