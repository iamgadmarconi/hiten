r"""
hamiltonian.lie
==============

Base implementation of Lie-series based normalization of polynomial. Used by
parital and full normal forms.
"""


import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.polynomial.base import (_factorial, _decode_multiindex,
                                        _make_poly)
from hiten.algorithms.polynomial.operations import (_polynomial_clean,
                                              _polynomial_poisson_bracket,
                                              _polynomial_zero_list)
from hiten.algorithms.utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _solve_homological_equation(
p_elim: np.ndarray, 
n: int, 
eta: np.ndarray, 
clmo: np.ndarray) -> np.ndarray:
    r"""
    Solve the homological equation to find the generating function.
    
    Parameters
    ----------
    p_elim : numpy.ndarray
        Coefficient array containing the terms to be eliminated
    n : int
        Degree of the homogeneous terms
    eta : numpy.ndarray
        Array containing the eigenvalues :math:`[\lambda, i\omega_1, i\omega_2]`
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    numpy.ndarray
        Coefficient array for the generating function of degree n
        
    Notes
    -----
    The homological equation is solved by dividing each coefficient by
    the corresponding eigenvalue combination:
    
    :math:`g_k = -h_k / ((k_3-k_0)\lambda + (k_4-k_1)i\omega_1 + (k_5-k_2)i\omega_2)`
    
    where :math:`k = [k_0, k_1, k_2, k_3, k_4, k_5]` are the exponents of the monomial.
    """
    p_G = np.zeros_like(p_elim)
    for i in range(p_elim.shape[0]):
        c = p_elim[i]
        if c != 0.0:
            k = _decode_multiindex(i, n, clmo)
            denom = ((k[3]-k[0]) * eta[0] +
                     (k[4]-k[1]) * eta[1] +
                     (k[5]-k[2]) * eta[2])

            if abs(denom) < 1e-14:
                continue
            p_G[i] = -c / denom
    return p_G


@njit(fastmath=FASTMATH, cache=False)
def _apply_poly_transform(
poly_H: List[np.ndarray], 
p_G_n: np.ndarray, 
deg_G: int, 
N_max: int, 
psi: np.ndarray, 
clmo: np.ndarray, 
encode_dict_list: List[dict], 
tol: float) -> List[np.ndarray]:
    r"""
    Apply a Lie transform with generating function G to a Hamiltonian.
    
    Parameters
    ----------
    poly_H : List[numpy.ndarray]
        Original Hamiltonian polynomial
    p_G_n : numpy.ndarray
        Coefficient array for the generating function of degree deg_G
    deg_G : int
        Degree of the generating function
    N_max : int
        Maximum degree to include in the transformed Hamiltonian
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
    encode_dict_list : List
        List of dictionaries mapping packed multi-indices to their positions
    tol : float
        Tolerance for cleaning small coefficients
        
    Returns
    -------
    list[numpy.ndarray]
        The transformed Hamiltonian polynomial
        
    Notes
    -----
    This function implements the Lie transform:
    
    H' = exp(L_G) H = H + {G,H} + 1/2!{{G,H},G} + 1/3!{{{G,H},G},G} + ...
    
    where L_G is the Lie operator associated with G, and {,} denotes the Poisson bracket.
    
    The sum is truncated based on the maximum achievable degree from repeated
    Poisson brackets and the specified N_max.
    """
    # Initialize result by copying input polynomial
    poly_result = List()
    for i in range(N_max + 1):
        if i < len(poly_H):
            poly_result.append(poly_H[i].copy())
        else:
            poly_result.append(_make_poly(i, psi))
    
    # Build complete generator polynomial from single degree
    poly_G = _polynomial_zero_list(N_max, psi)
    if deg_G < len(poly_G):
        poly_G[deg_G] = p_G_n.copy()
    
    # Determine number of terms in Lie series
    if deg_G > 2:
        K = max(N_max, (N_max - deg_G) // (deg_G - 2) + 1)
    else:
        K = 1
    
    # Precompute factorials
    factorials = [_factorial(k) for k in range(K + 1)]
    
    # Initialize with H for Poisson bracket iteration
    poly_bracket = List()
    for i in range(len(poly_H)):
        poly_bracket.append(poly_H[i].copy())
    
    # Apply Lie series: H + {H,G} + (1/2!){{H,G},G} + ...
    for k in range(1, K + 1):
        # Compute next Poisson bracket
        poly_bracket = _polynomial_poisson_bracket(
            poly_bracket,
            poly_G,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )
        poly_bracket = _polynomial_clean(poly_bracket, tol)
        
        # Add to result with factorial coefficient
        coeff = 1.0 / factorials[k]
        for d in range(min(len(poly_bracket), len(poly_result))):
            poly_result[d] += coeff * poly_bracket[d]
    
    return _polynomial_clean(poly_result, tol)
