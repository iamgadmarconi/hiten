r"""
center.lie
==========

Numba-accelerated helpers for Lie-series based normalization of polynomial
Hamiltonians in the center-manifold reduction of the spatial restricted three-body
problem (RTBP).

References
----------
Jorba, À. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".
"""

import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                        _factorial, decode_multiindex,
                                        make_poly)
from hiten.algorithms.polynomial.operations import (polynomial_clean,
                                              polynomial_evaluate,
                                              polynomial_poisson_bracket,
                                              polynomial_total_degree,
                                              polynomial_zero_list)
from hiten.utils.config import FASTMATH
from hiten.utils.log_config import logger


def _lie_transform(
point, 
poly_init: List[np.ndarray], 
psi: np.ndarray, 
clmo: np.ndarray, 
max_degree: int, 
tol: float = 1e-30) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""
    Perform a Lie transformation to normalize a Hamiltonian.
    
    Parameters
    ----------
    point : object
        Object containing information about the linearized dynamics
        (eigenvalues and frequencies)
    poly_init : List[np.ndarray]
        Initial polynomial Hamiltonian to normalize
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
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
        - The normalized Hamiltonian
        - The generating function for the normalization
        - The eliminated terms at each degree (for testing homological equation)
        
    Notes
    -----
    This function implements Lie series normalization, which systematically 
    eliminates non-resonant terms in the Hamiltonian degree by degree.
    At each degree n, it:
    1. Identifies non-resonant terms to eliminate
    2. Solves the homological equation to find a generating function
    3. Applies the Lie transform to modify the Hamiltonian
    
    The transformation preserves the dynamical structure while simplifying
    the equations of motion.
    """
    lam, om1, om2 = point.linear_modes
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_trans = [h.copy() for h in poly_init]
    poly_G_total = polynomial_zero_list(max_degree, psi)
    poly_elim_total = polynomial_zero_list(max_degree, psi)  # Store eliminated terms

    for n in range(3, max_degree+1):
        logger.info(f"Normalizing at order: {n}")
        p_n = poly_trans[n]
        if not p_n.any():
            continue
        p_elim = _select_terms_for_elimination(p_n, n, clmo)
        if not p_elim.any():
            continue
            
        # Store the eliminated terms for this degree
        if n < len(poly_elim_total):
            poly_elim_total[n] = p_elim.copy()
            
        p_G_n = _solve_homological_equation(p_elim, n, eta, clmo)

        
        # Clean Gn using a Numba typed list for compatibility with polynomial_clean
        if p_G_n.any(): # Only clean if there's something to clean
            temp_G_n_list = List()
            temp_G_n_list.append(p_G_n)
            cleaned_G_n_list = polynomial_clean(temp_G_n_list, tol)
            p_G_n = cleaned_G_n_list[0]

        # Pass the cleaned Gn to _apply_poly_transform
        # Convert poly_trans to Numba typed list for _apply_poly_transform
        poly_trans_typed = List()
        for item_arr in poly_trans:
            poly_trans_typed.append(item_arr)
        # _apply_poly_transform expects a Numba List for poly_H and returns a Python list
        poly_trans = _apply_poly_transform(poly_trans_typed, p_G_n, n, max_degree, psi, clmo, encode_dict_list, tol)
        
        if n < len(poly_G_total) and poly_G_total[n].shape == p_G_n.shape:
            poly_G_total[n] += p_G_n
        elif n < len(poly_G_total) and poly_G_total[n].size == p_G_n.size:
            poly_G_total[n] += p_G_n.reshape(poly_G_total[n].shape)

        if not _select_terms_for_elimination(poly_trans[n], n, clmo).any():
            continue
            
    poly_G_total = polynomial_clean(poly_G_total, tol)
    poly_elim_total = polynomial_clean(poly_elim_total, tol)
    return poly_trans, poly_G_total, poly_elim_total


@njit(fastmath=FASTMATH, cache=False)
def _get_homogeneous_terms(
poly_H: List[np.ndarray],
n: int, 
psi: np.ndarray) -> np.ndarray:
    r"""
    Extract the homogeneous terms of degree n from a polynomial.
    
    Parameters
    ----------
    poly_H : List[numpy.ndarray]
        List of coefficient arrays representing a polynomial
    n : int
        Degree of the homogeneous terms to extract
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
        
    Returns
    -------
    numpy.ndarray
        Coefficient array for the homogeneous part of degree n
        
    Notes
    -----
    If the polynomial doesn't have terms of degree n, an empty array
    of the appropriate size is returned.
    """
    if n < len(poly_H):
        result = poly_H[n].copy()
    else:
        result = make_poly(n, psi)
    return result


@njit(fastmath=FASTMATH, cache=False)
def _select_terms_for_elimination(
p_n: np.ndarray, 
n: int, 
clmo: np.ndarray) -> np.ndarray:
    r"""
    Select non-resonant terms to be eliminated by the Lie transform.
    
    Parameters
    ----------
    p_n : numpy.ndarray
        Coefficient array for the homogeneous part of degree n
    n : int
        Degree of the homogeneous terms
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    numpy.ndarray
        Coefficient array containing only the non-resonant terms
        
    Notes
    -----
    This function identifies "bad" monomials which are non-resonant terms
    that need to be eliminated. A term is resonant if k[0] = k[3],
    meaning the powers of the center variables are equal.
    """
    p_elim = p_n.copy()           # independent buffer
    for i in range(p_n.shape[0]):
        if p_elim[i] != 0.0:     # skip explicit zeros
            k = decode_multiindex(i, n, clmo)
            if k[0] == k[3]:   # not a "bad" monomial -> zero it
                p_elim[i] = 0.0
    return p_elim


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
        Array containing the eigenvalues [λ, iω₁, iω₂]
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
    
    g_k = -h_k / ((k₃-k₀)λ + (k₄-k₁)iω₁ + (k₅-k₂)iω₂)
    
    where k = [k₀, k₁, k₂, k₃, k₄, k₅] are the exponents of the monomial.
    
    This version includes a resonance check to avoid division by zero
    for nearly resonant terms.
    """
    p_G = np.zeros_like(p_elim)
    for i in range(p_elim.shape[0]):
        c = p_elim[i]
        if c != 0.0:
            k = decode_multiindex(i, n, clmo)
            denom = ((k[3]-k[0]) * eta[0] +
                     (k[4]-k[1]) * eta[1] +
                     (k[5]-k[2]) * eta[2])
            # Check for resonance (near-zero denominator)
            if abs(denom) < 1e-14:
                continue  # Skip resonant terms - keep in normal form, should not occur.
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
        Combinatorial table from init_index_tables
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
            poly_result.append(make_poly(i, psi))
    
    # Build complete generator polynomial from single degree
    poly_G = polynomial_zero_list(N_max, psi)
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
        poly_bracket = polynomial_poisson_bracket(
            poly_bracket,
            poly_G,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )
        poly_bracket = polynomial_clean(poly_bracket, tol)
        
        # Add to result with factorial coefficient
        coeff = 1.0 / factorials[k]
        for d in range(min(len(poly_bracket), len(poly_result))):
            poly_result[d] += coeff * poly_bracket[d]
    
    return polynomial_clean(poly_result, tol)


def _lie_expansion(
poly_G_total: List[np.ndarray], 
max_degree: int, psi: np.ndarray, 
clmo: np.ndarray, 
tol: float = 1e-30,
inverse: bool = False, # If False, Generators are applied in ascending order. If True, Generators are applied in descending order.
sign: int = None, # If None, the sign is determined by the inverse flag. If not None, the sign is used to determine sign of the generator.
restrict: bool = True) -> List[List[np.ndarray]]:
    r"""
    Perform inverse Lie transformation from center manifold coordinates to complex-diagonalized coordinates.
    
    Parameters
    ----------
    poly_G_total : List[np.ndarray]
        List of generating functions for the Lie transformation
    max_degree : int
        Maximum polynomial degree for the transformation
    psi : np.ndarray
        Combinatorial table from init_index_tables
    clmo : np.ndarray
        List of arrays containing packed multi-indices
    tol : float, optional
        Tolerance for cleaning small coefficients

    Returns
    -------
    List[List[np.ndarray]]
        Six polynomial expansions for [q1, q2, q3, p1, p2, p3]
    """
    # Create encode_dict_list from clmo
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    current_coords = []
    for i in range(6):
        poly = polynomial_zero_list(max_degree, psi)
        poly[1] = np.zeros(6, dtype=np.complex128)
        poly[1][i] = 1.0 + 0j       # identity for q₁,q₂,q₃,p₁,p₂,p₃
        current_coords.append(poly) # [q1, q2, q3, p1, p2, p3]
    
    if inverse:
        start = max_degree
        stop = 2
        step = -1
        sign = -1 if sign is None else sign
    else:
        start = 3
        stop = max_degree + 1
        step = 1
        sign = 1 if sign is None else sign

    for n in range(start, stop, step):
        if n >= len(poly_G_total) or not poly_G_total[n].any():
            continue

        G_n = sign * poly_G_total[n]
        poly_G = polynomial_zero_list(max_degree, psi)
        poly_G[n] = G_n.copy()
        
        new_coords = []
        for i in range(6):
            current_poly_typed = List()
            for arr in current_coords[i]:
                current_poly_typed.append(arr)

            new_poly = _apply_coord_transform(
                current_poly_typed, poly_G, max_degree, psi, clmo, encode_dict_list, tol
            )
            new_coords.append(new_poly)
        
        # Update all coordinates for next iteration
        current_coords = new_coords
    
    # Convert to proper Numba List[List[np.ndarray]] before returning
    result = List()
    for coord_expansion in current_coords:
        result.append(coord_expansion)
    
    if restrict:
        result = _zero_q1p1(result, clmo, tol)

    return result


@njit(fastmath=FASTMATH, cache=False)
def _apply_coord_transform(
poly_X: List[np.ndarray], 
poly_G: List[np.ndarray], 
N_max: int, 
psi: np.ndarray, 
clmo: np.ndarray, 
encode_dict_list: List[dict], 
tol: float) -> List[np.ndarray]:
    r"""
    Apply inverse Lie series transformation to a coordinate polynomial.
    
    Parameters
    ----------
    poly_X : List[np.ndarray]
        Current coordinate polynomial
    poly_G : List[np.ndarray]
        Generating function polynomial
    N_max : int
        Maximum degree for the result
    psi, clmo, encode_dict_list : arrays
        Polynomial indexing structures
    tol : float
        Tolerance for cleaning
        
    Returns
    -------
    list[np.ndarray]
        Transformed coordinate polynomial
    """

    poly_result = List()
    for i in range(N_max + 1):
        if i < len(poly_X):
            poly_result.append(poly_X[i].copy())
        else:
            poly_result.append(make_poly(i, psi))

    # Find degree of generating function
    deg_G = polynomial_total_degree(poly_G, psi)

    if deg_G > 2:
        K_max = max(N_max, (N_max - 1) // (deg_G - 2) + 1)
    else:
        K_max = 1
    
    # Precompute factorials
    factorials = [_factorial(k) for k in range(K_max + 1)]
    
    # Initialize bracket with X for iteration
    poly_bracket = List()
    for i in range(len(poly_X)):
        poly_bracket.append(poly_X[i].copy())
    
    # Apply Lie series: X + {X,G} + (1/2!){{X,G},G} + ...
    for k in range(1, K_max + 1):

        # Compute next Poisson bracket
        poly_bracket = polynomial_poisson_bracket(
            poly_bracket,
            poly_G,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )

        poly_bracket = polynomial_clean(poly_bracket, tol)

        coeff = 1.0 / factorials[k]
        for d in range(min(len(poly_bracket), len(poly_result))):
            poly_result[d] += coeff * poly_bracket[d]

    return polynomial_clean(poly_result, tol)


@njit(fastmath=FASTMATH, cache=False)
def _evaluate_transform(
expansions: List[List[np.ndarray]], 
coords_cm_complex: np.ndarray, 
clmo: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the six polynomial expansions at given center manifold values.
    
    Parameters
    ----------
    expansions : List[List[np.ndarray]]
        Six polynomial expansions from inverse_lie_transform
    coords_cm_complex : np.ndarray
        Center manifold coordinates [q1, q2, q3, p1, p2, p3]
    clmo : np.ndarray
        List of arrays containing packed multi-indices
        
    Returns
    -------
    np.ndarray
        Complex array of shape (6,) containing [q̃1, q̃2, q̃3, p̃1, p̃2, p̃3]
    """

    result = np.zeros(6, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]
    
    for i in range(6):
        # Evaluate each polynomial at the given point
        result[i] = polynomial_evaluate(expansions[i], coords_cm_complex, clmo)
    
    return result # [q̃1, q̃2, q̃3, p̃1, p̃2, p̃3]


def _zero_q1p1(
    expansions: List[List[np.ndarray]], 
    clmo: np.ndarray, 
    tol: float = 1e-30
) -> List[List[np.ndarray]]:
    r"""
    Restrict coordinate expansions to the center manifold by eliminating 
    terms containing q1 or p1.
    
    After this restriction, all 6 coordinate expansions will depend only 
    on the 4 center manifold variables (q2, p2, q3, p3).
    """
    restricted_expansions = List()
    
    for expansion in expansions:
        # Create a new Numba List to maintain type consistency
        restricted_poly = List()
        for h in expansion:
            restricted_poly.append(h.copy())
            
        for deg, coeff_vec in enumerate(restricted_poly):
            if coeff_vec.size == 0:
                continue
            for pos, c in enumerate(coeff_vec):
                if abs(c) <= tol:
                    coeff_vec[pos] = 0.0
                    continue
                k = decode_multiindex(pos, deg, clmo)
                if k[0] != 0 or k[3] != 0:  # q1 or p1 exponent non-zero
                    coeff_vec[pos] = 0.0
        restricted_expansions.append(restricted_poly)
    
    return restricted_expansions