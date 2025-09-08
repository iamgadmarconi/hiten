r"""Lie series transformations for Hamiltonian normal form computation.

This module provides JIT-compiled routines for performing Lie series transformations
to compute normal forms of polynomial Hamiltonian systems. The implementation follows
the methodology for center manifold reduction in the spatial Circular Restricted
Three-Body Problem (CR3BP).

The normalization process systematically eliminates non-resonant terms degree by
degree using Lie series transformations, resulting in simplified Hamiltonian
systems that retain the essential dynamics while being more amenable to analysis.

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
                                              _decode_multiindex, _factorial,
                                              _make_poly)
from hiten.algorithms.polynomial.operations import (
    _polynomial_clean, _polynomial_evaluate, _polynomial_poisson_bracket,
    _polynomial_total_degree, _polynomial_zero_list)
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


def _lie_transform(
point, 
poly_init: List[np.ndarray], 
psi: np.ndarray, 
clmo: np.ndarray, 
degree: int, 
tol: float = 1e-30) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""Perform Lie series normalization of polynomial Hamiltonian.

    Implements the partial normal form algorithm that systematically eliminates
    non-resonant terms from a polynomial Hamiltonian using Lie series transformations.
    The process preserves the Hamiltonian structure while simplifying the dynamics.

    Parameters
    ----------
    point : object
        Linear dynamics object containing eigenvalues and frequencies of the
        linearized system. Must have a `linear_modes` attribute with
        (lambda, omega1, omega2) values.
    poly_init : List[ndarray]
        Initial polynomial Hamiltonian coefficients organized by degree.
        Each element contains coefficients for homogeneous terms of that degree.
    psi : ndarray
        Combinatorial lookup table from polynomial indexing system.
    clmo : ndarray
        Coefficient layout mapping objects for polynomial operations.
    degree : int
        Maximum polynomial degree to include in normalization.
    tol : float, optional
        Tolerance for cleaning small coefficients. Default is 1e-30.
        
    Returns
    -------
    poly_trans : List[ndarray]
        Normalized Hamiltonian with non-resonant terms eliminated.
    poly_G_total : List[ndarray]
        Complete generating function used for the normalization.
    poly_elim_total : List[ndarray]
        Eliminated terms at each degree (useful for verification).
        
    Notes
    -----
    The normalization process operates degree by degree:
    
    1. **Term Selection**: Identifies non-resonant terms using resonance condition
    2. **Homological Equation**: Solves for generating function to eliminate these terms
    3. **Lie Transform**: Applies transformation to modify all polynomial terms
    4. **Iteration**: Continues until all degrees are processed
    
    A term is considered resonant if k[0] = k[3] where k is the multi-index
    of exponents. Non-resonant terms are systematically eliminated while
    preserving the essential dynamics captured by resonant terms.
    
    The transformation is canonical, preserving the symplectic structure
    of the Hamiltonian system.
    
    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.center._lie._select_terms_for_elimination` : Identifies non-resonant terms
    :func:`~hiten.algorithms.hamiltonian.lie._solve_homological_equation` : Solves for generators
    :func:`~hiten.algorithms.hamiltonian.lie._apply_poly_transform` : Applies Lie transform
    """
    lam, om1, om2 = point.linear_modes
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_trans = [h.copy() for h in poly_init]
    poly_G_total = _polynomial_zero_list(degree, psi)
    poly_elim_total = _polynomial_zero_list(degree, psi)  # Store eliminated terms

    for n in range(3, degree+1):
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

        
        # Clean Gn using a Numba typed list for compatibility with _polynomial_clean
        if p_G_n.any(): # Only clean if there's something to clean
            temp_G_n_list = List()
            temp_G_n_list.append(p_G_n)
            cleaned_G_n_list = _polynomial_clean(temp_G_n_list, tol)
            p_G_n = cleaned_G_n_list[0]

        # Pass the cleaned Gn to _apply_poly_transform
        # Convert poly_trans to Numba typed list for _apply_poly_transform
        poly_trans_typed = List()
        for item_arr in poly_trans:
            poly_trans_typed.append(item_arr)
        # _apply_poly_transform expects a Numba List for poly_H and returns a Python list
        poly_trans = _apply_poly_transform(poly_trans_typed, p_G_n, n, degree, psi, clmo, encode_dict_list, tol)
        
        if n < len(poly_G_total) and poly_G_total[n].shape == p_G_n.shape:
            poly_G_total[n] += p_G_n
        elif n < len(poly_G_total) and poly_G_total[n].size == p_G_n.size:
            poly_G_total[n] += p_G_n.reshape(poly_G_total[n].shape)

        if not _select_terms_for_elimination(poly_trans[n], n, clmo).any():
            continue
            
    poly_G_total = _polynomial_clean(poly_G_total, tol)
    poly_elim_total = _polynomial_clean(poly_elim_total, tol)
    return poly_trans, poly_G_total, poly_elim_total


@njit(fastmath=FASTMATH, cache=False)
def _get_homogeneous_terms(
poly_H: List[np.ndarray],
n: int, 
psi: np.ndarray) -> np.ndarray:
    r"""Extract homogeneous terms of specified degree from polynomial.
    
    JIT-compiled function that extracts the coefficient array corresponding
    to homogeneous terms of a specific degree from a polynomial representation.
    
    Parameters
    ----------
    poly_H : List[ndarray]
        Polynomial represented as list of coefficient arrays by degree.
    n : int
        Degree of homogeneous terms to extract.
    psi : ndarray
        Combinatorial lookup table for polynomial indexing.
        
    Returns
    -------
    ndarray
        Coefficient array for degree-n homogeneous terms.
        Returns appropriately-sized zero array if degree n is not present.
        
    Notes
    -----
    - Used internally by normalization routines to access specific degrees
    - Handles cases where polynomial doesn't contain the requested degree
    - JIT-compiled for efficient repeated access during normalization
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.base._make_poly` : Creates zero polynomial arrays
    """
    if n < len(poly_H):
        result = poly_H[n].copy()
    else:
        result = _make_poly(n, psi)
    return result


@njit(fastmath=FASTMATH, cache=False)
def _select_terms_for_elimination(
p_n: np.ndarray, 
n: int, 
clmo: np.ndarray) -> np.ndarray:
    r"""Identify non-resonant terms for elimination in Lie normalization.
    
    JIT-compiled function that selects polynomial terms to be eliminated
    based on resonance conditions. Terms are classified as resonant or
    non-resonant according to their multi-index structure.
    
    Parameters
    ----------
    p_n : ndarray
        Coefficient array for homogeneous polynomial terms of degree n.
    n : int
        Degree of the homogeneous terms being processed.
    clmo : ndarray
        Coefficient layout mapping objects for multi-index decoding.
        
    Returns
    -------
    ndarray
        Coefficient array containing only non-resonant terms to eliminate.
        Resonant terms are set to zero in the returned array.
        
    Notes
    -----
    **Resonance Condition**: A monomial with multi-index k = [k0, k1, k2, k3, k4, k5]
    is considered resonant if k[0] = k[3], where:
    
    - k[0], k[3] correspond to the hyperbolic (center) mode exponents
    - k[1], k[2], k[4], k[5] correspond to the elliptic mode exponents
    
    Non-resonant terms (k[0] != k[3]) are "bad" monomials that can be
    eliminated through canonical transformations without affecting the
    essential dynamics.
    
    The function creates an independent copy to avoid modifying the input
    and processes each coefficient individually for thread safety.
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.base._decode_multiindex` : Multi-index decoding
    :func:`~hiten.algorithms.hamiltonian.center._lie._lie_transform` : Uses this function for term selection
    """
    p_elim = p_n.copy()           # independent buffer
    for i in range(p_n.shape[0]):
        if p_elim[i] != 0.0:     # skip explicit zeros
            k = _decode_multiindex(i, n, clmo)
            if k[0] == k[3]:   # not a "bad" monomial -> zero it
                p_elim[i] = 0.0
    return p_elim


def _lie_expansion(
poly_G_total: List[np.ndarray], 
degree: int, psi: np.ndarray, 
clmo: np.ndarray, 
tol: float = 1e-30,
inverse: bool = False,
sign: int = None,
restrict: bool = True) -> List[List[np.ndarray]]:
    r"""Compute coordinate transformations using Lie series expansions.
    
    Performs Lie series transformations to compute polynomial expansions
    that relate center manifold coordinates to the original (or intermediate)
    coordinate system. Can operate in forward or inverse mode.
    
    Parameters
    ----------
    poly_G_total : List[ndarray]
        Complete set of generating functions for the Lie transformation,
        organized by polynomial degree.
    degree : int
        Maximum polynomial degree for the transformation series.
    psi : ndarray
        Combinatorial lookup table for polynomial indexing.
    clmo : ndarray
        Coefficient layout mapping objects for polynomial operations.
    tol : float, optional
        Tolerance for cleaning small coefficients. Default is 1e-30.
    inverse : bool, optional
        Transformation direction. If False, applies generators in ascending
        order (forward). If True, applies in descending order (inverse).
        Default is False.
    sign : int or None, optional
        Sign for generator application. If None, determined by inverse flag:
        +1 for forward, -1 for inverse. Default is None.
    restrict : bool, optional
        Whether to restrict results to center manifold by eliminating
        terms containing q1 or p1. Default is True.
        
    Returns
    -------
    List[List[ndarray]]
        Six polynomial expansions representing the coordinate transformation:
        [q1_expansion, q2_expansion, q3_expansion, p1_expansion, p2_expansion, p3_expansion]
        Each expansion is a list of coefficient arrays by degree.
        
    Notes
    -----
    **Lie Series Method**: The transformation uses the Lie series expansion:
    
    exp(L_G) * X = X + {X,G} + (1/2!){{X,G},G} + (1/3!){{X,G},G},G} + ...
    
    where L_G is the Lie derivative operator, {*,*} denotes Poisson brackets,
    and G is the generating function.
    
    **Coordinate Systems**:
    
    - **Forward Mode**: From normalized to original coordinates
    - **Inverse Mode**: From original to normalized coordinates
    - **Restriction**: Eliminates dependence on hyperbolic variables (q1, p1)
    
    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.center._lie._apply_coord_transform` : Applies transformation to single coordinate
    :func:`~hiten.algorithms.hamiltonian.center._lie._zero_q1p1` : Restricts expansions to center manifold
    :func:`~hiten.algorithms.hamiltonian.center._lie._evaluate_transform` : Evaluates expansions at specific points
    """
    # Create encode_dict_list from clmo
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    current_coords = []
    for i in range(6):
        poly = _polynomial_zero_list(degree, psi)
        poly[1] = np.zeros(6, dtype=np.complex128)
        poly[1][i] = 1.0 + 0j       # identity for q_1,q_2,q_3,p_1,p_2,p_3
        current_coords.append(poly) # [q1, q2, q3, p1, p2, p3]
    
    if inverse:
        start = degree
        stop = 2
        step = -1
        sign = -1 if sign is None else sign
    else:
        start = 3
        stop = degree + 1
        step = 1
        sign = 1 if sign is None else sign

    for n in range(start, stop, step):
        if n >= len(poly_G_total) or not poly_G_total[n].any():
            continue

        G_n = sign * poly_G_total[n]
        poly_G = _polynomial_zero_list(degree, psi)
        poly_G[n] = G_n.copy()
        
        new_coords = []
        for i in range(6):
            current_poly_typed = List()
            for arr in current_coords[i]:
                current_poly_typed.append(arr)

            new_poly = _apply_coord_transform(
                current_poly_typed, poly_G, degree, psi, clmo, encode_dict_list, tol
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
    r"""Apply Lie series transformation to single coordinate polynomial.
    
    JIT-compiled function that applies a Lie series transformation to transform
    a coordinate polynomial using a generating function. Implements the series
    expansion with automatic truncation and factorial coefficients.
    
    Parameters
    ----------
    poly_X : List[ndarray]
        Input coordinate polynomial to transform, organized by degree.
    poly_G : List[ndarray]
        Generating function polynomial for the transformation.
    N_max : int
        Maximum polynomial degree for the output series.
    psi : ndarray
        Combinatorial lookup table for polynomial indexing.
    clmo : ndarray
        Coefficient layout mapping objects.
    encode_dict_list : List[dict]
        Encoding dictionaries for polynomial operations.
    tol : float
        Tolerance for coefficient cleaning during computation.
        
    Returns
    -------
    List[ndarray]
        Transformed coordinate polynomial with same degree structure as input.
        
    Notes
    -----
    **Lie Series Formula**: Computes the transformation:
    
    exp(L_G) * X = X + {X,G} + (1/2!){{X,G},G} + (1/3!){{X,G},G},G} + ...
    
    where:
    
    - L_G is the Lie derivative operator generated by G
    - {*,*} denotes the Poisson bracket operation
    - Factorial coefficients ensure proper series convergence
    
    **Algorithm Steps**:
    
    1. Initialize result with input polynomial X
    2. Iteratively compute higher-order Poisson brackets
    3. Apply factorial coefficients (1/k!) for k-th order terms
    4. Accumulate contributions up to maximum degree
    5. Clean small coefficients at each step
    
    **Truncation**: Series is automatically truncated when terms would exceed
    N_max degree or when the generating function degree limits further terms.
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.operations._polynomial_poisson_bracket` : Poisson bracket computation
    :func:`~hiten.algorithms.polynomial.base._factorial` : Factorial computation
    """

    poly_result = List()
    for i in range(N_max + 1):
        if i < len(poly_X):
            poly_result.append(poly_X[i].copy())
        else:
            poly_result.append(_make_poly(i, psi))

    # Find degree of generating function
    deg_G = _polynomial_total_degree(poly_G, psi)

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
        poly_bracket = _polynomial_poisson_bracket(
            poly_bracket,
            poly_G,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )

        poly_bracket = _polynomial_clean(poly_bracket, tol)

        coeff = 1.0 / factorials[k]
        for d in range(min(len(poly_bracket), len(poly_result))):
            poly_result[d] += coeff * poly_bracket[d]

    return _polynomial_clean(poly_result, tol)


@njit(fastmath=FASTMATH, cache=False)
def _evaluate_transform(
expansions: List[List[np.ndarray]], 
coords_cm_complex: np.ndarray, 
clmo: np.ndarray) -> np.ndarray:
    r"""Evaluate coordinate transformation at specific center manifold point.
    
    JIT-compiled function that evaluates six polynomial expansions representing
    coordinate transformations at a given point in center manifold coordinates.
    Used to convert between coordinate systems.
    
    Parameters
    ----------
    expansions : List[List[ndarray]]
        Six polynomial expansions from Lie series transformation:
        [q1_expansion, q2_expansion, q3_expansion, p1_expansion, p2_expansion, p3_expansion]
        Each expansion is organized by polynomial degree.
    coords_cm_complex : ndarray, shape (6,)
        Center manifold coordinates [q1, q2, q3, p1, p2, p3] where evaluation
        is performed. Complex-valued for generality.
    clmo : ndarray
        Coefficient layout mapping objects for polynomial evaluation.
        
    Returns
    -------
    ndarray, shape (6,)
        Transformed coordinates [q1_tilde, q2_tilde, q3_tilde, p1_tilde, p2_tilde, p3_tilde]
        Complex-valued result from polynomial evaluation.
        
    Notes
    -----
    **Evaluation Process**: For each coordinate i = 0, 1, ..., 5:
    
    1. Takes the i-th polynomial expansion from the transformation
    2. Evaluates this polynomial at the given center manifold point
    3. Returns the complex result as the i-th transformed coordinate
    
    **Coordinate Mapping**:
    
    - Input: Center manifold coordinates (usually restricted to q2, p2, q3, p3)
    - Output: Transformed coordinates in original or intermediate system
    - Complex arithmetic preserves full generality of the transformation
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.operations._polynomial_evaluate` : Core polynomial evaluation
    :func:`~hiten.algorithms.hamiltonian.center._lie._lie_expansion` : Generates the expansions used here
    """

    result = np.zeros(6, dtype=np.complex128) # [q1, q2, q3, p1, p2, p3]
    
    for i in range(6):
        # Evaluate each polynomial at the given point
        result[i] = _polynomial_evaluate(expansions[i], coords_cm_complex, clmo)
    
    return result # [q1_tilde, q2_tilde, q3_tilde, p1_tilde, p2_tilde, p3_tilde]


def _zero_q1p1(
    expansions: List[List[np.ndarray]], 
    clmo: np.ndarray, 
    tol: float = 1e-30
) -> List[List[np.ndarray]]:
    r"""Restrict polynomial expansions to center manifold subspace.
    
    Eliminates all terms in coordinate expansions that depend on the hyperbolic
    variables q1 or p1, effectively restricting the expansions to the center
    manifold. This produces parameterizations that depend only on the stable
    center manifold coordinates.
    
    Parameters
    ----------
    expansions : List[List[ndarray]]
        Six coordinate expansions to restrict. Each expansion is organized
        by polynomial degree.
    clmo : ndarray
        Coefficient layout mapping objects for multi-index operations.
    tol : float, optional
        Tolerance for coefficient cleaning. Default is 1e-30.
        
    Returns
    -------
    List[List[ndarray]]
        Restricted coordinate expansions depending only on (q2, p2, q3, p3).
        Structure matches input but with hyperbolic terms eliminated.
        
    Notes
    -----
    **Center Manifold Restriction**: The center manifold in CR3BP applications
    typically corresponds to the stable/neutral directions associated with
    the elliptic modes, while q1, p1 correspond to the hyperbolic mode.
    
    **Term Elimination**: For each polynomial term with multi-index k:
    
    - If k[0] != 0 or k[3] != 0: Term depends on q1 or p1 -> eliminate
    - Otherwise: Term depends only on (q2, p2, q3, p3) -> keep
    
    **Result Properties**:
    
    - All six coordinate expansions become functions of 4 variables only
    - Expansions represent the center manifold embedding in full phase space
    - Suitable for reduced-order modeling and long-term dynamics analysis
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.base._decode_multiindex` : Multi-index decoding
    :func:`~hiten.algorithms.hamiltonian.center._lie._lie_expansion` : Often used with restrict=True to call this function
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
                k = _decode_multiindex(pos, deg, clmo)
                if k[0] != 0 or k[3] != 0:  # q1 or p1 exponent non-zero
                    coeff_vec[pos] = 0.0
        restricted_expansions.append(restricted_poly)
    
    return restricted_expansions