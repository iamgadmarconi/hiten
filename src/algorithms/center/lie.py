import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.algebra import _evaluate_reduced_monomial
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               _factorial, decode_multiindex,
                                               make_poly)
from algorithms.center.polynomial.operations import (
    polynomial_clean, polynomial_poisson_bracket, polynomial_zero_list)
from config import FASTMATH
from utils.log_config import logger


def lie_transform(point, poly_init: list[np.ndarray], psi: np.ndarray, clmo: np.ndarray, max_degree: int, tol: float = 1e-15) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Perform a Lie transformation to normalize a Hamiltonian.
    
    Parameters
    ----------
    point : object
        Object containing information about the linearized dynamics
        (eigenvalues and frequencies)
    poly_init : list[numpy.ndarray]
        Initial polynomial Hamiltonian to normalize
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    max_degree : int
        Maximum degree to include in the normalized Hamiltonian
    tol : float, optional
        Tolerance for cleaning small coefficients, default is 1e-15
        
    Returns
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        A tuple containing:
        - The normalized Hamiltonian
        - The generating function for the normalization
        
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
    lam, om1, om2 = point.linear_modes()
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_trans = [h.copy() for h in poly_init]
    poly_G_total = polynomial_zero_list(max_degree, psi)

    for n in range(3, max_degree+1):
        logger.info(f"Normalizing at order: {n}")
        p_n = poly_trans[n]
        if not p_n.any():
            continue
        p_elim = _select_terms_for_elimination(p_n, n, clmo)
        if not p_elim.any():
            continue
        p_G_n = _solve_homological_equation(p_elim, n, eta, clmo)
        
        # Clean Gn using a Numba typed list for compatibility with polynomial_clean
        if p_G_n.any(): # Only clean if there's something to clean
            temp_G_n_list = List()
            temp_G_n_list.append(p_G_n)
            cleaned_G_n_list = polynomial_clean(temp_G_n_list, tol)
            p_G_n = cleaned_G_n_list[0]

        # Pass the cleaned Gn to _apply_lie_transform
        # Convert poly_trans to Numba typed list for _apply_lie_transform
        poly_trans_typed = List()
        for item_arr in poly_trans:
            poly_trans_typed.append(item_arr)
        # _apply_lie_transform expects a Numba List for poly_H and returns a Python list
        poly_trans = _apply_lie_transform(poly_trans_typed, p_G_n, n, max_degree, psi, clmo, encode_dict_list, tol)
        
        if n < len(poly_G_total) and poly_G_total[n].shape == p_G_n.shape:
            poly_G_total[n] += p_G_n
        elif n < len(poly_G_total) and poly_G_total[n].size == p_G_n.size:
            poly_G_total[n] += p_G_n.reshape(poly_G_total[n].shape)

        if not _select_terms_for_elimination(poly_trans[n], n, clmo).any():
            continue
            
    poly_G_total = polynomial_clean(poly_G_total, tol)
    return poly_trans, poly_G_total


@njit(fastmath=FASTMATH, cache=True)
def _get_homogeneous_terms(poly_H: List[np.ndarray], n: int, psi: np.ndarray) -> np.ndarray:
    """
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


@njit(fastmath=FASTMATH, cache=True)
def _select_terms_for_elimination(p_n: np.ndarray, n: int, clmo: np.ndarray) -> np.ndarray:
    """
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


@njit(fastmath=FASTMATH, cache=True)
def _solve_homological_equation(p_elim: np.ndarray, n: int, eta: np.ndarray, clmo: np.ndarray) -> np.ndarray:
    """
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
    """
    p_G = np.zeros_like(p_elim)
    for i in range(p_elim.shape[0]):
        c = p_elim[i]
        if c != 0.0:
            k = decode_multiindex(i, n, clmo)
            denom = ((k[3]-k[0]) * eta[0] +
                     (k[4]-k[1]) * eta[1] +
                     (k[5]-k[2]) * eta[2])
            p_G[i] = -c / denom
    return p_G


@njit(fastmath=FASTMATH, cache=False)
def _apply_lie_transform(poly_H: List[np.ndarray], p_G_n: np.ndarray, deg_G: int, N_max: int, psi: np.ndarray, clmo, encode_dict_list, tol: float) -> list[np.ndarray]:
    """
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
    poly_new = polynomial_zero_list(N_max, psi) # Use helper for clarity
    for i in range(min(len(poly_H), N_max + 1)):
        if i < len(poly_H) and poly_H[i].shape == poly_new[i].shape:
            poly_new[i] = poly_H[i].copy()
        elif i < len(poly_H) and poly_H[i].size == poly_new[i].size: # check for size if shape is different but compatible
            poly_new[i] = poly_H[i].copy().reshape(poly_new[i].shape)


    if deg_G > 2:
        K = (N_max - deg_G) // (deg_G - 2) + 1
    else:  # quadratic generator –very rare here–
        K = 1
    factorials = [_factorial(k) for k in range(K + 1)]

    poly_PB_term = List()
    for d in range(N_max + 1):
        if d < len(poly_H):
            poly_PB_term.append(poly_H[d].copy())
        else:
            poly_PB_term.append(make_poly(d, psi))

    poly_G = polynomial_zero_list(N_max, psi) # Ensure poly_G can go up to N_max if deg_G is high
    if deg_G <= N_max and deg_G < len(poly_G): # Check if deg_G is a valid index for poly_G
        if poly_G[deg_G].shape == p_G_n.shape:
            poly_G[deg_G] = p_G_n.copy()
        elif poly_G[deg_G].size == p_G_n.size : # check for size if shape is different but compatible
            poly_G[deg_G] = p_G_n.copy().reshape(poly_G[deg_G].shape)


    for k in range(1, K + 1):
        poly_PB_term = polynomial_poisson_bracket(
            poly_PB_term,
            poly_G,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )
        poly_PB_term = polynomial_clean(poly_PB_term, tol)

        inv_fact = 1.0 / factorials[k]
        for d in range(N_max + 1):
            if d < len(poly_PB_term) and d < len(poly_new) and \
                poly_new[d].shape == poly_PB_term[d].shape:
                poly_new[d] += poly_PB_term[d] * inv_fact
            elif d < len(poly_PB_term) and d < len(poly_new) and \
                poly_new[d].size == poly_PB_term[d].size: # check for size if shape is different but compatible
                poly_new[d] += poly_PB_term[d].reshape(poly_new[d].shape) * inv_fact

    poly_new = polynomial_clean(poly_new, tol)
    return poly_new


@njit(fastmath=FASTMATH, cache=True)
def _compute_pb_coord(
    G_coeffs: np.ndarray,
    degree: int,
    coord_idx: int,
    coords: np.ndarray,
    clmo: np.ndarray,
    encode_dict_list: List
) -> np.complex128:
    """
    Compute {G, x_coord_idx} evaluated at the given coordinates.
    
    For a monomial G_term = coeff * q^a * p^b:
    {G_term, q_i} = -coeff * b_i * q^a * p^(b with p_i reduced by 1)  
    {G_term, p_i} = +coeff * a_i * q^(a with q_i reduced by 1) * p^b
    """

    result = 0.0 + 0.0j
    
    for pos in range(G_coeffs.shape[0]):
        coeff = G_coeffs[pos]
        if abs(coeff) < 1e-15:
            continue
            
        # Decode the multi-index for this monomial
        k = decode_multiindex(pos, degree, clmo)
        
        if coord_idx < 3:  # Position coordinate q_i
            # {G_term, q_i} = -k[i+3] * G_term with p_i exponent reduced by 1
            p_idx = coord_idx + 3
            if k[p_idx] > 0:
                # Evaluate monomial with p_i exponent reduced by 1
                monomial_val = _evaluate_reduced_monomial(k, coords, p_idx, -1)
                result -= coeff * k[p_idx] * monomial_val
        else:  # Momentum coordinate p_i  
            # {G_term, p_i} = +k[i] * G_term with q_i exponent reduced by 1
            q_idx = coord_idx - 3
            if k[q_idx] > 0:
                # Evaluate monomial with q_i exponent reduced by 1
                monomial_val = _evaluate_reduced_monomial(k, coords, q_idx, -1)
                result += coeff * k[q_idx] * monomial_val
                
    return result


@njit(fastmath=FASTMATH, cache=True)
def _apply_single_inverse_generator(
    coords: np.ndarray,
    G_n: np.ndarray,
    degree: int,
    psi: np.ndarray,
    clmo: np.ndarray,
    encode_dict_list: List
) -> np.ndarray:
    """
    Apply inverse of a single generating function using first-order approximation.
    
    For the inverse transformation, we use:
    x_new = x_old - {G, x_old} + O(G^2)
    
    This is the first-order approximation of exp(-L_G).
    """
    new_coords = coords.copy()
    
    # Compute Poisson brackets {G, x_i} for each coordinate
    for i in range(6):
        pb_term = _compute_pb_coord(
            G_n, degree, i, coords, clmo, encode_dict_list
        )
        new_coords[i] -= pb_term  # Negative sign for inverse transform
    
    return new_coords


@njit(fastmath=FASTMATH, cache=True)
def _apply_inverse_lie_transforms(
    cm_coords: np.ndarray,
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    tol: float
) -> np.ndarray:
    """Apply inverse Lie transforms to go from center manifold to complex normal coordinates."""
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Start with 6D coordinates: [q1=0, q2, q3, p1=0, p2, p3]
    coords = np.zeros(6, dtype=np.complex128)
    coords[1] = cm_coords[0]  # q2
    coords[2] = cm_coords[2]  # q3
    coords[4] = cm_coords[1]  # p2  
    coords[5] = cm_coords[3]  # p3
    # q1=0, p1=0 remain zero (center manifold constraint)

    for degree in range(max_degree, 2, -1):
        if degree < len(poly_G_total) and np.any(poly_G_total[degree]):
            coords = _apply_single_inverse_generator(
                coords, poly_G_total[degree], degree, psi, clmo, encode_dict_list
            )
    
    return coords