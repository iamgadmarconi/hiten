import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               _factorial, decode_multiindex,
                                               encode_multiindex, make_poly)
from algorithms.center.polynomial.operations import (
    polynomial_clean, polynomial_poisson_bracket, polynomial_zero_list, polynomial_evaluate)
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
        K = (N_max - deg_G) // (deg_G - 2) + 1
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


@njit(fastmath=FASTMATH, cache=False)
def _apply_coordinate_lie_transform(
    coords: np.ndarray,
    G_n: np.ndarray,
    deg_G: int,
    psi: np.ndarray,
    clmo: np.ndarray,
    encode_dict_list,
    max_degree: int,
    tol: float,
    forward: bool = True
) -> np.ndarray:
    """
    Apply Lie transform with -G_n to coordinates.
    
    This transforms coordinates according to:
    q_new = exp(L_{-G}) q_old = q_old - {q_old, G} - (1/2!){{q_old, G}, G} - ...
    """
    new_coords = coords.copy()
    
    # Process each coordinate that needs transformation
    for coord_idx in [1, 2, 4, 5]:  # q2, q3, p2, p3
        # Build polynomial representing the coordinate transformation
        # Start with: q_i = current_value + delta_q_i
        poly_coord = polynomial_zero_list(max_degree, psi)
        
        # Constant term: current coordinate value
        if len(poly_coord) > 0:
            poly_coord[0][0] = coords[coord_idx]
        
        # Linear term: identity for this coordinate
        if len(poly_coord) > 1:
            k = np.zeros(6, dtype=np.int64)
            k[coord_idx] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            poly_coord[1][pos] = 1.0
        
        poly_transformed = _apply_lie_transform(
            poly_coord,
            G_n if forward else -G_n,
            deg_G,
            max_degree,
            psi,
            clmo,
            encode_dict_list,
            tol
        )
        
        # Evaluate at origin (because we embedded current value as constant)
        origin = np.zeros(6, dtype=np.complex128)
        new_coords[coord_idx] = polynomial_evaluate(poly_transformed, origin, clmo)
    
    return new_coords


def inverse_lie_transform(
    cm_coords: np.ndarray,
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    tol: float = 1e-15
) -> np.ndarray:
    """
    Apply inverse Lie transformation to coordinates.
    
    This is the coordinate transformation analogue of lie_transform.
    It applies -G_n, -G_{n-1}, ..., -G_3 sequentially to transform
    from center manifold coordinates back to original coordinates.
    
    Parameters
    ----------
    cm_coords : numpy.ndarray
        Center manifold coordinates [q2, p2, q3, p3]
    poly_G_total : List[numpy.ndarray]
        List of generating functions from the forward normalization
    psi, clmo : numpy.ndarray
        Index tables
    max_degree : int
        Maximum degree of transformation
    tol : float
        Tolerance for cleaning
        
    Returns
    -------
    numpy.ndarray
        Original coordinates (6D complex vector)
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Initialize 6D coordinate vector from 4D CM coordinates
    coords = np.zeros(6, dtype=np.complex128)
    coords[1] = cm_coords[0]  # q2
    coords[2] = cm_coords[2]  # q3
    coords[4] = cm_coords[1]  # p2
    coords[5] = cm_coords[3]  # p3
    
    # Apply inverse transformations degree by degree (reverse order)
    for deg_G in range(max_degree, 2, -1):
        if deg_G >= len(poly_G_total) or not np.any(poly_G_total[deg_G]):
            continue
        
        # Get the generating function for this degree
        G_n = poly_G_total[deg_G]
        
        # Apply -G_n to current coordinates
        coords = _apply_coordinate_lie_transform(
            coords, G_n, deg_G, psi, clmo, encode_dict_list, max_degree, tol, forward=False
        )
    
    return coords

def forward_lie_transform(
    coords_phys: np.ndarray,
    poly_G_total: List[np.ndarray],
    psi: np.ndarray,
    clmo: np.ndarray,
    max_degree: int,
    tol: float = 1e-15
) -> np.ndarray:
    """
    Apply forward Lie transformation to coordinates.
    
    This is the coordinate transformation analogue of inverse_lie_transform.
    It applies G_n, G_{n-1}, ..., G_3 sequentially to transform
    from original coordinates to center manifold coordinates.
    
    Parameters
    ----------
    coords_phys : numpy.ndarray
        Original physical coordinates (6D complex vector) [q1, q2, q3, p1, p2, p3]
    poly_G_total : List[numpy.ndarray]
        List of generating functions from the forward normalization
    psi, clmo : numpy.ndarray
        Index tables
    max_degree : int
        Maximum degree of transformation
    tol : float
        Tolerance for cleaning
        
    Returns
    -------
    numpy.ndarray
        Center manifold coordinates (4D complex vector) [q2, p2, q3, p3]
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    # Initialize 6D coordinate vector from full 6D physical coordinates
    coords = coords_phys.copy()

    # +Gₙ applied in ascending order 3 … N
    for deg_G in range(3, max_degree + 1):
        if deg_G >= len(poly_G_total) or not np.any(poly_G_total[deg_G]):
            continue

        G_n = poly_G_total[deg_G]

        coords = _apply_coordinate_lie_transform(
            coords, G_n, deg_G, psi, clmo, encode_dict_list, max_degree, tol, forward=True)

    # Extract center manifold coordinates from 6D coordinate vector
    cm_coords = np.zeros(4, dtype=np.complex128)
    cm_coords[0] = coords[1]  # q2
    cm_coords[1] = coords[4]  # p2
    cm_coords[2] = coords[2]  # q3
    cm_coords[3] = coords[5]  # p3
    
    return cm_coords