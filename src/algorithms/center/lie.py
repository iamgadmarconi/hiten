import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               _factorial, decode_multiindex,
                                               make_poly)
from algorithms.center.polynomial.operations import (
    polynomial_clean, polynomial_poisson_bracket, polynomial_zero_list)
from log_config import logger


def lie_transform(point, H_init_coeffs: list[np.ndarray], psi: np.ndarray, clmo: np.ndarray, max_degree: int, tol: float = 1e-15) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lam, om1, om2 = point.linear_modes()
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)

    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    H_trans = [h.copy() for h in H_init_coeffs]
    G_total = polynomial_zero_list(max_degree, psi)

    for n in range(3, max_degree+1):
        logger.info(f"Normalizing at order: {n}")
        Hn = H_trans[n]
        if not Hn.any():
            continue
        to_eliminate = _select_terms_for_elimination(Hn, n, clmo)
        if not to_eliminate.any():
            continue
        Gn = _solve_homological_equation(to_eliminate, n, eta, clmo)
        
        # Clean Gn using a Numba typed list for compatibility with polynomial_clean
        if Gn.any(): # Only clean if there's something to clean
            temp_Gn_list = List()
            temp_Gn_list.append(Gn)
            cleaned_Gn_list = polynomial_clean(temp_Gn_list, tol)
            Gn = cleaned_Gn_list[0]

        # Pass the cleaned Gn to _apply_lie_transform
        # Convert H_trans to Numba typed list for _apply_lie_transform
        h_trans_typed = List()
        for item_arr in H_trans:
            h_trans_typed.append(item_arr)
        # _apply_lie_transform expects a Numba List for H_coeffs_py and returns a Python list
        H_trans = _apply_lie_transform(h_trans_typed, Gn, n, max_degree, psi, clmo, encode_dict_list, tol)
        
        if n < len(G_total) and G_total[n].shape == Gn.shape:
             G_total[n] += Gn
        elif n < len(G_total) and G_total[n].size == Gn.size:
             G_total[n] += Gn.reshape(G_total[n].shape)

        if not _select_terms_for_elimination(H_trans[n], n, clmo).any():
            continue
            
    G_total = polynomial_clean(G_total, tol)
    return H_trans, G_total


@njit(fastmath=True, cache=True)
def _get_homogeneous_terms(H_coeffs: List[np.ndarray], n: int, psi: np.ndarray) -> np.ndarray:
    """
    Return the degree-n homogeneous component of H.

    H_coeffs : list of np.ndarray
        H_coeffs[d] is the array of coefficients for degree d.
    n : int
        Desired homogeneous degree.
    psi : np.ndarray
        The psi table from init_index_tables (for zero-padding when needed).

    Returns
    -------
    np.ndarray
        The coefficient array of length psi[6,n], either the stored H_coeffs[n],
        or zeros if n is beyond the current maximum degree.
    """
    if n < len(H_coeffs):
        result = H_coeffs[n].copy()
    else:
        result = make_poly(n, psi)
    return result


@njit(fastmath=True, cache=True)
def _select_terms_for_elimination(Hn: np.ndarray, n: int, clmo: np.ndarray) -> np.ndarray:
    E = Hn.copy()           # independent buffer
    for i in range(Hn.shape[0]):
        if E[i] != 0.0:     # skip explicit zeros
            k = decode_multiindex(i, n, clmo)
            if k[0] == k[3]:   # not a "bad" monomial -> zero it
                E[i] = 0.0
    return E


@njit(fastmath=True, cache=True)
def _solve_homological_equation(Hn_bad: np.ndarray, n: int, eta: np.ndarray, clmo: np.ndarray) -> np.ndarray:
    G = np.zeros_like(Hn_bad)
    for i in range(Hn_bad.shape[0]):
        c = Hn_bad[i]
        if c != 0.0:
            k = decode_multiindex(i, n, clmo)
            denom = ((k[3]-k[0]) * eta[0] +
                     (k[4]-k[1]) * eta[1] +
                     (k[5]-k[2]) * eta[2])
            G[i] = -c / denom
    return G


@njit(fastmath=True, cache=True)
def _apply_lie_transform(H_coeffs_py: List[np.ndarray], G_n: np.ndarray, deg_G: int, N_max: int, psi: np.ndarray, clmo, encode_dict_list, tol: float) -> list[np.ndarray]:

    H_new_py = polynomial_zero_list(N_max, psi) # Use helper for clarity
    for i in range(min(len(H_coeffs_py), N_max + 1)):
        if i < len(H_coeffs_py) and H_coeffs_py[i].shape == H_new_py[i].shape:
             H_new_py[i] = H_coeffs_py[i].copy()
        elif i < len(H_coeffs_py) and H_coeffs_py[i].size == H_new_py[i].size: # check for size if shape is different but compatible
             H_new_py[i] = H_coeffs_py[i].copy().reshape(H_new_py[i].shape)


    if deg_G > 2:
        K = (N_max - deg_G) // (deg_G - 2) + 1
    else:  # quadratic generator –very rare here–
        K = 1
    factorials = [_factorial(k) for k in range(K + 1)]

    PB_term_list_typed = List()
    for d in range(N_max + 1):
        if d < len(H_coeffs_py):
            PB_term_list_typed.append(H_coeffs_py[d].copy())
        else:
            PB_term_list_typed.append(make_poly(d, psi))

    G_n_as_list = polynomial_zero_list(N_max, psi) # Ensure G_n_as_list can go up to N_max if deg_G is high
    if deg_G <= N_max and deg_G < len(G_n_as_list): # Check if deg_G is a valid index for G_n_as_list
        if G_n_as_list[deg_G].shape == G_n.shape:
            G_n_as_list[deg_G] = G_n.copy()
        elif G_n_as_list[deg_G].size == G_n.size : # check for size if shape is different but compatible
            G_n_as_list[deg_G] = G_n.copy().reshape(G_n_as_list[deg_G].shape)


    for k in range(1, K + 1):
        PB_term_list_typed = polynomial_poisson_bracket(
            PB_term_list_typed,
            G_n_as_list,
            N_max,
            psi,
            clmo,
            encode_dict_list
        )
        PB_term_list_typed = polynomial_clean(PB_term_list_typed, tol)

        inv_fact = 1.0 / factorials[k]
        for d in range(N_max + 1):
            if d < len(PB_term_list_typed) and d < len(H_new_py) and \
               H_new_py[d].shape == PB_term_list_typed[d].shape:
                 H_new_py[d] += PB_term_list_typed[d] * inv_fact
            elif d < len(PB_term_list_typed) and d < len(H_new_py) and \
                 H_new_py[d].size == PB_term_list_typed[d].size: # check for size if shape is different but compatible
                 H_new_py[d] += PB_term_list_typed[d].reshape(H_new_py[d].shape) * inv_fact

    H_new_py = polynomial_clean(H_new_py, tol)
    return H_new_py
