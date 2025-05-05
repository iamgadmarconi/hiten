import math

from numba import njit
from numba.typed import List
import numpy as np

from algorithms.variables import N_VARS
from algorithms.polynomial.base import decode_multiindex, make_poly
from algorithms.polynomial.algebra import poisson


def lie_transform(
    point,
    H_init_coeffs: list[np.ndarray],
    psi: np.ndarray,
    clmo,
    max_degree: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Bring H_init_coeffs to partial normal form up to max_degree.

    point: LibrationPoint with method linear_modes() -> (lambda1, omega1, omega2)
    H_init_coeffs: list of degree-indexed coeff arrays
    psi, clmo: index tables
    max_degree: highest degree to normalize
    Returns: (H_normalized_coeffs, G_total_coeffs)
    """
    # extract linear eigenvalues
    lam, om1, om2 = point.linear_modes()
    eta = np.array([lam, 1j*om1, 1j*om2], dtype=np.complex128)

    # initialize
    H_trans = [h.copy() for h in H_init_coeffs]
    G_total = [make_poly(d, psi, complex_dtype=True)
               for d in range(max_degree+1)]

    # quadratic part
    # H2 = H_trans[2]  # not directly needed here

    for n in range(3, max_degree+1):
        Hn = H_trans[n]
        if not Hn.any(): continue
        ToKill = _select_terms_for_elimination(Hn, n, psi, clmo)
        if not ToKill.any(): continue
        Gn = _solve_homological_equation(ToKill, n, eta, psi, clmo)
        H_trans = _apply_lie_transform(H_trans, Gn, n, max_degree, psi, clmo)
        # accumulate G_total
        G_total[n] += Gn
    return H_trans, G_total


def _get_homogeneous_terms(H_coeffs: List[np.ndarray], n: int,
                          psi: np.ndarray, complex_dtype: bool=False) -> np.ndarray:
    """
    Return the degree-n homogeneous component of H.

    H_coeffs : list of np.ndarray
        H_coeffs[d] is the array of coefficients for degree d.
    n : int
        Desired homogeneous degree.
    psi : np.ndarray
        The psi table from init_index_tables (for zero-padding when needed).
    complex_dtype : bool
        If True, create a complex-typed zero array when n exceeds len(H_coeffs).

    Returns
    -------
    np.ndarray
        The coefficient array of length psi[6,n], either the stored H_coeffs[n],
        or zeros if n is beyond the current maximum degree.
    """
    if n < len(H_coeffs):
        # We already have exactly the degree-n component
        return H_coeffs[n]
    else:
        # No degree-n terms were ever set → return a zero array
        if complex_dtype:
            return np.zeros(psi[N_VARS, n], dtype=np.complex128)
        else:
            return np.zeros(psi[N_VARS, n], dtype=np.float64)


@njit(fastmath=True, cache=True)
def _select_terms_for_elimination(
    Hn: np.ndarray,   # coeffs of H_n, length = psi[6,n]
    n: int,           # the degree n
    psi: np.ndarray,  # the psi table
    clmo             # the clmo list
) -> np.ndarray:
    """
    Return an array E of the same shape as Hn, with
    E[i] = Hn[i]  if exponent(q1)!=exponent(p1) in monomial i,
         = 0       otherwise.
    """
    E = np.zeros_like(Hn)
    for i in range(Hn.shape[0]):
        ci = Hn[i]
        if ci != 0.0:
            k = decode_multiindex(i, n, clmo)
            # k[0] is exponent of q1, k[3] is exponent of p1
            if k[0] != k[3]:
                E[i] = ci
    return E


@njit(fastmath=True, cache=True)
def _solve_homological_equation(
    Hn_elim: np.ndarray,   # degree-n coefficients to eliminate
    n: int,                # the degree n
    eta: np.ndarray,       # length-3 array of eta_1, eta_2, eta_3 (complex128)
    psi: np.ndarray,       # psi table from init_index_tables
    clmo                  # clmo list from init_index_tables
) -> np.ndarray:
    """
    Solve {H2, G_n} = -Hn_elim for G_n in complex normal form:
        G_n[i] = - Hn_elim[i] / dot(kp-kq, eta)
    """
    # allocate G_n with same shape & dtype
    G = np.zeros_like(Hn_elim)
    for i in range(Hn_elim.shape[0]):
        ci = Hn_elim[i]
        if ci != 0:
            # decode full 6-vector of exponents
            k = decode_multiindex(i, n, clmo)
            # split into q-exponents (k[0..2]) and p-exponents (k[3..5])
            # compute kp − kq in length-3
            denom = (k[3] - k[0]) * eta[0] \
                  + (k[4] - k[1]) * eta[1] \
                  + (k[5] - k[2]) * eta[2]
            # small-divisor check (paper guarantees denom ≠ 0 for selected terms)
            # but we guard against numerical zero
            if denom == 0:
                # you could raise, or skip, or set G[i]=0
                # here we simply leave G[i]=0
                continue
            G[i] = -ci / denom
    return G


def _apply_lie_transform(
    H_coeffs: list[np.ndarray],  # H_coeffs[d] is coeff‐array for degree d
    G_n:     np.ndarray,         # coeff‐array for generating function (degree=deg_G)
    deg_G:   int,                # total degree of G_n
    N_max:   int,                # truncate final H at this degree
    psi:     np.ndarray,
    clmo,
) -> list[np.ndarray]:
    """
    Apply H -> exp(L_{G_n}) H up to order N_max, i.e.
    H_new = sum_{k=0}^K  (1/k!) ad_{G_n}^k (H),
    where K = max(1, deg_G-1), and ad = Poisson bracket.
    """
    # 1) make a *deep* copy of H_coeffs to accumulate into
    H_new = [ h.copy() for h in H_coeffs ]

    # 2) prepare factorials 1!,2!,…,(deg_G-1)! in Python
    K = max(1, deg_G - 1)
    factorials = [math.factorial(k) for k in range(K+1)]  # factorials[0]=1 for consistency

    # 3) initialize PB_term_list = H_coeffs  (so ad^0 H = H itself)
    PB_term_list = H_coeffs

    # 4) iteratively compute ad_{G_n}^k H
    for k in range(1, K+1):
        # build next Poisson-bracket term: PB_term_list <- { PB_term_list, G_n }
        PB_next = [ np.zeros(psi[N_VARS, d], dtype=G_n.dtype) 
                    for d in range(N_max+1) ]

        # loop over all degrees in PB_term_list
        for deg_H, H_d in enumerate(PB_term_list):
            if not H_d.any(): 
                continue
            deg_R = deg_H + deg_G - 2
            if 0 <= deg_R <= N_max:
                # compute the bracket of the two homogeneous arrays
                R_d = poisson(H_d, deg_H, G_n, deg_G, psi, clmo)
                # accumulate into PB_next[deg_R]
                PB_next[deg_R] += R_d

        PB_term_list = PB_next

        # scale by 1/k! and add into H_new
        inv_fact = 1.0 / factorials[k]
        for d in range(N_max+1):
            H_new[d] += PB_term_list[d] * inv_fact

    # 5) finally, truncate (zero‐out any degrees > N_max) and return
    return H_new
