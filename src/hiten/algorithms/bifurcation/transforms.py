"""Polynomial transformations for Birkhoff normal form analysis.

This module provides transformations between different polynomial representations
used in bifurcation analysis, specifically converting Birkhoff normal form
polynomials to action-angle form for libration points in the CR3BP.
"""

import numpy as np

from hiten.algorithms.fourier.base import (_create_encode_dict_fourier,
                                           _encode_fourier_index,
                                           _init_fourier_tables)
from hiten.algorithms.polynomial.algebra import _get_degree
from hiten.algorithms.polynomial.base import (_CLMO_GLOBAL, _PSI_GLOBAL,
                                              _decode_multiindex)


def _nf2aa_ee(poly_nf_complex: np.ndarray) -> np.ndarray:
    """Convert Birkhoff normal form polynomial to action-angle form.

    Transforms canonical (q,p) polynomial representation to action-angle
    (I,theta) representation for elliptic-elliptic libration points (L4, L5).
    Only monomials with even total canonical degree contribute to integer
    action exponents.

    Parameters
    ----------
    poly_nf_complex : np.ndarray
        Coefficient array of Birkhoff normal form complex polynomial.
        Shape corresponds to canonical polynomial degree structure.

    Returns
    -------
    np.ndarray
        Action-angle polynomial coefficients. Returns zero array for
        odd-degree input polynomials (no valid action-angle monomials).
        
    Raises
    ------
    ValueError
        If polynomial degree cannot be inferred from coefficient array size.
        
    Notes
    -----
    The transformation uses the canonical-to-action-angle mapping:
    - Action variables: I_j = (q_j^2 + p_j^2)/2
    - Angle variables: theta_j = arctan(p_j/q_j)
    - Fourier indices: k_j = q_j_exp - p_j_exp
    - Prefactor: (-i)^(sum of p exponents)
    
    Only processes monomials where each (q_j + p_j) sum is even,
    ensuring integer action exponents.
    """
    deg_nf: int = _get_degree(poly_nf_complex, _PSI_GLOBAL)
    if deg_nf < 0:
        raise ValueError("Unable to infer polynomial degree from coefficient array size.")

    #  A necessary condition for a monomial to map to integer action exponents
    #  is that the *total* canonical degree is **even**.  For odd degrees we
    #  return an appropriately-shaped zero array to keep the calling code
    #  simple (rather than raising).
    if deg_nf % 2:
        #  Pick degree floor(d/2) for shape (convention) - result is identically zero.
        deg_aa = deg_nf // 2
        psiF, _ = _init_fourier_tables(deg_aa, 0)
        return np.zeros(psiF[deg_aa], dtype=np.complex128)

    #  Target action degree
    deg_aa: int = deg_nf // 2

    k_max_required = 0  # maximum |k_j| encountered (same for all j)

    for pos in range(poly_nf_complex.shape[0]):
        c = poly_nf_complex[pos]
        if c == 0.0:
            continue

        # Decode (q,p) exponents: k = (q1,q2,q3,p1,p2,p3)
        k_vec = _decode_multiindex(pos, deg_nf, _CLMO_GLOBAL)

        # Quickly reject monomials that lead to *non-integer* action exponents
        valid = True
        for j in range(3):
            a = k_vec[j]       # q_j exponent
            b = k_vec[j + 3]   # p_j exponent
            if (a + b) & 1:    # parity check (odd total)
                valid = False
                break
        if not valid:
            continue

        # Fourier indices k_j = a - b
        for j in range(3):
            k_val = k_vec[j] - k_vec[j + 3]
            if abs(k_val) > k_max_required:
                k_max_required = abs(k_val)

    # Cap to the implementation hard limit (<= 63) - the helper truncates anyway
    k_max_required = min(k_max_required, 63)

    psiF, clmoF = _init_fourier_tables(deg_aa, k_max_required)
    encodeF = _create_encode_dict_fourier(clmoF)

    # Allocate output block
    coeffs_aa = np.zeros(psiF[deg_aa], dtype=np.complex128)

    for pos in range(poly_nf_complex.shape[0]):
        c = poly_nf_complex[pos]
        if c == 0.0:
            continue

        k_vec = _decode_multiindex(pos, deg_nf, _CLMO_GLOBAL)

        # Build n_j and k_j  (action degree & Fourier index) for j = 1..3
        n = [0, 0, 0]
        k = [0, 0, 0]
        p_tot = 0  # Total p-exponent (for the (-i)^sum(b) prefactor)

        valid = True
        for j in range(3):
            a = k_vec[j]       # exponent of q_j
            b = k_vec[j + 3]   # exponent of p_j
            s = a + b
            if s & 1:          # odd => half-integer action power - ignore
                valid = False
                break
            n[j] = s >> 1      # integer division by 2
            k[j] = a - b
            p_tot += b
        if not valid:
            continue

        # Target degree consistency check
        if n[0] + n[1] + n[2] != deg_aa:
            # This should not occur if deg_aa = deg_nf//2, but guard anyway
            continue

        # Prefactor (-i)^{sum(b_j)}
        pref = ((-1j) ** p_tot) * c

        # Encode (n1,n2,n3,k1,k2,k3) into coefficient array position
        idx_tuple = (n[0], n[1], n[2], k[0], k[1], k[2])
        pos_aa = _encode_fourier_index(idx_tuple, deg_aa, encodeF)
        if pos_aa != -1:
            coeffs_aa[pos_aa] += pref

    return coeffs_aa

def _nf2aa_sc(poly_nf_complex: np.ndarray) -> np.ndarray:
    """Convert Birkhoff normal form polynomial to action-angle form.

    Transforms canonical (q,p) polynomial representation to action-angle
    (I,theta) representation for saddle-center libration points (L1, L2, L3).
    
    The center directions follow identical mapping rules as elliptic-elliptic
    systems. The hyperbolic direction does not introduce additional angle
    dependence, so the transformation logic remains unchanged.

    Parameters
    ----------
    poly_nf_complex : np.ndarray
        Coefficient array of Birkhoff normal form complex polynomial.

    Returns
    -------
    np.ndarray
        Action-angle polynomial coefficients.
        
    Notes
    -----
    Currently delegates to :func:`~hiten.algorithms.bifurcation.transforms._nf2aa_ee` since the transformation
    rules for center directions are identical. Future extensions could
    implement saddle-center specific processing (e.g., filtering
    hyperbolic Fourier harmonics) without affecting the API.
    """
    # Implementation note: the mapping rules that convert canonical
    # monomials (q, p) -> (I, theta) are identical for the centre directions
    # of a saddle-centre system. The hyperbolic pair does not introduce
    # any additional angle dependence, but the algebraic prefactors and
    # exponent bookkeeping remain unchanged. Consequently, the existing
    # elliptic-elliptic helper covers the required functionality. We
    # therefore delegate to it directly - this keeps the public API
    # intact and avoids code duplication. If a future extension needs
    # to treat the hyperbolic degree differently (for example, to drop
    # Fourier harmonics with k_1 != 0), it can be implemented here without
    # affecting callers.
    return _nf2aa_ee(poly_nf_complex)