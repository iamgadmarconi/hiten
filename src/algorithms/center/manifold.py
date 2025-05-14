import numpy as np
from numba.typed import List

from algorithms.center.lie import lie_transform
from algorithms.center.hamiltonian import rn2cn, phys2rn, build_physical_hamiltonian
from algorithms.center.polynomial.base import init_index_tables


def compute_center_manifold_arrays(
    point,
    max_degree: int,
    *,
    psi=None,
    clmo=None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Return (H_cnt, G_tot) - the Jorba-Masdemont normalised Hamiltonian
    and the accumulated generating function, both in complex canonical
    variables (q1,q2,q3,p1,p2,p3) and dense-array form.
    """
    # tables
    if psi is None or clmo is None:
        psi, clmo = init_index_tables(max_degree)

    # build physical Hamiltonian
    H_phys = build_physical_hamiltonian(point, max_degree, psi, clmo)

    # linear changes of coordinates
    H_rn   = phys2rn(point, H_phys, max_degree, psi, clmo)
    H_cn   = rn2cn(H_rn, max_degree, psi, clmo)

    # Lie-series partial normal form
    H_cnt, G_tot = lie_transform(point, H_cn, psi, clmo, max_degree)

    return H_cnt, G_tot


def reduce_center_manifold_arrays(
    point,
    max_degree: int,
    *,
    psi=None,
    clmo=None
) -> List[np.ndarray]:
    """
    Return the Hamiltonian on the 4-dimensional centre manifold (q1 = p1 = 0),
    still expressed in complex canonical variables, as dense-array Poly.
    """
    if psi is None or clmo is None:
        psi, clmo = init_index_tables(max_degree)

    H_cnt, _ = compute_center_manifold_arrays(point, max_degree,
                                            psi=psi, clmo=clmo)

    # zero-out hyperbolic pair
    _zero_q1_p1(H_cnt, psi, clmo)

    return H_cnt


def _zero_q1_p1(H: List[np.ndarray], psi, clmo) -> None:
    """
    In-place: set to zero every monomial that contains q1 (var #0) or p1 (var #3).
    """
    for deg, coeff_vec in enumerate(H):
        if not coeff_vec.any():
            continue
        for idx, c in enumerate(coeff_vec):
            if c == 0.0:
                continue
            k = decode_multiindex(idx, deg, clmo)
            if k[0] > 0 or k[3] > 0:          # q1 or p1 exponent ≠ 0
                coeff_vec[idx] = 0.0