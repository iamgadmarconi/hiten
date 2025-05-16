import numpy as np
from numba.typed import List
from typing import Tuple

from algorithms.center.lie import lie_transform
from algorithms.center.hamiltonian import rn2cn, phys2rn, cn2rn, build_physical_hamiltonian
from algorithms.center.polynomial.base import init_index_tables, decode_multiindex


def coefficients_to_table_arrays(
    H_cnr: List[np.ndarray],                       # complex canonical, q1=p1=0
    psi,
    clmo,
    *,
    save: bool = False,
    filename: str = "cm_coeffs.txt",
    tol: float = 1e-14
) -> str:
    """
    Format the coefficients that actually live on the 4-dof centre manifold
    (q2, p2, q3, p3) as a nice printable table.

    Returns the table string and optionally writes it to *filename*.
    """
    # mapping from variable index to output column
    # indices in canonical order: q1(0) q2(1) q3(2) p1(3) p2(4) p3(5)
    idx_map = {1: 0, 4: 1, 2: 2, 5: 3}          # (q2,p2,q3,p3) → (k1..k4)

    rows: list[Tuple[int,int,int,int,complex]] = []

    for deg, coeff_vec in enumerate(H_cnr):
        if not coeff_vec.any():
            continue
        for pos, c in enumerate(coeff_vec):
            if abs(c) < tol:
                continue
            k = decode_multiindex(pos, deg, clmo)
            if k[0] != 0 or k[3] != 0:          # q1 or p1 present – should be zero already
                continue
            # extract exponents of q2,p2,q3,p3
            k_out = [0, 0, 0, 0]
            for var_idx, col in idx_map.items():
                k_out[col] = k[var_idx]
            rows.append((*k_out, c))

    # sort by total degree then lexicographically
    rows.sort(key=lambda r: (sum(r[:4]), *r[:4]))

    # build ascii table
    header = (
        "Coefficients of the centre-manifold Hamiltonian  H_cn(q2,p2,q3,p3)\n"
        "k1…k4 correspond to (q2, p2, q3, p3)\n"
        + "="*100 + "\n"
        + "{:<4} {:<4} {:<4} {:<4} {:>25}\n".format("k1","k2","k3","k4","coeff")
        + "-"*100 + "\n"
    )
    lines = [
        "{:<4} {:<4} {:<4} {:<4} {:>25.16e}".format(*k1234, c)
        for *k1234, c in rows
    ]
    table_str = header + "\n".join(lines)

    if save:
        with open(filename, "w", encoding="utf8") as f:
            f.write(table_str)
        logger.info("Coefficient table saved to %s", filename)

    return table_str

def real_normal_center_manifold_arrays(
    point,
    max_degree: int,
    *,
    psi=None,
    clmo=None
) -> List[np.ndarray]:
    """
    Return the centre-manifold Hamiltonian in **real normal variables**
    (x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn), dense-array form.
    """
    if psi is None or clmo is None:
        psi, clmo = init_index_tables(max_degree)

    # 1. complex canonical on the centre manifold (q1=p1=0)
    H_cnr = reduce_center_manifold_arrays(point, max_degree,
                                          psi=psi, clmo=clmo)

    # 2. back to real-normal frame
    H_rnr = cn2rn(H_cnr, max_degree, psi, clmo)

    return H_rnr

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
    # Ensure we have the actual psi and clmo tables for use in this function's scope
    # (e.g., for phys2rn, rn2cn, lie_transform)
    psi_tables_for_manifold_scope = psi
    clmo_tables_for_manifold_scope = clmo
    if psi_tables_for_manifold_scope is None or clmo_tables_for_manifold_scope is None:
        psi_tables_for_manifold_scope, clmo_tables_for_manifold_scope = init_index_tables(max_degree)

    # Prepare arguments for build_physical_hamiltonian to match its fixed internal logic:
    # 1. For its 'psi_config' parameter: build_physical_hamiltonian internally uses psi_config[1]
    #    as the complex_dtype (boolean). For the physical Hamiltonian (real), this should be False.
    #    The element at index 0 of this list is not used by build_physical_hamiltonian.
    bph_psi_config_arg = [None, False]

    # 2. For its 'clmo_tables_deg: int' parameter: build_physical_hamiltonian internally uses this
    #    as an integer degree for its own call to init_index_tables.
    bph_clmo_deg_arg = max_degree

    # Call build_physical_hamiltonian with these specially crafted arguments
    H_phys = build_physical_hamiltonian(point, max_degree)

    # Subsequent linear changes of coordinates and Lie transforms use the main psi/clmo tables
    # prepared at the beginning of this function.
    H_rn   = phys2rn(point, H_phys, max_degree, psi_tables_for_manifold_scope, clmo_tables_for_manifold_scope)
    H_cn   = rn2cn(H_rn, max_degree, psi_tables_for_manifold_scope, clmo_tables_for_manifold_scope)

    # Lie-series partial normal form
    H_cnt, G_tot = lie_transform(point, H_cn, psi_tables_for_manifold_scope, clmo_tables_for_manifold_scope, max_degree)

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