from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import decode_multiindex
from algorithms.center.transforms import cn2rn, phys2rn, rn2cn


def center_manifold_rn(point, psi, clmo, max_deg=5):
    H_cm_cn = center_manifold_cn(point, psi, clmo, max_deg)
    H_cm_rn = cn2rn(H_cm_cn, max_deg, psi, clmo)

    return H_cm_rn


def center_manifold_cn(point, psi, clmo, max_deg=5):
    # 0) physical quadratic-diagonalisation
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn   = phys2rn(point, H_phys, max_deg, psi, clmo)
    H_cn   = rn2cn(H_rn,     max_deg, psi, clmo)

    # 1) Lie-series partial normal form
    H_trans, _ = lie_transform(point, H_cn, psi, clmo, max_deg)

    # 2) restrict to I1 = q1 p1 = 0  (centre manifold)
    H_cm_cn = restrict_to_center_manifold(H_trans, clmo, tol=1e-14)

    # 3) H_cm_cn is a list where index is degree.
    # The full list is returned for further processing.
    # table_slices = [H_cm_cn[d] for d in range(3, min(5, max_deg) + 1)]

    return H_cm_cn


def restrict_to_center_manifold(H_cn, clmo, tol=1e-14):
    """
    Zero all monomials that contain q1 or p1 (k0 or k3 > 0).
    tol :  coefficients with |c| <= tol are also set to zero.
    """
    H_cm = [h.copy() for h in H_cn]
    for deg, coeff_vec in enumerate(H_cm):
        if coeff_vec.size == 0:
            continue
        for pos, c in enumerate(coeff_vec):
            if abs(c) <= tol:
                coeff_vec[pos] = 0.0
                continue
            k = decode_multiindex(pos, deg, clmo)
            if k[0] != 0 or k[3] != 0:       # q1 or p1 exponent non-zero
                coeff_vec[pos] = 0.0
    return H_cm
