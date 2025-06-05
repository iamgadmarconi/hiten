from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import decode_multiindex
from algorithms.center.transforms import complexify, phys2rn, realify
from utils.log_config import logger


def center_manifold_real(point, psi, clmo, max_deg=5):
    """
    Compute the realified Hamiltonian restricted to the center manifold.
    Uses caching to avoid recomputation.
    """
    cached_cm_real = point.cache_get(('hamiltonian', max_deg, 'center_manifold_real'))

    if cached_cm_real is not None:
        return [h.copy() for h in cached_cm_real]

    poly_cm_complex = center_manifold_complex(point, psi, clmo, max_deg)
    poly_cm_real = realify(poly_cm_complex, max_deg, psi, clmo)

    point.cache_set(('hamiltonian', max_deg, 'center_manifold_real'), [h.copy() for h in poly_cm_real])
    
    return poly_cm_real


def center_manifold_complex(point, psi, clmo, max_deg=5):
    """
    Compute the Hamiltonian restricted to the center manifold in complex normal form.
    Uses caching to avoid recomputation and caches all intermediate representations.
    """

    cached_cm_complex = point.cache_get(('hamiltonian', max_deg, 'center_manifold_complex'))

    if cached_cm_complex is not None:
        return [h.copy() for h in cached_cm_complex]

    logger.info(f"Computing center manifold for {type(point).__name__}, max_deg={max_deg}")

    poly_phys = point.cache_get(('hamiltonian', max_deg, 'physical'))

    if poly_phys is None:

        poly_phys = build_physical_hamiltonian(point, max_deg)
        point.cache_set(('hamiltonian', max_deg, 'physical'), [h.copy() for h in poly_phys])

    else:
        poly_phys = [h.copy() for h in poly_phys]

    poly_rn = point.cache_get(('hamiltonian', max_deg, 'real_normal'))

    if poly_rn is None:
        poly_rn = phys2rn(point, poly_phys, max_deg, psi, clmo)
        point.cache_set(('hamiltonian', max_deg, 'real_normal'), [h.copy() for h in poly_rn])

    else:
        poly_rn = [h.copy() for h in poly_rn]

    poly_cn = point.cache_get(('hamiltonian', max_deg, 'complex_normal'))

    if poly_cn is None:
        poly_cn = complexify(poly_rn, max_deg, psi, clmo)
        point.cache_set(('hamiltonian', max_deg, 'complex_normal'), [h.copy() for h in poly_cn])

    else:
        poly_cn = [h.copy() for h in poly_cn]

    poly_trans = point.cache_get(('hamiltonian', max_deg, 'normalized'))
    poly_G_total = point.cache_get(('generating_functions', max_deg))
    
    if poly_trans is None or poly_G_total is None:
        poly_trans, poly_G_total = lie_transform(point, poly_cn, psi, clmo, max_deg)
        point.cache_set(('hamiltonian', max_deg, 'normalized'), [h.copy() for h in poly_trans])
        point.cache_set(('generating_functions', max_deg), [g.copy() for g in poly_G_total])
    
    else:
        if poly_trans is not None:
            poly_trans = [h.copy() for h in poly_trans]
        if poly_G_total is not None:
            poly_G_total = [g.copy() for g in poly_G_total]

    poly_cm_complex = restrict_to_center_manifold(poly_trans, clmo, tol=1e-14)
    point.cache_set(('hamiltonian', max_deg, 'center_manifold_complex'), [h.copy() for h in poly_cm_complex])

    logger.info(f"Center manifold computation complete for {type(point).__name__}")
    return poly_cm_complex


def restrict_to_center_manifold(poly_H, clmo, tol=1e-14):
    """
    Restrict a Hamiltonian to the center manifold by eliminating hyperbolic variables.
    
    Parameters
    ----------
    poly_H : List[numpy.ndarray]
        Polynomial representation of the Hamiltonian in normal form
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    tol : float, optional
        Tolerance for considering coefficients as zero, default is 1e-14
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the Hamiltonian restricted to the center manifold
        
    Notes
    -----
    The center manifold is obtained by setting the hyperbolic variables (q1, p1)
    to zero. This function filters out all monomials that contain non-zero
    powers of q1 or p1.
    
    In the packed multi-index format, q1 corresponds to k[0] and p1 corresponds to k[3].
    Any term with non-zero exponents for these variables is eliminated.
    
    Additionally, terms with coefficients smaller than the tolerance are set to zero.
    """
    poly_cm = [h.copy() for h in poly_H]
    for deg, coeff_vec in enumerate(poly_cm):
        if coeff_vec.size == 0:
            continue
        for pos, c in enumerate(coeff_vec):
            if abs(c) <= tol:
                coeff_vec[pos] = 0.0
                continue
            k = decode_multiindex(pos, deg, clmo)
            if k[0] != 0 or k[3] != 0:       # q1 or p1 exponent non-zero
                coeff_vec[pos] = 0.0
    return poly_cm
