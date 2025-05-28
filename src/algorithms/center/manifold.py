from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import decode_multiindex
from algorithms.center.transforms import cn2rn, phys2rn, rn2cn
from utils.log_config import logger


def center_manifold_rn(point, psi, clmo, max_deg=5):
    """
    Compute the Hamiltonian restricted to the center manifold in real normal form.
    Uses caching to avoid recomputation.
    """
    # Check cache first
    cached_cm_rn = point.get_cached_hamiltonian(max_deg, 'center_manifold_rn')
    if cached_cm_rn is not None:
        logger.debug(f"Using cached center manifold RN for {type(point).__name__}, max_deg={max_deg}")
        return cached_cm_rn

    # Compute if not cached
    poly_cm_cn = center_manifold_cn(point, psi, clmo, max_deg)
    poly_cm_rn = cn2rn(poly_cm_cn, max_deg, psi, clmo)

    # Cache the result
    point._store_hamiltonian_cache(max_deg, 'center_manifold_rn', poly_cm_rn)
    
    return poly_cm_rn


def center_manifold_cn(point, psi, clmo, max_deg=5):
    """
    Compute the Hamiltonian restricted to the center manifold in complex normal form.
    Uses caching to avoid recomputation and caches all intermediate representations.
    """
    # Check cache first
    cached_cm_cn = point.get_cached_hamiltonian(max_deg, 'center_manifold_cn')
    if cached_cm_cn is not None:
        logger.debug(f"Using cached center manifold CN for {type(point).__name__}, max_deg={max_deg}")
        return cached_cm_cn

    logger.info(f"Computing center manifold for {type(point).__name__}, max_deg={max_deg}")

    # Check for cached intermediate representations
    poly_phys = point.get_cached_hamiltonian(max_deg, 'physical')
    if poly_phys is None:
        from algorithms.center.hamiltonian import build_physical_hamiltonian
        poly_phys = build_physical_hamiltonian(point, max_deg)
        point._store_hamiltonian_cache(max_deg, 'physical', poly_phys)
        logger.debug("Computed and cached physical Hamiltonian")

    poly_rn = point.get_cached_hamiltonian(max_deg, 'real_normal') 
    if poly_rn is None:
        poly_rn = phys2rn(point, poly_phys, max_deg, psi, clmo)
        point._store_hamiltonian_cache(max_deg, 'real_normal', poly_rn)
        logger.debug("Computed and cached real normal Hamiltonian")

    poly_cn = point.get_cached_hamiltonian(max_deg, 'complex_normal')
    if poly_cn is None:
        poly_cn = rn2cn(poly_rn, max_deg, psi, clmo)
        point._store_hamiltonian_cache(max_deg, 'complex_normal', poly_cn)
        logger.debug("Computed and cached complex normal Hamiltonian")

    # Check for cached normalized Hamiltonian and generating functions
    poly_trans = point.get_cached_hamiltonian(max_deg, 'normalized')
    poly_G_total = point.get_cached_generating_functions(max_deg)
    
    if poly_trans is None or poly_G_total is None:
        poly_trans, poly_G_total = lie_transform(point, poly_cn, psi, clmo, max_deg)
        point._store_hamiltonian_cache(max_deg, 'normalized', poly_trans)
        point._store_generating_functions_cache(max_deg, poly_G_total)
        logger.debug("Computed and cached normalized Hamiltonian and generating functions")

    # Compute and cache center manifold restriction
    poly_cm_cn = restrict_to_center_manifold(poly_trans, clmo, tol=1e-14)
    point._store_hamiltonian_cache(max_deg, 'center_manifold_cn', poly_cm_cn)

    logger.info(f"Center manifold computation complete for {type(point).__name__}")
    return poly_cm_cn


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
