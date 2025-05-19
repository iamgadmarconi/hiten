from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import decode_multiindex
from algorithms.center.transforms import cn2rn, phys2rn, rn2cn


def center_manifold_rn(point, psi, clmo, max_deg=5):
    """
    Compute the Hamiltonian restricted to the center manifold in real normal form.
    
    Parameters
    ----------
    point : object
        Object representing a collinear point, with methods for linearized dynamics
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    max_deg : int, optional
        Maximum degree for polynomial representations, default is 5
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the center manifold Hamiltonian in real normal form
        
    Notes
    -----
    This function first computes the center manifold Hamiltonian in complex normal form,
    then transforms it back to real normal form using the cn2rn transformation.
    
    The center manifold is the invariant manifold tangent to the center eigenspace
    of the linearized system at the equilibrium point. It contains all bounded
    dynamics near the equilibrium.
    """
    poly_cm_cn = center_manifold_cn(point, psi, clmo, max_deg)
    poly_cm_rn = cn2rn(poly_cm_cn, max_deg, psi, clmo)

    return poly_cm_rn


def center_manifold_cn(point, psi, clmo, max_deg=5):
    """
    Compute the Hamiltonian restricted to the center manifold in complex normal form.
    
    Parameters
    ----------
    point : object
        Object representing a collinear point, with methods for linearized dynamics
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
    max_deg : int, optional
        Maximum degree for polynomial representations, default is 5
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial representation of the center manifold Hamiltonian in complex normal form
        
    Notes
    -----
    This function performs a series of transformations:
    1. Builds the physical Hamiltonian in the original coordinates
    2. Transforms to real normal form coordinates around the equilibrium point
    3. Transforms to complex normal form coordinates
    4. Applies a Lie transform to normalize the Hamiltonian
    5. Restricts to the center manifold by setting all terms with hyperbolic variables to zero
    
    The resulting Hamiltonian describes the dynamics on the center manifold
    in complex normal form coordinates.
    """
    poly_phys = build_physical_hamiltonian(point, max_deg)
    poly_rn = phys2rn(point, poly_phys, max_deg, psi, clmo)
    poly_cn = rn2cn(poly_rn, max_deg, psi, clmo)
    poly_trans, _ = lie_transform(point, poly_cn, psi, clmo, max_deg)
    poly_cm_cn = restrict_to_center_manifold(poly_trans, clmo, tol=1e-14)

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
