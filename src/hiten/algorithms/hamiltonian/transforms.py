r"""
hamiltonian.transforms
=================

Linear coordinate transformations and helper utilities used in the centre
manifold normal-form pipeline of the spatial circular restricted three body
problem (CRTBP).

References
----------
Jorba, À. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".
"""

import numpy as np
from numba.typed import List

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _decode_multiindex)
from hiten.algorithms.polynomial.coordinates import (_clean_coordinates,
                                                     _substitute_coordinates)
from hiten.algorithms.polynomial.operations import (_polynomial_clean,
                                                    _substitute_linear)
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.libration.triangular import TriangularPoint
from hiten.utils.log_config import logger


def _build_complexification_matrix(mix_indices):

    half = 1.0 / np.sqrt(2.0)

    # Start with identity (pairs not mixed are left untouched).
    M = np.eye(6, dtype=np.complex128)

    for j in mix_indices:
        q_idx = j       # q1, q2, q3  -> indices 0,1,2
        p_idx = 3 + j   # p1, p2, p3  -> indices 3,4,5

        # Zero-out the rows we are about to overwrite (they currently contain
        # the identity entries inserted by np.eye).
        M[q_idx, :] = 0.0
        M[p_idx, :] = 0.0

        # Fill-in the 2×2 mixing block for the selected canonical pair.
        # q_j(real)  =  (      q_j^c +   i p_j^c) / sqrt(2)
        # p_j(real)  =  (  i q_j^c +       p_j^c) / sqrt(2)
        M[q_idx, q_idx] = half
        M[q_idx, p_idx] = 1j * half
        M[p_idx, q_idx] = 1j * half
        M[p_idx, p_idx] = half

    return M

def _M(mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    r"""
    Return the canonical complexification matrix that *only* mixes the second
    and third canonical pairs, leaving the first pair \((q_1, p_1)\) real.

    This corresponds to the typical linearised dynamics around the collinear
    libration points, where \( (q_1, p_1) \) is hyperbolic while the other two
    pairs are elliptic and therefore profit from the complex representation.
    """
    return _build_complexification_matrix(mix_pairs)

def _M_inv(mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    r"""Inverse of :pyfunc:`_M`.  Because the matrix is unitary we can use the
    conjugate transpose rather than an explicit matrix inversion."""
    M = _M(mix_pairs)
    return M.conjugate().T  # complex = M_inv @ real


def _substitute_complex(poly_rn: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12, *, mix_pairs: tuple[int, ...] = (1, 2)) -> List[np.ndarray]:
    r"""
    Transform a polynomial from real normal form to complex normal form.
    
    Parameters
    ----------
    poly_rn : List[numpy.ndarray]
        Polynomial in real normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in complex normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from real normal form coordinates
    to complex normal form coordinates using the predefined transformation matrix _M_inv().
    Since complex = M_inv @ real, we use _M_inv() for the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_rn, _M(mix_pairs), max_deg, psi, clmo, encode_dict_list), tol)

def _substitute_real(poly_cn: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12, *, mix_pairs: tuple[int, ...] = (1, 2)) -> List[np.ndarray]:
    r"""
    Transform a polynomial from complex normal form to real normal form.
    
    Parameters
    ----------
    poly_cn : List[numpy.ndarray]
        Polynomial in complex normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in real normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from complex normal form coordinates
    to real normal form coordinates using the predefined transformation matrix _M().
    Since real = M @ complex, we use _M() for the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_cn, _M_inv(mix_pairs), max_deg, psi, clmo, encode_dict_list), tol)

def _solve_complex(real_coords: np.ndarray, tol: float = 1e-30, *, mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    r"""
    Return complex coordinates given real coordinates using the map `M_inv`.

    Parameters
    ----------
    real_coords : np.ndarray
        Real coordinates [q1, q2, q3, p1, p2, p3]

    Returns
    -------
    np.ndarray
        Complex coordinates [q1c, q2c, q3c, p1c, p2c, p3c]
    """
    return _clean_coordinates(_substitute_coordinates(real_coords, _M_inv(mix_pairs)), tol)


def _solve_real(real_coords: np.ndarray, tol: float = 1e-30, *, mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    r"""
    Return real coordinates given complex coordinates using the map `M`.

    Parameters
    ----------
    real_coords : np.ndarray
        Real coordinates [q1, q2, q3, p1, p2, p3]

    Returns
    -------
    np.ndarray
        Real coordinates [q1r, q2r, q3r, p1r, p2r, p3r]
    """
    return _clean_coordinates(_substitute_coordinates(real_coords, _M(mix_pairs)), tol)


def _polylocal2realmodal(point, poly_local: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12) -> List[np.ndarray]:
    r"""
    Transform a polynomial from local frame to real modal frame.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    poly_phys : List[numpy.ndarray]
        Polynomial in physical coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in real modal coordinates
        
    Notes
    -----
    This function transforms a polynomial from local coordinates to
    real modal coordinates using the transformation matrix obtained
    from the point object.
    """
    C, _ = point.normal_form_transform()
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_local, C, max_deg, psi, clmo, encode_dict_list), tol)

def _polyrealmodal2local(point, poly_realmodal: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12) -> List[np.ndarray]:
    r"""
    Transform a polynomial from real modal frame to local frame.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    poly_realmodal : List[numpy.ndarray]
        Polynomial in real modal coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from _init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in local coordinates
        
    Notes
    -----
    This function transforms a polynomial from real modal coordinates to
    local coordinates using the inverse of the transformation matrix obtained
    from the point object.
    """
    _, C_inv = point.normal_form_transform()
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_realmodal, C_inv, max_deg, psi, clmo, encode_dict_list), tol)

def _coordrealmodal2local(point, modal_coords: np.ndarray, tol=1e-30) -> np.ndarray:
    r"""
    Transform coordinates from real modal to local frame.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    modal_coords : np.ndarray
        Coordinates in real modal frame

    Returns
    -------
    np.ndarray
        Coordinates in local frame

    Notes
    -----
    - Modal coordinates are ordered as [q1, q2, q3, px1, px2, px3].
    - Local coordinates are ordered as [x1, x2, x3, px1, px2, px3].
    """
    C, _ = point.normal_form_transform()
    return _clean_coordinates(C.dot(modal_coords), tol)

def _coordlocal2realmodal(point, local_coords: np.ndarray, tol=1e-30) -> np.ndarray:
    r"""
    Transform coordinates from local to real modal frame.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    local_coords : np.ndarray
        Coordinates in local frame

    Returns
    -------
    np.ndarray
        Coordinates in real modal frame

    Notes
    -----
    - Local coordinates are ordered as [x1, x2, x3, px1, px2, px3].
    - Modal coordinates are ordered as [q1, q2, q3, px1, px2, px3].
    """
    _, C_inv = point.normal_form_transform()
    return _clean_coordinates(C_inv.dot(local_coords), tol)

def _local2synodic_collinear(point: CollinearPoint, local_coords: np.ndarray, tol=1e-14) -> np.ndarray:
    r"""
    Transform coordinates from local to synodic frame for the collinear points.

    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    local_coords : np.ndarray
        Coordinates in local frame

    Returns
    -------
    np.ndarray
        Coordinates in synodic frame

    Notes
    -----
    - Local coordinates are ordered as [x1, x2, x3, px1, px2, px3].
    - Synodic coordinates are ordered as [X, Y, Z, Vx, Vy, Vz].

    Raises
    ------
    ValueError
        If *local_coords* is not a flat array of length 6 or contains an
        imaginary part larger than the tolerance (``1e-16``).
    """
    gamma, mu, sgn, a = point.gamma, point.mu, point.sign, point.a

    c_complex = np.asarray(local_coords, dtype=np.complex128)
    if np.any(np.abs(np.imag(c_complex)) > tol):
        err = f"_local2synodic_collinear received coords with non-negligible imaginary part; max |Im(coords)| = {np.max(np.abs(np.imag(c_complex))):.3e} > {tol}."
        logger.error(err)
        raise ValueError(err)

    # From here on we work with the real part only.
    c = c_complex.real.astype(np.float64)

    if c.ndim != 1 or c.size != 6:
        raise ValueError(
            f"coords must be a flat array of 6 elements, got shape {c.shape}"
        )

    syn = np.empty(6, dtype=np.float64)

    # Positions
    syn[0] = sgn * gamma * c[0] + mu + a # X
    syn[1] = sgn * gamma * c[1] # Y
    syn[2] = gamma * c[2]  # Z

    # Local momenta to synodic velocities
    vx = c[3] + c[1]
    vy = c[4] - c[0]
    vz = c[5]

    syn[3] = gamma * vx  # Vx
    syn[4] = gamma * vy  # Vy
    syn[5] = gamma * vz  # Vz

    # Flip X and Vx according to NASA/Szebehely convention (see standard relations)
    syn[[0, 3]] *= -1.0

    return syn

def _synodic2local_collinear(point: CollinearPoint, synodic_coords: np.ndarray, tol=1e-14) -> np.ndarray:
    r"""
    Transform coordinates from synodic to local frame for the collinear points.

    This is the exact inverse of :func:`_local2synodic_collinear`.

    Parameters
    ----------
    point : CollinearPoint
        Collinear libration point providing the geometric parameters ``gamma``,
        ``mu``, ``sign`` and ``a``.
    synodic_coords : np.ndarray
        Coordinates in synodic frame ``[X, Y, Z, Vx, Vy, Vz]``.

    Returns
    -------
    np.ndarray
        Coordinates in local frame ``[x1, x2, x3, px1, px2, px3]``.

    Raises
    ------
    ValueError
        If *synodic_coords* is not a flat array of length 6 or contains an
        imaginary part larger than the tolerance (``1e-16``).
    """
    gamma, mu, sgn, a = point.gamma, point.mu, point.sign, point.a

    s_complex = np.asarray(synodic_coords, dtype=np.complex128)
    if np.any(np.abs(np.imag(s_complex)) > tol):
        err = (
            f"_synodic2local_collinear received coords with non-negligible imaginary part; "
            f"max |Im(coords)| = {np.max(np.abs(np.imag(s_complex))):.3e} > {tol}."
        )
        logger.error(err)
        raise ValueError(err)

    s = s_complex.real.astype(np.float64)

    if s.ndim != 1 or s.size != 6:
        raise ValueError(
            f"coords must be a flat array of 6 elements, got shape {s.shape}"
        )

    # Allocate output array
    local = np.empty(6, dtype=np.float64)

    # Invert position mapping (X was translated and scaled by gamma, with a sign adjustment)
    # X coordinate
    local[0] = (-s[0] - mu - a) / (sgn * gamma)
    # Y coordinate
    local[1] = s[1] / (sgn * gamma)
    # Z coordinate
    local[2] = s[2] / gamma

    # Invert velocity mapping
    # px1 from Vx (note the sign flip on Vx)
    local[3] = -s[3] / gamma - local[1]
    # px2 from Vy
    local[4] = s[4] / gamma + local[0]
    # px3 from Vz
    local[5] = s[5] / gamma

    return local

def _local2synodic_triangular(point: TriangularPoint, local_coords: np.ndarray, tol=1e-14) -> np.ndarray:
    r"""
    Transform coordinates from local to synodic frame for the equilateral points.
    
    Parameters
    ----------
    point : object
        An object with a normal_form_transform method that returns the transformation matrix
    local_coords : np.ndarray
        Coordinates in local frame

    Returns
    -------
    np.ndarray
        Coordinates in synodic frame

    Notes
    -----
    - Local coordinates are ordered as [x1, x2, x3, px1, px2, px3].
    - Synodic coordinates are ordered as [X, Y, Z, Vx, Vy, Vz].

    Raises
    ------
    ValueError
        If *local_coords* is not a flat array of length 6 or contains an
        imaginary part larger than the tolerance (``1e-16``).
    """
    mu, sgn = point.mu, point.sign

    c_complex = np.asarray(local_coords, dtype=np.complex128)
    if np.any(np.abs(np.imag(c_complex)) > tol):
        err = f"_local2synodic_triangular received coords with non-negligible imaginary part; max |Im(coords)| = {np.max(np.abs(np.imag(c_complex))):.3e} > {tol}."
        logger.error(err)
        raise ValueError(err)

    # From here on we work with the real part only.
    c = c_complex.real.astype(np.float64)

    if c.ndim != 1 or c.size != 6:
        raise ValueError(
            f"coords must be a flat array of 6 elements, got shape {c.shape}"
        )

    syn = np.empty(6, dtype=np.float64)

    # Positions
    syn[0] = c[0] - mu + 1 / 2 # X
    syn[1] = c[1] + sgn * np.sqrt(3) / 2 # Y
    syn[2] = c[2]  # Z

    # Local momenta to synodic velocities
    vx = c[3] - sgn * np.sqrt(3) / 2
    vy = c[4] - mu  + 1 / 2
    vz = c[5]

    syn[3] = vx  # Vx
    syn[4] = vy  # Vy
    syn[5] = vz  # Vz

    # Flip X and Vx according to NASA/Szebehely convention (see standard relations)
    syn[[0, 3]] *= -1.0

    return syn

def _synodic2local_triangular(point: TriangularPoint, synodic_coords: np.ndarray, tol=1e-14) -> np.ndarray:
    r"""
    Transform coordinates from synodic to local frame for the triangular (equilateral) points.

    This is the exact inverse of :func:`_local2synodic_triangular`.

    Parameters
    ----------
    point : TriangularPoint
        Triangular libration point providing the geometric parameters ``mu``
        and ``sign``.
    synodic_coords : np.ndarray
        Coordinates in synodic frame ``[X, Y, Z, Vx, Vy, Vz]``.

    Returns
    -------
    np.ndarray
        Coordinates in local frame ``[x1, x2, x3, px1, px2, px3]``.

    Raises
    ------
    ValueError
        If *synodic_coords* is not a flat array of length 6 or contains an
        imaginary part larger than the tolerance (``1e-16``).
    """
    mu, sgn = point.mu, point.sign

    s_complex = np.asarray(synodic_coords, dtype=np.complex128)
    if np.any(np.abs(np.imag(s_complex)) > tol):
        err = (
            f"_synodic2local_triangular received coords with non-negligible imaginary part; "
            f"max |Im(coords)| = {np.max(np.abs(np.imag(s_complex))):.3e} > {tol}."
        )
        logger.error(err)
        raise ValueError(err)

    s = s_complex.real.astype(np.float64)

    if s.ndim != 1 or s.size != 6:
        raise ValueError(
            f"coords must be a flat array of 6 elements, got shape {s.shape}"
        )

    # Allocate output array
    local = np.empty(6, dtype=np.float64)

    # Invert position mapping (forward transform shifted X by mu - 0.5 and flipped its sign)
    local[0] = mu - 0.5 - s[0]  # x1
    local[1] = s[1] - sgn * np.sqrt(3) / 2  # x2
    local[2] = s[2]  # x3 (Z)

    # Invert velocity mapping (forward transform flipped Vx's sign and shifted Vy by mu - 0.5)
    local[3] = sgn * np.sqrt(3) / 2 - s[3]  # px1 from Vx (with sign flip)
    local[4] = s[4] + mu - 0.5  # px2 from Vy
    local[5] = s[5]  # px3 from Vz

    return local


def _restrict_poly_to_center_manifold(point, poly_H, clmo, tol=1e-14):
    r"""
    Restrict a Hamiltonian to the center manifold by eliminating hyperbolic variables.
    """
    # For triangular points, all directions are centre-type, so we do NOT
    # eliminate any terms involving (q1, p1).  The original behaviour of
    # zeroing these terms is only appropriate for collinear points where
    # (q1, p1) span the hyperbolic sub-space.

    if isinstance(point, TriangularPoint):
        # Simply return a *copy* of the input to avoid accidental mutation
        return [h.copy() for h in poly_H]

    # Collinear case - remove all terms containing q1 or p1 exponents.
    poly_cm = [h.copy() for h in poly_H]
    for deg, coeff_vec in enumerate(poly_cm):
        if coeff_vec.size == 0:
            continue
        for pos, c in enumerate(coeff_vec):
            if abs(c) <= tol:
                coeff_vec[pos] = 0.0
                continue
            k = _decode_multiindex(pos, deg, clmo)
            if k[0] != 0 or k[3] != 0:       # q1 or p1 exponent non-zero
                coeff_vec[pos] = 0.0
    return poly_cm