"""Provide coordinate transformations for CR3BP normal form computations.

This module provides comprehensive coordinate transformation utilities for the
center manifold normal form pipeline in the Circular Restricted Three-Body
Problem (CR3BP). It implements transformations between multiple coordinate
systems including local, modal, complex, and synodic frames around both
collinear and triangular equilibrium points.

The transformation pipeline enables the construction of normal forms by working
in the most appropriate coordinate system at each stage of the computation,
from physical coordinates through modal coordinates to complex normal forms.

Coordinate Systems
------------------
- **Local**: Coordinates centered at equilibrium point
- **Synodic**: Standard CR3BP rotating frame coordinates
- **Modal**: Coordinates aligned with linear stability eigenvectors
- **Complex**: Complexified coordinates for elliptic directions

See Also
--------
:mod:`~hiten.algorithms.hamiltonian.center`
    Center manifold computations using these transformations.
:mod:`~hiten.algorithms.hamiltonian.normal`
    Normal form computations in various coordinate systems.
:mod:`~hiten.system.libration.collinear`
    Collinear libration point classes with transformation methods.
:mod:`~hiten.system.libration.triangular`
    Triangular libration point classes with transformation methods.

References
----------
Jorba, A. (1999). A methodology for the numerical computation of normal forms,
centre manifolds and first integrals of Hamiltonian systems. Experimental
Mathematics, 8(2), 155-195.

Gomez, G., Llibre, J., Martinez, R., Simo, C. (2001). Dynamics and Mission Design
Near Libration Points. World Scientific, Chapter 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba.typed import List

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _decode_multiindex)
from hiten.algorithms.polynomial.coordinates import (_clean_coordinates,
                                                     _substitute_coordinates)
from hiten.algorithms.polynomial.operations import (_polynomial_clean,
                                                    _substitute_linear)
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint


def _build_complexification_matrix(mix_indices):
    """Build complexification transformation matrix for canonical coordinate pairs.
    
    The transformation is given by:
    q_j(complex) = (q_j(real) + i*p_j(real)) / sqrt(2)
    p_j(complex) = (i*q_j(real) + p_j(real)) / sqrt(2)
    where j is the index of the canonical coordinate pair.

    Parameters
    ----------
    mix_indices : tuple of int
        Indices of canonical coordinate pairs to complexify.

    Returns
    -------
    np.ndarray
        Complexification transformation matrix.

    References
    ----------
    Jorba, A. (1999). A methodology for the numerical computation of normal forms,
    centre manifolds and first integrals of Hamiltonian systems. Experimental
    Mathematics, 8(2), 155-195.
    """

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

        # Fill-in the 2x2 mixing block for the selected canonical pair.
        # q_j(real)  =  (      q_j^c +   i p_j^c) / sqrt(2)
        # p_j(real)  =  (  i q_j^c +       p_j^c) / sqrt(2)
        M[q_idx, q_idx] = half
        M[q_idx, p_idx] = 1j * half
        M[p_idx, q_idx] = 1j * half
        M[p_idx, p_idx] = half

    return M

def _M(mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Return complexification transformation matrix for canonical coordinate pairs.

    Generate the unitary transformation matrix that converts real canonical
    coordinates to complex coordinates for specified coordinate pairs, typically
    used to complexify elliptic directions while leaving hyperbolic directions real.

    Parameters
    ----------
    mix_pairs : tuple of int, default (1, 2)
        Indices of canonical coordinate pairs to complexify. Each index j
        corresponds to the pair (q_{j+1}, p_{j+1}) where j=0,1,2 maps to
        pairs (q1,p1), (q2,p2), (q3,p3) respectively.

    Returns
    -------
    ndarray, shape (6, 6), complex
        Unitary complexification matrix M where complex_coords = M @ real_coords.
        The matrix is structured as 2x2 blocks for each canonical pair.

    Notes
    -----
    For each coordinate pair j in mix_pairs, the transformation is:
    q_j(complex) = (q_j(real) + i*p_j(real)) / sqrt(2)
    p_j(complex) = (i*q_j(real) + p_j(real)) / sqrt(2)

    This complexification is particularly useful for collinear libration points
    where (q1, p1) corresponds to the hyperbolic direction (left real) while
    (q2, p2) and (q3, p3) correspond to elliptic directions (complexified).

    The transformation preserves the canonical structure and enables efficient
    normal form computations in the complex domain where elliptic motion
    becomes purely rotational.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._M_inv`
        Inverse transformation from complex to real coordinates.
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Apply complexification to polynomial expressions.

    References
    ----------
    Meyer, K.R., Hall, G.R. (1992). Introduction to Hamiltonian Dynamical
    Systems and the N-Body Problem. Springer-Verlag, Section 4.3.
    """
    return _build_complexification_matrix(mix_pairs)

def _M_inv(mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Return inverse complexification matrix for real coordinate recovery.

    Compute the inverse of the complexification matrix M, which transforms
    complex canonical coordinates back to real coordinates. Since M is unitary,
    the inverse is computed as the conjugate transpose.

    Parameters
    ----------
    mix_pairs : tuple of int, default (1, 2)
        Indices of canonical coordinate pairs that were complexified.
        Must match the mix_pairs used in the forward transformation.

    Returns
    -------
    ndarray, shape (6, 6), complex
        Inverse complexification matrix M_inv where real_coords = M_inv @ complex_coords.
        Computed as the conjugate transpose of M due to unitarity.

    Notes
    -----
    The inverse transformation recovers real coordinates from complex coordinates:
    q_j(real) = Re(q_j(complex)) * sqrt(2)
    p_j(real) = Im(q_j(complex)) * sqrt(2)

    This is used to convert results from complex normal form computations
    back to real coordinates for physical interpretation and integration.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._M`
        Forward complexification transformation.
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Apply inverse complexification to polynomial expressions.
    """
    M = _M(mix_pairs)
    return M.conjugate().T  # complex = M_inv @ real


def _substitute_complex(poly_rn: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12, *, mix_pairs: tuple[int, ...] = (1, 2)) -> List[np.ndarray]:
    """Transform polynomial from real to complex coordinates.

    Apply complexification transformation to a polynomial expressed in real
    normal form coordinates, converting it to complex coordinates suitable
    for elliptic normal form analysis.

    Parameters
    ----------
    poly_rn : List[ndarray]
        Polynomial coefficients in real normal form coordinates, organized
        by degree, in nondimensional energy units.
    max_deg : int
        Maximum polynomial degree to retain in the transformation.
    psi : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    tol : float, default 1e-12
        Numerical tolerance for cleaning small coefficients after transformation.
    mix_pairs : tuple of int, default (1, 2)
        Canonical coordinate pairs to complexify.

    Returns
    -------
    List[ndarray]
        Polynomial coefficients in complex coordinates, organized by degree,
        in nondimensional energy units.

    Notes
    -----
    The transformation uses the complexification matrix M to convert polynomial
    expressions from real to complex coordinates. This enables efficient
    computation of normal forms around elliptic equilibria where the complex
    representation naturally captures rotational motion.

    The transformation preserves polynomial structure and energy scaling,
    with complex coordinates maintaining the same physical units as their
    real counterparts.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Inverse transformation from complex to real coordinates.
    :func:`~hiten.algorithms.hamiltonian.transforms._M`
        Complexification matrix used in the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_rn, _M(mix_pairs), max_deg, psi, clmo, encode_dict_list), tol)

def _substitute_real(poly_cn: List[np.ndarray], max_deg: int, psi, clmo, tol=1e-12, *, mix_pairs: tuple[int, ...] = (1, 2)) -> List[np.ndarray]:
    """Transform polynomial from complex to real coordinates.

    Apply inverse complexification transformation to convert a polynomial from
    complex coordinates back to real normal form coordinates for physical
    interpretation and further analysis.

    Parameters
    ----------
    poly_cn : List[ndarray]
        Polynomial coefficients in complex coordinates, organized by degree,
        in nondimensional energy units.
    max_deg : int
        Maximum polynomial degree to retain in the transformation.
    psi : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    tol : float, default 1e-12
        Numerical tolerance for cleaning small coefficients after transformation.
    mix_pairs : tuple of int, default (1, 2)
        Canonical coordinate pairs that were complexified.

    Returns
    -------
    List[ndarray]
        Polynomial coefficients in real coordinates, organized by degree,
        in nondimensional energy units.

    Notes
    -----
    This is the inverse of :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`, using the inverse
    complexification matrix M_inv to recover real polynomial expressions
    from their complex representations.

    The transformation is essential for converting normal form results back
    to real coordinates for physical interpretation, trajectory integration,
    and comparison with numerical simulations.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Forward transformation from real to complex coordinates.
    :func:`~hiten.algorithms.hamiltonian.transforms._M_inv`
        Inverse complexification matrix used in the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return _polynomial_clean(_substitute_linear(poly_cn, _M_inv(mix_pairs), max_deg, psi, clmo, encode_dict_list), tol)

def _solve_complex(real_coords: np.ndarray, tol: float = 1e-30, *, mix_pairs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Transform real coordinates to complex coordinates.

    Convert a real coordinate vector to complex coordinates using the
    complexification transformation, typically for elliptic normal form analysis.

    Parameters
    ----------
    real_coords : ndarray, shape (6,)
        Real canonical coordinates [q1, q2, q3, p1, p2, p3] in nondimensional
        position and momentum units.
    tol : float, default 1e-30
        Tolerance for cleaning small imaginary parts in the result.
    mix_pairs : tuple of int, default (1, 2)
        Canonical coordinate pairs to complexify.

    Returns
    -------
    ndarray, shape (6,), complex
        Complex coordinates [q1c, q2c, q3c, p1c, p2c, p3c] where complexified
        pairs are in complex units and unmixed pairs remain real.

    Notes
    -----
    This transformation is used to convert initial conditions or intermediate
    results from real coordinates to the complex representation needed for
    normal form analysis around elliptic equilibria.

    The transformation preserves the canonical structure and maintains
    appropriate scaling for further computations.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._solve_real`
        Inverse transformation from complex to real coordinates.
    :func:`~hiten.algorithms.hamiltonian.transforms._M_inv`
        Complexification matrix used in this transformation.
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
    C, _ = point.normal_form_transform
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
    _, C_inv = point.normal_form_transform
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
    C, _ = point.normal_form_transform
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
    _, C_inv = point.normal_form_transform
    return _clean_coordinates(C_inv.dot(local_coords), tol)

def _local2synodic_collinear(point: CollinearPoint, local_coords: np.ndarray, tol=1e-14) -> np.ndarray:
    """Transform coordinates from local to synodic frame for collinear points.

    Convert coordinates from the local frame centered at a collinear equilibrium
    point to the standard CR3BP synodic rotating frame coordinates.

    Parameters
    ----------
    point : CollinearPoint
        Collinear libration point object providing geometric parameters gamma,
        mu, sign, and a for the coordinate transformation.
    local_coords : ndarray, shape (6,)
        Local coordinates [x1, x2, x3, px1, px2, px3] in nondimensional units
        where positions are in distance units and momenta are canonical.
    tol : float, default 1e-14
        Tolerance for detecting non-negligible imaginary parts in input coordinates.

    Returns
    -------
    ndarray, shape (6,)
        Synodic coordinates [X, Y, Z, Vx, Vy, Vz] in nondimensional CR3BP units
        where positions are distances and velocities are time derivatives.

    Raises
    ------
    ValueError
        If local_coords is not a flat array of length 6 or contains imaginary
        parts larger than the specified tolerance.

    Notes
    -----
    The transformation implements the mapping from local coordinates centered
    at the equilibrium point to the standard synodic frame where the primaries
    are located at (-mu, 0, 0) and (1-mu, 0, 0).

    The coordinate transformation includes:
    - Position scaling by the characteristic length gamma
    - Translation by the equilibrium point offset (mu + a)
    - Momentum to velocity conversion with Coriolis terms
    - Sign convention adjustments for NASA/Szebehely compatibility

    Local frame: centered at equilibrium, canonical coordinates
    Synodic frame: standard CR3BP rotating frame, Cartesian coordinates

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._synodic2local_collinear`
        Inverse transformation from synodic to local coordinates.
    :class:`~hiten.system.libration.collinear.CollinearPoint`
        Collinear point class providing transformation parameters.

    References
    ----------
    Szebehely, V. (1967). Theory of Orbits. Academic Press, Chapter 7.
    """
    gamma, mu, sgn, a = point.dynamics.gamma, point.mu, point.dynamics.sign, point.dynamics.a

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

    This is the exact inverse of :func:`~hiten.algorithms.hamiltonian.transforms._local2synodic_collinear`.

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
    gamma, mu, sgn, a = point.dynamics.gamma, point.mu, point.dynamics.sign, point.dynamics.a

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
    mu, sgn = point.mu, point.dynamics.sign

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

    This is the exact inverse of :func:`~hiten.algorithms.hamiltonian.transforms._local2synodic_triangular`.

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
    mu, sgn = point.mu, point.dynamics.sign

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
    """Restrict polynomial Hamiltonian to center manifold by eliminating hyperbolic terms.

    Project a polynomial Hamiltonian onto the center manifold by removing all
    terms that depend on hyperbolic variables, retaining only the dynamics
    within the center-stable/center-unstable subspace.

    Parameters
    ----------
    point : CollinearPoint or TriangularPoint
        Libration point object determining the manifold structure.
    poly_H : List[ndarray]
        Polynomial Hamiltonian coefficients organized by degree, in
        nondimensional energy units.
    clmo : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    tol : float, default 1e-14
        Tolerance for zeroing small coefficients during restriction.

    Returns
    -------
    List[ndarray]
        Restricted polynomial with hyperbolic terms eliminated, maintaining
        the same degree structure as the input.

    Notes
    -----
    For collinear points, the first canonical pair (q1, p1) corresponds to
    the hyperbolic direction, so all terms with non-zero exponents in these
    variables are eliminated. This leaves only the center manifold dynamics
    in the (q2, p2, q3, p3) subspace.

    For triangular points, all directions are elliptic (center-type), so
    the function returns a copy of the original polynomial without restriction.

    This restriction is fundamental to center manifold theory, which reduces
    the dimensionality of the dynamics by focusing on the neutrally stable
    directions while eliminating the exponentially growing/decaying modes.

    See Also
    --------
    :func:`~hiten.algorithms.polynomial.base._decode_multiindex`
        Multi-index decoding used to identify hyperbolic terms.
    :class:`~hiten.system.libration.collinear.CollinearPoint`
        Collinear points with hyperbolic directions.
    :class:`~hiten.system.libration.triangular.TriangularPoint`
        Triangular points with all elliptic directions.

    References
    ----------
    Carr, J. (1981). Applications of Centre Manifold Theory. Springer-Verlag.

    Jorba, A., Masdemont, J. (1999). Dynamics in the center manifold of the
    collinear points. Physica D, 132(1-2), 189-213.
    """
    # For triangular points, all directions are centre-type, so we do NOT
    # eliminate any terms involving (q1, p1).  The original behaviour of
    # zeroing these terms is only appropriate for collinear points where
    # (q1, p1) span the hyperbolic sub-space.
    from hiten.system.libration.triangular import TriangularPoint
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