r"""
center.transforms
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

from hiten.algorithms.polynomial.base import _create_encode_dict_from_clmo
from hiten.algorithms.polynomial.coordinates import (_clean_coordinates,
                                               _substitute_coordinates)
from hiten.algorithms.polynomial.operations import (polynomial_clean,
                                              substitute_linear)
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.libration.triangular import TriangularPoint
from hiten.utils.log_config import logger


def M() -> np.ndarray:
    r"""
    Return the linear map from complex modal to real modal coordinates.

    Returns
    -------
    numpy.ndarray
        A :math:`6 x 6` complex-valued matrix :math:`M` such that
        :math:`\mathbf{z}_{\text{real}} = M\,\mathbf{z}_{\text{complex}}`.

    Notes
    -----
    The matrix is unitary up to scaling and preserves the canonical
    symplectic structure.
    """
    return np.array([[1, 0, 0, 0, 0, 0],
        [0, 1/np.sqrt(2), 0, 0, 1j/np.sqrt(2), 0],
        [0, 0, 1/np.sqrt(2), 0, 0, 1j/np.sqrt(2)],
        [0, 0, 0, 1, 0, 0],
        [0, 1j/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0],
        [0, 0, 1j/np.sqrt(2), 0, 0, 1/np.sqrt(2)]], dtype=np.complex128) #  real = M @ complex

def M_inv() -> np.ndarray:
    r"""
    Return the inverse transformation :math:`M^{-1}`.

    Returns
    -------
    numpy.ndarray
        The inverse of :pyfunc:`M`, satisfying
        :math:`\mathbf{z}_{\text{complex}} = M^{-1}\,\mathbf{z}_{\text{real}}`.
    """
    return np.linalg.inv(M()) # complex = M_inv @ real

def substitute_complex(poly_rn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    r"""
    Transform a polynomial from real normal form to complex normal form.
    
    Parameters
    ----------
    poly_rn : List[numpy.ndarray]
        Polynomial in real normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in complex normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from real normal form coordinates
    to complex normal form coordinates using the predefined transformation matrix M_inv().
    Since complex = M_inv @ real, we use M_inv() for the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return polynomial_clean(substitute_linear(poly_rn, M(), max_deg, psi, clmo, encode_dict_list), 1e-14)

def substitute_real(poly_cn: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
    r"""
    Transform a polynomial from complex normal form to real normal form.
    
    Parameters
    ----------
    poly_cn : List[numpy.ndarray]
        Polynomial in complex normal form coordinates
    max_deg : int
        Maximum degree for polynomial representations
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices
        
    Returns
    -------
    List[numpy.ndarray]
        Polynomial in real normal form coordinates
        
    Notes
    -----
    This function transforms a polynomial from complex normal form coordinates
    to real normal form coordinates using the predefined transformation matrix M().
    Since real = M @ complex, we use M() for the transformation.
    """
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    return polynomial_clean(substitute_linear(poly_cn, M_inv(), max_deg, psi, clmo, encode_dict_list), 1e-14)

def solve_complex(real_coords: np.ndarray) -> np.ndarray:
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
    return _clean_coordinates(_substitute_coordinates(real_coords, M_inv())) # [q1c, q2c, q3c, p1c, p2c, p3c]

def solve_real(real_coords: np.ndarray) -> np.ndarray:
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
    return _clean_coordinates(_substitute_coordinates(real_coords, M())) # [q1r, q2r, q3r, p1r, p2r, p3r]

def _local2realmodal(point, poly_local: List[np.ndarray], max_deg: int, psi, clmo) -> List[np.ndarray]:
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
        Combinatorial table from init_index_tables
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
    return substitute_linear(poly_local, C, max_deg, psi, clmo, encode_dict_list)

def _realmodal2local(point, modal_coords: np.ndarray) -> np.ndarray:
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
    return _clean_coordinates(C.dot(modal_coords))

def _local2synodic_collinear(point: CollinearPoint, local_coords: np.ndarray) -> np.ndarray:
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

    tol = 1e-16
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

def _local2synodic_triangular(point: TriangularPoint, local_coords: np.ndarray) -> np.ndarray:
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

    tol = 1e-16
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