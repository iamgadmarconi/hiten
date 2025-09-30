"""Provide polynomial Hamiltonian construction for CR3BP equilibrium points.

This module provides polynomial representations of the Circular Restricted Three-Body
Problem (CR3BP) Hamiltonian around collinear and triangular equilibrium points.
The routines generate multivariate polynomial expansions that encode the rotating-frame
Hamiltonian up to a prescribed truncation degree in nondimensional units.

The polynomial objects form the algebraic foundation for center-manifold reductions,
normal-form computations, and invariant manifold analyses throughout the package.

See Also
--------
:mod:`~hiten.algorithms.polynomial.base`
    Polynomial data structures and indexing operations.
:mod:`~hiten.algorithms.polynomial.operations`
    Polynomial arithmetic and manipulation routines.
:mod:`~hiten.system.libration`
    Libration point classes that provide coefficient sequences.

References
----------
Jorba, A., Masdemont, J. (1999). Dynamics in the center manifold of the collinear 
points of the restricted three body problem. Physica D, 132(1-2), 189-213.

Gomez, G., Llibre, J., Martinez, R., Simo, C. (2001). Dynamics and Mission Design 
Near Libration Points. World Scientific.
"""

from typing import Tuple

import numpy as np
from numba import njit, types
from numba.typed import List

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.polynomial.operations import (_polynomial_add_inplace,
                                                    _polynomial_multiply,
                                                    _polynomial_variable,
                                                    _polynomial_zero_list)
from hiten.algorithms.utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _build_T_polynomials(poly_x, poly_y, poly_z, max_deg: int, psi_table, clmo_table, encode_dict_list) -> types.ListType:
    """Compute Chebyshev polynomials of the first kind for collinear point expansions.

    Generate the sequence T_n(r) where r = x / sqrt(x^2 + y^2 + z^2) using
    the classical Chebyshev recurrence relation adapted for Cartesian coordinates
    in nondimensional CR3BP units.

    Parameters
    ----------
    poly_x, poly_y, poly_z : List[ndarray]
        Polynomial representations of Cartesian coordinates x, y, z in
        nondimensional distance units.
    max_deg : int
        Maximum polynomial degree n such that T_n is computed.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo_table : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    encode_dict_list : List[dict]
        Lookup tables mapping multi-indices to coefficient positions.

    Returns
    -------
    List[List[ndarray]]
        Numba typed list where element i contains coefficients of T_i(r).

    Notes
    -----
    Implements the Chebyshev recurrence relation:
    T_0 = 1, T_1 = r, T_n = 2*r*T_{n-1} - T_{n-2}

    Adapted for Cartesian coordinates as:
    T_n = ((2n-1)/n) * x * T_{n-1} - ((n-1)/n) * (x^2 + y^2 + z^2) * T_{n-2}

    Used for inverse distance expansions around collinear equilibrium points
    (L1, L2, L3) in the CR3BP where the gravitational potential has the form
    U = -sum_{n>=2} c_n * T_n(r).

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_R_polynomials`
        Auxiliary polynomials for Lindstedt-Poincare formulation.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_potential_U`
        Gravitational potential construction using Chebyshev polynomials.

    References
    ----------
    Szebehely, V. (1967). Theory of Orbits. Academic Press, Chapter 7.
    """
    poly_T_list_of_polys = List()
    for _ in range(max_deg + 1):
        poly_T_list_of_polys.append(_polynomial_zero_list(max_deg, psi_table))

    if max_deg >= 0 and len(poly_T_list_of_polys[0]) > 0 and len(poly_T_list_of_polys[0][0]) > 0:
        poly_T_list_of_polys[0][0][0] = 1.0
    if max_deg >= 1:
        poly_T_list_of_polys[1] = poly_x

    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        term1_mult = _polynomial_multiply(poly_x, poly_T_list_of_polys[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1 = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term1, term1_mult, a)

        poly_x_sq = _polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_y_sq = _polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_z_sq = _polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)

        poly_sum_sq = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(poly_sum_sq, poly_x_sq, 1.0)
        _polynomial_add_inplace(poly_sum_sq, poly_y_sq, 1.0)
        _polynomial_add_inplace(poly_sum_sq, poly_z_sq, 1.0)

        term2_mult = _polynomial_multiply(poly_sum_sq, poly_T_list_of_polys[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term2 = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term2, term2_mult, -b)

        poly_Tn = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(poly_Tn, term1, 1.0)
        _polynomial_add_inplace(poly_Tn, term2, 1.0)
        poly_T_list_of_polys[n] = poly_Tn
    return poly_T_list_of_polys


@njit(fastmath=FASTMATH, cache=False)
def _build_R_polynomials(poly_x, poly_y, poly_z, poly_T: types.ListType, max_deg: int, psi_table, clmo_table, encode_dict_list) -> types.ListType:
    """Generate auxiliary R_n polynomials for Lindstedt-Poincare formulation.

    Compute the sequence R_n required for the Lindstedt-Poincare right-hand
    side polynomials in collinear point normal form computations.

    Parameters
    ----------
    poly_x, poly_y, poly_z : List[ndarray]
        Polynomial representations of coordinates x, y, z in nondimensional
        distance units.
    poly_T : List[List[ndarray]]
        Chebyshev polynomials from :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`.
    max_deg : int
        Maximum polynomial degree for R_n computation.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo_table : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    encode_dict_list : List[dict]
        Lookup tables mapping multi-indices to coefficient positions.

    Returns
    -------
    List[List[ndarray]]
        Auxiliary polynomials R_0, R_1, ..., R_max_deg.

    Notes
    -----
    Implements the recurrence relation:
    R_0 = -1
    R_1 = -3*x
    R_n = ((2n+3)/(n+2)) * x * R_{n-1} - ((2n+2)/(n+2)) * T_n - ((n+1)/(n+2)) * rho^2 * R_{n-2}

    where rho^2 = x^2 + y^2 + z^2.

    These polynomials appear in the Lindstedt-Poincare method for constructing
    periodic solutions around collinear equilibrium points in the CR3BP.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`
        Chebyshev polynomials used in this computation.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_lindstedt_poincare_rhs_polynomials`
        Final right-hand side construction using R_n polynomials.

    References
    ----------
    Jorba, A., Masdemont, J. (1999). Dynamics in the center manifold of the
    collinear points. Physica D, 132(1-2), 189-213.
    """
    poly_R_list_of_polys = List()
    for _ in range(max_deg + 1):
        poly_R_list_of_polys.append(_polynomial_zero_list(max_deg, psi_table))

    if max_deg >= 0:
        # R_0 = -1
        if len(poly_R_list_of_polys[0]) > 0 and len(poly_R_list_of_polys[0][0]) > 0:
            poly_R_list_of_polys[0][0][0] = -1.0
    
    if max_deg >= 1:
        # R_1 = -3x
        r1_poly = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(r1_poly, poly_x, -3.0)
        poly_R_list_of_polys[1] = r1_poly

    # Pre-calculate x^2, y^2, z^2, and x^2 + y^2 + z^2 as they are used in the loop
    poly_x_sq = None # Represents x^2
    poly_y_sq = None # Represents y^2
    poly_z_sq = None # Represents z^2
    poly_rho_sq = None # Represents x^2 + y^2 + z^2

    if max_deg >=2: # Only needed if the loop runs
        poly_x_sq = _polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_y_sq = _polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
        poly_z_sq = _polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
        
        poly_rho_sq = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(poly_rho_sq, poly_x_sq, 1.0)
        _polynomial_add_inplace(poly_rho_sq, poly_y_sq, 1.0)
        _polynomial_add_inplace(poly_rho_sq, poly_z_sq, 1.0)

    for n in range(2, max_deg + 1):
        n_ = float(n)
        
        coeff1 = (2.0 * n_ + 3.0) / (n_ + 2.0)
        coeff2 = (2.0 * n_ + 2.0) / (n_ + 2.0)
        coeff3 = (n_ + 1.0) / (n_ + 2.0)

        # Term 1: coeff1 * x * R_{n-1}
        term1_mult_x_Rnm1 = _polynomial_multiply(poly_x, poly_R_list_of_polys[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term1_poly = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term1_poly, term1_mult_x_Rnm1, coeff1)

        # Term 2: -coeff2 * T_n
        term2_poly = _polynomial_zero_list(max_deg, psi_table)
        # poly_T[n] is T_n
        _polynomial_add_inplace(term2_poly, poly_T[n], -coeff2)
        
        # Term 3: -coeff3 * (x^2 + y^2 + z^2) * R_{n-2}
        # poly_rho_sq is already computed if needed
        term3_mult_rhosq_Rnm2 = _polynomial_multiply(poly_rho_sq, poly_R_list_of_polys[n - 2], max_deg, psi_table, clmo_table, encode_dict_list)
        term3_poly = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term3_poly, term3_mult_rhosq_Rnm2, -coeff3)
        
        # Combine terms for R_n
        poly_Rn = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(poly_Rn, term1_poly, 1.0)
        _polynomial_add_inplace(poly_Rn, term2_poly, 1.0)
        _polynomial_add_inplace(poly_Rn, term3_poly, 1.0)
        poly_R_list_of_polys[n] = poly_Rn
        
    return poly_R_list_of_polys


def _build_potential_U(poly_T, point, max_deg: int, psi_table) -> List[np.ndarray]:
    """Assemble gravitational potential expansion for collinear points.

    Construct the effective potential U = -sum_{n>=2} c_n * T_n(r) using
    Chebyshev polynomial expansions and libration point coefficients.

    Parameters
    ----------
    poly_T : List[List[ndarray]]
        Chebyshev polynomials T_n from :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`.
    point : object
        Libration point object with method cn(k) returning the potential
        coefficient c_k (dimensionless).
    max_deg : int
        Maximum polynomial degree for potential truncation.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.

    Returns
    -------
    List[ndarray]
        Polynomial representation of the gravitational potential U in
        nondimensional energy units.

    Notes
    -----
    The gravitational potential around collinear equilibrium points has the
    series expansion U = -sum_{n>=2} c_n * T_n(r) where T_n are Chebyshev
    polynomials and c_n are coefficients determined by the mass parameter
    and equilibrium point location.

    The expansion starts from n=2 since the linear terms (n=0,1) are absorbed
    into the equilibrium point definition and coordinate transformation.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`
        Chebyshev polynomials used in this expansion.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_collinear`
        Complete Hamiltonian construction using this potential.

    References
    ----------
    Szebehely, V. (1967). Theory of Orbits. Academic Press, Section 7.2.
    """
    poly_U = _polynomial_zero_list(max_deg, psi_table)
    for n in range(2, max_deg + 1):
        _polynomial_add_inplace(poly_U, poly_T[n], -point.dynamics.cn(n))
    return poly_U


def _build_kinetic_energy_terms(poly_px, poly_py, poly_pz, max_deg: int, psi_table, clmo_table, encode_dict_list) -> List[np.ndarray]:
    """Build kinetic energy polynomial T = (1/2) * (px^2 + py^2 + pz^2).

    Construct the kinetic energy contribution to the Hamiltonian using
    canonical momentum polynomials in nondimensional CR3BP units.

    Parameters
    ----------
    poly_px, poly_py, poly_pz : List[ndarray]
        Polynomial representations of canonical momenta px, py, pz in
        nondimensional momentum units.
    max_deg : int
        Maximum polynomial degree for kinetic energy computation.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo_table : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    encode_dict_list : List[dict]
        Lookup tables mapping multi-indices to coefficient positions.

    Returns
    -------
    List[ndarray]
        Polynomial representation of kinetic energy T in nondimensional
        energy units.

    Notes
    -----
    The kinetic energy in the rotating frame has the standard form
    T = (1/2) * (px^2 + py^2 + pz^2) where px, py, pz are the canonical
    momenta conjugate to the position coordinates x, y, z.

    In nondimensional CR3BP units, energy is normalized by mu * n^2 * a^2
    where mu is the total mass, n is the mean motion, and a is the
    primary-secondary separation.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_rotational_terms`
        Coriolis terms that couple with kinetic energy in rotating frame.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_collinear`
        Complete Hamiltonian assembly including kinetic energy.
    """
    poly_kinetic = _polynomial_zero_list(max_deg, psi_table)
    for poly_momentum in (poly_px, poly_py, poly_pz):
        term = _polynomial_multiply(poly_momentum, poly_momentum, max_deg, psi_table, clmo_table, encode_dict_list)
        _polynomial_add_inplace(poly_kinetic, term, 0.5)
    return poly_kinetic


def _build_rotational_terms(poly_x, poly_y, poly_px, poly_py, max_deg: int, psi_table, clmo_table, encode_dict_list) -> List[np.ndarray]:
    """Construct Coriolis (rotational) terms C = y*px - x*py for rotating frame.

    Build the Coriolis contribution to the Hamiltonian that arises from the
    transformation to the rotating synodic reference frame in the CR3BP.

    Parameters
    ----------
    poly_x, poly_y : List[ndarray]
        Polynomial representations of position coordinates x, y in
        nondimensional distance units.
    poly_px, poly_py : List[ndarray]
        Polynomial representations of canonical momenta px, py in
        nondimensional momentum units.
    max_deg : int
        Maximum polynomial degree for Coriolis term computation.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo_table : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    encode_dict_list : List[dict]
        Lookup tables mapping multi-indices to coefficient positions.

    Returns
    -------
    List[ndarray]
        Polynomial representation of Coriolis terms C in nondimensional
        energy units.

    Notes
    -----
    The Coriolis terms C = y*px - x*py arise from the transformation to
    the rotating synodic frame where the two primaries remain fixed.
    These terms couple position and momentum and are essential for
    capturing the dynamics in the rotating reference frame.

    The Coriolis terms vanish in the inertial frame but become important
    in the rotating frame where they represent fictitious forces due to
    the frame rotation.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_kinetic_energy_terms`
        Kinetic energy terms that couple with Coriolis terms.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_collinear`
        Complete Hamiltonian including Coriolis terms.

    References
    ----------
    Szebehely, V. (1967). Theory of Orbits. Academic Press, Section 2.3.
    """
    poly_rot = _polynomial_zero_list(max_deg, psi_table)
    
    term_ypx = _polynomial_multiply(poly_y, poly_px, max_deg, psi_table, clmo_table, encode_dict_list)
    _polynomial_add_inplace(poly_rot, term_ypx, 1.0)

    term_xpy = _polynomial_multiply(poly_x, poly_py, max_deg, psi_table, clmo_table, encode_dict_list)
    _polynomial_add_inplace(poly_rot, term_xpy, -1.0)
    
    return poly_rot


def _build_physical_hamiltonian_collinear(point, max_deg: int) -> List[np.ndarray]:
    """Build complete rotating-frame Hamiltonian H = T + U + C for collinear points.

    Construct the full CR3BP Hamiltonian around collinear equilibrium points
    by combining kinetic energy, gravitational potential, and Coriolis terms.

    Parameters
    ----------
    point : object
        Libration point object with method cn(k) returning the potential
        coefficient c_k (dimensionless) for the gravitational expansion.
    max_deg : int
        Maximum polynomial degree for Hamiltonian truncation.

    Returns
    -------
    List[ndarray]
        Complete Hamiltonian polynomial coefficients up to max_deg in
        nondimensional energy units.

    Notes
    -----
    The rotating-frame Hamiltonian has the form:
    H = T + U + C = (1/2)*(px^2 + py^2 + pz^2) + U(x,y,z) + (y*px - x*py)

    where:
    - T: kinetic energy in canonical coordinates
    - U: gravitational potential expanded in Chebyshev polynomials
    - C: Coriolis terms from rotating frame transformation

    The constant term (equilibrium potential energy) is removed since we
    work with perturbations around the equilibrium point.

    All coordinates are in nondimensional CR3BP units:
    - Positions: primary-secondary separation = 1
    - Momenta: normalized by sqrt(mu * n^2 * a^3)
    - Energy: normalized by mu * n^2 * a^2

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_kinetic_energy_terms`
        Kinetic energy polynomial construction.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_potential_U`
        Gravitational potential using Chebyshev expansion.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_rotational_terms`
        Coriolis terms for rotating frame.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_triangular`
        Hamiltonian construction for triangular points.

    Examples
    --------
    >>> # Construct Hamiltonian for L1 point with degree 6 truncation
    >>> H = _build_physical_hamiltonian_collinear(l1_point, max_deg=6)  # doctest: +SKIP
    """
    psi_table, clmo_table = _init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    poly_H = _polynomial_zero_list(max_deg, psi_table)

    poly_x, poly_y, poly_z, poly_px, poly_py, poly_pz = [
        _polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list) for i in range(6)
    ]

    poly_kinetic = _build_kinetic_energy_terms(poly_px, poly_py, poly_pz, max_deg, psi_table, clmo_table, encode_dict_list)
    _polynomial_add_inplace(poly_H, poly_kinetic, 1.0)

    poly_rot = _build_rotational_terms(poly_x, poly_y, poly_px, poly_py, max_deg, psi_table, clmo_table, encode_dict_list)
    _polynomial_add_inplace(poly_H, poly_rot, 1.0)

    poly_T = _build_T_polynomials(poly_x, poly_y, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
    
    poly_U = _build_potential_U(poly_T, point, max_deg, psi_table)

    _polynomial_add_inplace(poly_H, poly_U, 1.0)

    # Remove the constant term (equilibrium potential energy) 
    if len(poly_H) > 0 and len(poly_H[0]) > 0:
        poly_H[0][0] = 0.0

    return poly_H

@njit(fastmath=FASTMATH, cache=False)
def _build_A_polynomials(poly_x, poly_y, poly_z, d_x: float, d_y: float, max_deg: int, psi_table, clmo_table, encode_dict_list) -> types.ListType:
    """Generate Legendre-type polynomials A_n for triangular point expansions.

    Compute the sequence A_n used for inverse distance expansions around
    triangular equilibrium points (L4/L5) in the CR3BP using a Legendre-type
    recurrence relation.

    Parameters
    ----------
    poly_x, poly_y, poly_z : List[ndarray]
        Polynomial representations of coordinates x, y, z in nondimensional
        distance units.
    d_x, d_y : float
        Components of primary offset vector in nondimensional distance units.
        For L4/L5, these represent the displacement from the equilibrium to
        the primary masses.
    max_deg : int
        Maximum polynomial degree for A_n computation.
    psi_table : ndarray
        Combinatorial index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    clmo_table : List[ndarray]
        Packed multi-index table from :func:`~hiten.algorithms.polynomial.base._init_index_tables`.
    encode_dict_list : List[dict]
        Lookup tables mapping multi-indices to coefficient positions.

    Returns
    -------
    List[List[ndarray]]
        Legendre-type polynomials A_0, A_1, ..., A_max_deg for inverse
        distance expansions.

    Notes
    -----
    Implements the recurrence relation:
    A_0 = 1
    A_1 = d_x*x + d_y*y
    A_{n+1} = ((2n+1)/(n+1)) * (d_dot_r) * A_n - (n/(n+1)) * rho^2 * A_{n-1}

    where d_dot_r = d_x*x + d_y*y and rho^2 = x^2 + y^2 + z^2.

    These polynomials are used to expand the inverse distances 1/r_PS and 1/r_PJ
    in the triangular point Hamiltonian, where PS and PJ denote distances to
    the primary and secondary masses respectively.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_triangular`
        Triangular point Hamiltonian using these polynomials.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`
        Analogous Chebyshev polynomials for collinear points.

    References
    ----------
    Gomez, G., et al. (2001). Dynamics and Mission Design Near Libration Points.
    World Scientific, Chapter 3.
    """
    poly_A_list = List()
    for _ in range(max_deg + 1):
        poly_A_list.append(_polynomial_zero_list(max_deg, psi_table))

    # A_0 = 1
    if max_deg >= 0 and len(poly_A_list[0]) > 0 and len(poly_A_list[0][0]) > 0:
        poly_A_list[0][0][0] = 1.0

    # A_1 = d_x * x + d_y * y  (note: z component of d is zero in planar primaries)
    if max_deg >= 1:
        poly_A1 = _polynomial_zero_list(max_deg, psi_table)
        if d_x != 0.0:
            _polynomial_add_inplace(poly_A1, poly_x, d_x)
        if d_y != 0.0:
            _polynomial_add_inplace(poly_A1, poly_y, d_y)
        poly_A_list[1] = poly_A1

    if max_deg < 2:
        return poly_A_list

    # Pre-compute rho^2 = x^2 + y^2 + z^2 and dot = d_x x + d_y y
    poly_x_sq = _polynomial_multiply(poly_x, poly_x, max_deg, psi_table, clmo_table, encode_dict_list)
    poly_y_sq = _polynomial_multiply(poly_y, poly_y, max_deg, psi_table, clmo_table, encode_dict_list)
    poly_z_sq = _polynomial_multiply(poly_z, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)

    poly_rho2 = _polynomial_zero_list(max_deg, psi_table)
    _polynomial_add_inplace(poly_rho2, poly_x_sq, 1.0)
    _polynomial_add_inplace(poly_rho2, poly_y_sq, 1.0)
    _polynomial_add_inplace(poly_rho2, poly_z_sq, 1.0)

    poly_dot = _polynomial_zero_list(max_deg, psi_table)
    if d_x != 0.0:
        _polynomial_add_inplace(poly_dot, poly_x, d_x)
    if d_y != 0.0:
        _polynomial_add_inplace(poly_dot, poly_y, d_y)

    # Recurrence: A_{n+1} = ((2n+1)/(n+1)) (dot) A_n - (n/(n+1)) rho2 A_{n-1}
    for n in range(1, max_deg):  # n corresponds to current highest index (A_n)
        n_ = float(n)
        coeff1 = (2.0 * n_ + 1.0) / (n_ + 1.0)
        coeff2 = n_ / (n_ + 1.0)

        # term1 = coeff1 * dot * A_n
        term1_mult = _polynomial_multiply(poly_dot, poly_A_list[n], max_deg, psi_table, clmo_table, encode_dict_list)
        term1 = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term1, term1_mult, coeff1)

        # term2 = -coeff2 * rho2 * A_{n-1}
        term2_mult = _polynomial_multiply(poly_rho2, poly_A_list[n - 1], max_deg, psi_table, clmo_table, encode_dict_list)
        term2 = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(term2, term2_mult, -coeff2)

        # Combine to obtain A_{n+1}
        poly_An1 = _polynomial_zero_list(max_deg, psi_table)
        _polynomial_add_inplace(poly_An1, term1, 1.0)
        _polynomial_add_inplace(poly_An1, term2, 1.0)

        poly_A_list[n + 1] = poly_An1

    return poly_A_list


def _build_physical_hamiltonian_triangular(point, max_deg: int) -> List[np.ndarray]:
    """Build rotating-frame Hamiltonian for triangular points (L4/L5).

    Construct the complete CR3BP Hamiltonian around triangular equilibria
    using Legendre-type polynomial expansions for the inverse distance terms
    to the primary masses.

    Parameters
    ----------
    point : object
        Triangular libration point object (L4 or L5) with attributes mu (mass
        parameter) and sign (+1 for L4, -1 for L5).
    max_deg : int
        Maximum polynomial degree for Hamiltonian truncation.

    Returns
    -------
    List[ndarray]
        Complete triangular point Hamiltonian coefficients up to max_deg in
        nondimensional energy units.

    Notes
    -----
    The triangular point Hamiltonian has the form:
    H = (1/2)*(px^2 + py^2 + pz^2) + y*px - x*py + ((1/2) - mu)*x + s*sqrt(3)/2*y - (1-mu)/r_PS - mu/r_PJ

    where:
    - (px, py, pz): canonical momenta in nondimensional momentum units
    - (x, y, z): local coordinates around the triangular equilibrium
    - mu: mass parameter mu = m2/(m1 + m2)
    - s: sign parameter (+1 for L4, -1 for L5)
    - r_PS, r_PJ: distances to primary and secondary masses

    The distances to the primaries in local coordinates are:
    r_PS^2 = (x - x_S)^2 + (y - y_S)^2 + z^2
    r_PJ^2 = (x - x_J)^2 + (y - y_J)^2 + z^2

    with offset vectors:
    (x_S, y_S) = (+1/2, s*sqrt(3)/2)
    (x_J, y_J) = (-1/2, s*sqrt(3)/2)

    At equilibrium, both distances equal 1, allowing expansion of inverse
    distances using Legendre-type polynomials via the binomial series:
    1/r = (1 + u)^(-1/2) = sum_{k>=0} C(-1/2,k) * u^k
    where u = r^2 - 1 and C(alpha,k) are binomial coefficients.

    The constant term (equilibrium potential energy = -1) is removed.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_A_polynomials`
        Legendre-type polynomials for inverse distance expansions.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_physical_hamiltonian_collinear`
        Analogous Hamiltonian for collinear points.

    References
    ----------
    Gomez, G., Llibre, J., Martinez, R., Simo, C. (2001). Dynamics and Mission
    Design Near Libration Points. World Scientific, Equation 3.57.
    """
    psi_table, clmo_table = _init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    poly_x, poly_y, poly_z, poly_px, poly_py, poly_pz = [
        _polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list)
        for i in range(6)
    ]

    poly_H = _polynomial_zero_list(max_deg, psi_table)

    poly_T = _build_kinetic_energy_terms(
        poly_px, poly_py, poly_pz,
        max_deg, psi_table, clmo_table, encode_dict_list,
    )
    _polynomial_add_inplace(poly_H, poly_T, 1.0)

    poly_C = _build_rotational_terms(
        poly_x, poly_y, poly_px, poly_py,
        max_deg, psi_table, clmo_table, encode_dict_list,
    )
    _polynomial_add_inplace(poly_H, poly_C, 1.0)

    mu = float(point.mu)
    sgn = float(point.dynamics.sign)  # +1 for L4, -1 for L5

    poly_linear = _polynomial_zero_list(max_deg, psi_table)
    _polynomial_add_inplace(poly_linear, poly_x, 0.5 - mu)
    _polynomial_add_inplace(poly_linear, poly_y, - sgn * np.sqrt(3) / 2.0)
    _polynomial_add_inplace(poly_H, poly_linear, 1.0)

    # Linear dot products with the primary offsets
    d_Sx, d_Sy = 0.5, sgn * np.sqrt(3) / 2.0    # Primary at negative x
    d_Jx, d_Jy = -0.5, sgn * np.sqrt(3) / 2.0     # Secondary at positive x

    # Construct inverse distances via Legendre-type homogeneous polynomials
    poly_A_S = _build_A_polynomials(
        poly_x, poly_y, poly_z,
        d_Sx, d_Sy,
        max_deg, psi_table, clmo_table, encode_dict_list,
    )
    poly_A_J = _build_A_polynomials(
        poly_x, poly_y, poly_z,
        d_Jx, d_Jy,
        max_deg, psi_table, clmo_table, encode_dict_list,
    )

    # The expansion of 1/r_PS is A_0 + A_1 + A_2 + ... 
    # At equilibrium, A_0 = 1, and we want to subtract the constant part
    poly_inv_r_S = _polynomial_zero_list(max_deg, psi_table)
    poly_inv_r_J = _polynomial_zero_list(max_deg, psi_table)

    # Add all A_n terms for each primary
    for n in range(0, min(len(poly_A_S), max_deg + 1)):
        _polynomial_add_inplace(poly_inv_r_S, poly_A_S[n], 1.0)
        
    for n in range(0, min(len(poly_A_J), max_deg + 1)):
        _polynomial_add_inplace(poly_inv_r_J, poly_A_J[n], 1.0)

    # Construct the potential: -(1-mu)/r_PS - mu/r_PJ
    poly_U = _polynomial_zero_list(max_deg, psi_table)
    _polynomial_add_inplace(poly_U, poly_inv_r_S, -(1.0 - mu))
    _polynomial_add_inplace(poly_U, poly_inv_r_J, -mu)
    _polynomial_add_inplace(poly_H, poly_U, 1.0)

    # Remove the constant term (equilibrium potential energy = -1)
    if len(poly_H) > 0 and len(poly_H[0]) > 0:
        poly_H[0][0] = 0.0

    return poly_H


def _build_lindstedt_poincare_rhs_polynomials(point, max_deg: int) -> Tuple[List, List, List]:
    """Compute right-hand side polynomials for Lindstedt-Poincare method.

    Generate the polynomial right-hand sides for the x, y, z equations in
    the first iteration of the Lindstedt-Poincare method for constructing
    periodic solutions around collinear equilibrium points.

    Parameters
    ----------
    point : object
        Libration point object with method cn(k) returning the potential
        coefficient c_k (dimensionless) for the gravitational expansion.
    max_deg : int
        Maximum polynomial degree for right-hand side truncation.

    Returns
    -------
    tuple of (List[ndarray], List[ndarray], List[ndarray])
        Polynomial coefficients for the x-, y-, and z-equation right-hand
        sides respectively, in nondimensional acceleration units.

    Notes
    -----
    The Lindstedt-Poincare method constructs periodic solutions by expanding
    both the solution and the frequency in powers of a small parameter.
    The right-hand sides have the form:

    RHS_x = sum_{n>=2} c_{n+1} * (n+1) * T_n(r)
    RHS_y = y * sum_{n>=2} c_{n+1} * R_{n-1}(r)
    RHS_z = z * sum_{n>=2} c_{n+1} * R_{n-1}(r)

    where T_n are Chebyshev polynomials, R_n are auxiliary polynomials,
    and c_k are the potential coefficients from the libration point.

    These polynomials appear in the differential equations for the first-order
    corrections to the linear center manifold dynamics around collinear points.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_T_polynomials`
        Chebyshev polynomials used in x-equation.
    :func:`~hiten.algorithms.hamiltonian.hamiltonian._build_R_polynomials`
        Auxiliary polynomials used in y,z-equations.

    Examples
    --------
    >>> # Generate RHS polynomials for L1 point with degree 6
    >>> rhs_x, rhs_y, rhs_z = _build_lindstedt_poincare_rhs_polynomials(l1_point, 6)  # doctest: +SKIP

    References
    ----------
    Jorba, A., Masdemont, J. (1999). Dynamics in the center manifold of the
    collinear points. Physica D, 132(1-2), 189-213.
    """
    psi_table, clmo_table = _init_index_tables(max_deg)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_table)

    poly_x, poly_y, poly_z = [
        _polynomial_variable(i, max_deg, psi_table, clmo_table, encode_dict_list) for i in range(3)
    ]

    poly_T_list = _build_T_polynomials(poly_x, poly_y, poly_z, max_deg, psi_table, clmo_table, encode_dict_list)
    poly_R_list = _build_R_polynomials(poly_x, poly_y, poly_z, poly_T_list, max_deg, psi_table, clmo_table, encode_dict_list)

    rhs_x_poly = _polynomial_zero_list(max_deg, psi_table)

    sum_term_for_y_z_eqs = _polynomial_zero_list(max_deg, psi_table)

    for n in range(2, max_deg + 1):
        cn_plus_1 = point.dynamics.cn(n + 1)
        coeff = cn_plus_1 * float(n + 1)
        _polynomial_add_inplace(rhs_x_poly, poly_T_list[n], coeff)

    for n in range(2, max_deg + 1):
        cn_plus_1 = point.dynamics.cn(n + 1)
        if (n - 1) < len(poly_R_list):
            _polynomial_add_inplace(sum_term_for_y_z_eqs, poly_R_list[n - 1], cn_plus_1)

    rhs_y_poly = _polynomial_multiply(poly_y, sum_term_for_y_z_eqs, max_deg, psi_table, clmo_table, encode_dict_list)

    rhs_z_poly = _polynomial_multiply(poly_z, sum_term_for_y_z_eqs, max_deg, psi_table, clmo_table, encode_dict_list)
    
    return rhs_x_poly, rhs_y_poly, rhs_z_poly
