"""Provide a Hamiltonian conversion registry and transformation wrappers.

This module provides a comprehensive conversion system for transforming polynomial
Hamiltonians between different coordinate systems and normal form representations
in the Circular Restricted Three-Body Problem (CR3BP). It implements a registry-based
architecture for automatic conversions between physical, modal, complex, and normal
form representations.

Key Features
------------
- Registry-based conversion system for automatic coordinate transformations
- Wrapper functions for all major Hamiltonian transformations
- Support for both collinear and triangular libration points
- Partial and full normal form computations via Lie series methods
- Center manifold restrictions for dimensional reduction
- Generating function extraction from normal form computations

The conversion system enables seamless transformation between coordinate systems:
Physical -> Modal -> Complex -> Normal Forms -> Center Manifold

All transformations preserve the Hamiltonian structure and maintain appropriate
scaling in nondimensional CR3BP units where:
- Positions: primary-secondary separation = 1
- Momenta: conjugate to position coordinates
- Energy: normalized by characteristic energy scale
- Time: normalized by orbital period / (2*pi)

Coordinate Representations
--------------------------
- **Physical**: Local coordinates centered at equilibrium point
- **Real Modal**: Coordinates aligned with linear stability eigenvectors
- **Complex Modal**: Complexified coordinates for elliptic directions
- **Partial Normal**: Partial normal form eliminating specific resonances
- **Full Normal**: Complete normal form eliminating all non-resonant terms
- **Center Manifold**: Restricted dynamics on center-stable subspace

See Also
--------
:mod:`~hiten.algorithms.hamiltonian.center`
    Partial normal form computations for center manifold analysis.
:mod:`~hiten.algorithms.hamiltonian.normal`
    Full normal form computations for complete dynamical reduction.
:mod:`~hiten.algorithms.hamiltonian.transforms`
    Coordinate transformation utilities used by these wrappers.
:mod:`~hiten.system.hamiltonians.base`
    Base Hamiltonian classes and conversion registry infrastructure.

References
----------
Jorba, A. (1999). A methodology for the numerical computation of normal forms,
centre manifolds and first integrals of Hamiltonian systems. Experimental
Mathematics, 8(2), 155-195.

Meyer, K.R., Hall, G.R. (1992). Introduction to Hamiltonian Dynamical Systems
and the N-Body Problem. Springer-Verlag, Chapters 4-5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hiten.algorithms.hamiltonian.center._lie import \
    _lie_transform as _lie_transform_partial
from hiten.algorithms.hamiltonian.normal._lie import \
    _lie_transform as _lie_transform_full
from hiten.algorithms.hamiltonian.transforms import (
    _polylocal2realmodal, _polyrealmodal2local,
    _restrict_poly_to_center_manifold, _substitute_complex, _substitute_real)
from hiten.algorithms.types.services import get_hamiltonian_services
from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction

if TYPE_CHECKING:
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint


def register_conversion(src_name: str, dst: "type[Hamiltonian] | str", 
                       required_context: list = None,
                       default_params: dict = None):
    """Decorator to register Hamiltonian coordinate system conversions.

    Register a conversion function in the global conversion registry, enabling
    automatic transformation between different Hamiltonian representations in
    the CR3BP normal form pipeline.

    Parameters
    ----------
    src_name : str
        Name identifier of the source Hamiltonian representation (e.g., "physical",
        "real_modal", "complex_modal").
    dst : type[Hamiltonian] or str
        Either the destination Hamiltonian class or its name string identifier.
        Using strings allows registration before class definition.
    required_context : list, optional
        List of required context keys that must be provided during conversion.
        Common requirements include ["point"] for libration point information.
    default_params : dict, optional
        Default parameter values for the conversion function (e.g., tolerances,
        numerical settings).

    Returns
    -------
    callable
        Decorator function that registers the conversion and returns the
        original function unchanged.

    Notes
    -----
    The conversion registry enables automatic transformation between coordinate
    systems by maintaining a lookup table of (source, destination) to function
    mappings. This supports the normal form computation pipeline where
    Hamiltonians must be transformed through multiple coordinate systems.

    Registered functions should follow the signature:
    conversion_func(ham: Hamiltonian, kwargs) returns :class:`~hiten.system.hamiltonian.Hamiltonian` or tuple

    The kwargs typically include context (like libration points) and numerical
    parameters (like tolerances). Some conversions return tuples containing
    both the transformed Hamiltonian and auxiliary data like generating functions.

    See Also
    --------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Base Hamiltonian class used in conversions.
    :func:`~hiten.system.hamiltonians.base._CONVERSION_REGISTRY`
        Global registry storing conversion functions.

    Examples
    --------
    >>> @register_conversion("physical", "real_modal", 
    ...                     required_context=["point"],
    ...                     default_params={"tol": 1e-12})
    ... def physical_to_modal(ham: Hamiltonian, kwargs) -> Hamiltonian:
    ...     point = kwargs["point"]
    ...     tol = kwargs.get("tol", 1e-12)
    ...     # Transform polynomial using point's eigenvectors
    ...     return transformed_hamiltonian
    """

    dst_name: str
    if isinstance(dst, str):
        dst_name = dst
    else:
        dst_name = dst.name

    registry = get_hamiltonian_services()

    def _decorator(func):
        registry._CONVERSION_REGISTRY[(src_name, dst_name)] = (
            func, 
            required_context or [], 
            default_params or {}
        )
        return func

    return _decorator


@register_conversion("physical", "real_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _physical_to_real_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from physical to real modal coordinates.

    Convert a polynomial Hamiltonian from local physical coordinates centered
    at an equilibrium point to real modal coordinates aligned with the linear
    stability eigenvectors of the equilibrium.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in physical coordinates, with polynomial coefficients
        in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point providing the modal transformation matrix.
        - tol : float, default 1e-12
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in real modal coordinates with name "real_modal".

    Notes
    -----
    The transformation uses the eigenvector matrix from the libration point's
    linearization to convert from local coordinates to modal coordinates where
    each coordinate pair corresponds to a specific eigenvalue/eigenvector pair
    of the linearized dynamics.

    This transformation is the first step in the normal form pipeline, enabling
    subsequent complexification and normal form reductions by working in the
    natural coordinate system defined by the linear stability properties.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._polylocal2realmodal`
        Underlying transformation function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._real_modal_to_physical`
        Inverse transformation back to physical coordinates.
    """
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-12)
    new_poly = _polylocal2realmodal(point, ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="real_modal")


@register_conversion("real_modal", "physical", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _real_modal_to_physical(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from real modal to physical coordinates.

    Convert a polynomial Hamiltonian from real modal coordinates aligned with
    linear stability eigenvectors back to local physical coordinates centered
    at the equilibrium point.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in real modal coordinates, with polynomial coefficients
        in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point providing the inverse modal transformation matrix.
        - tol : float, default 1e-12
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in physical coordinates with name "physical".

    Notes
    -----
    This transformation is the inverse of the physical to real modal conversion,
    using the inverse of the eigenvector matrix from the libration point's
    linearization to convert back from modal coordinates to local coordinates.

    The transformation restores the original physical coordinate system where
    coordinates represent displacements from the equilibrium point in the
    standard CR3BP reference frame.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._polyrealmodal2local`
        Underlying transformation function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._physical_to_real_modal`
        Forward transformation to real modal coordinates.
    """
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-12)
    new_poly = _polyrealmodal2local(point, ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="physical")


@register_conversion("real_modal", "complex_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _real_modal_to_complex_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from real modal to complex modal coordinates.

    Convert a polynomial Hamiltonian from real modal coordinates to complex
    modal coordinates by complexifying elliptic coordinate pairs. This enables
    the use of complex normal form techniques for analyzing the dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in real modal coordinates, with polynomial coefficients
        in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to complexify.
        - tol : float, default 1e-12
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in complex modal coordinates with name "complex_modal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are complexified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are complexified since all directions are elliptic.

    The complexification process introduces complex variables that simplify
    the analysis of elliptic dynamics and enable the application of complex
    normal form theory. The transformation preserves the Hamiltonian structure
    while providing a more convenient representation for subsequent normal
    form computations.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Underlying complexification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._complex_modal_to_real_modal`
        Inverse transformation back to real modal coordinates.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-12)
    new_poly = _substitute_complex(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="complex_modal")


@register_conversion("complex_modal", "real_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _complex_modal_to_real_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from complex modal to real modal coordinates.

    Convert a polynomial Hamiltonian from complex modal coordinates back to
    real modal coordinates by realifying complexified elliptic coordinate pairs.
    This provides the inverse of the complexification process.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex modal coordinates, with polynomial coefficients
        in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to realify.
        - tol : float, default 1e-12
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in real modal coordinates with name "real_modal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are realified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are realified since all directions are elliptic.

    The realification process converts complex variables back to real variables
    while preserving the Hamiltonian structure. This transformation is typically
    used after complex normal form computations to return to a real representation
    suitable for physical interpretation or further real-coordinate analysis.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Underlying realification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._real_modal_to_complex_modal`
        Forward transformation to complex modal coordinates.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-12)
    new_poly = _substitute_real(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="real_modal")


@register_conversion("complex_modal", "complex_partial_normal", 
                    required_context=["point"],
                    default_params={"tol_lie": 1e-30})
def _complex_modal_to_complex_partial_normal(ham: Hamiltonian, **kwargs) -> tuple[Hamiltonian, "LieGeneratingFunction"]:
    """Transform Hamiltonian to partial normal form via Lie series method.

    Apply partial normal form transformation to eliminate non-resonant terms
    from a complex modal Hamiltonian, retaining only terms that contribute to
    the center manifold dynamics and specific resonances.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex modal coordinates, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point providing eigenvalue information for resonance analysis.
        - tol_lie : float, default 1e-30
            Numerical tolerance for Lie series computations and coefficient cleaning.

    Returns
    -------
    tuple of (:class:`~hiten.system.hamiltonian.Hamiltonian`, :class:`~hiten.system.hamiltonian.LieGeneratingFunction`)
        - Transformed Hamiltonian in partial normal form with name "complex_partial_normal"
        - Generating functions used in the transformation, containing both the
          total generating function and eliminated terms

    Notes
    -----
    The partial normal form eliminates terms that do not contribute to the
    center manifold dynamics while preserving resonant terms and the structure
    needed for center manifold analysis. This is achieved through a sequence
    of canonical transformations generated by polynomial generating functions.

    The Lie series method implements transformations of the form:
    H' = exp(L_G) H
    where L_G is the Lie operator associated with generating function G.

    The returned generating functions contain the complete transformation
    history and can be used for coordinate transformations or analysis of
    the eliminated dynamics.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.center._lie._lie_transform`
        Underlying partial normal form computation.
    :func:`~hiten.algorithms.hamiltonian.wrappers._complex_modal_to_complex_full_normal`
        Complete normal form transformation.
    :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
        Container for generating function data.
    """
    point = kwargs["point"]
    tol_lie = kwargs.get("tol_lie", 1e-30)
    # This returns (poly_trans, poly_G_total, poly_elim_total)
    new_poly, poly_G_total, poly_elim_total = _lie_transform_partial(point, ham.dynamics.poly_H, ham.dynamics.psi, ham.dynamics.clmo, ham.dynamics.degree, tol=tol_lie)
    
    new_ham = Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="complex_partial_normal")
    generating_functions = LieGeneratingFunction(poly_G=poly_G_total, poly_elim=poly_elim_total, degree=ham.dynamics.degree, ndof=ham.dynamics.ndof, name="generating_functions_partial")
    
    return new_ham, generating_functions


@register_conversion("complex_partial_normal", "real_partial_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_partial_normal_to_real_partial_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from complex partial normal to real partial normal form.

    Convert a polynomial Hamiltonian from complex partial normal form to real
    partial normal form by realifying complexified elliptic coordinate pairs.
    This provides a real representation of the partially normalized dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex partial normal form, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to realify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in real partial normal form with name "real_partial_normal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are realified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are realified since all directions are elliptic.

    The partial normal form retains only terms that contribute to center manifold
    dynamics while eliminating non-resonant terms. The realification process
    converts the complex representation back to real variables while preserving
    the normalized structure, making the dynamics more interpretable in physical
    terms.

    This transformation is typically used after complex partial normal form
    computations to obtain a real representation suitable for center manifold
    analysis or further real-coordinate processing.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Underlying realification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._real_partial_normal_to_complex_partial_normal`
        Inverse transformation back to complex partial normal form.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="real_partial_normal")


@register_conversion("real_partial_normal", "complex_partial_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _real_partial_normal_to_complex_partial_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from real partial normal to complex partial normal form.

    Convert a polynomial Hamiltonian from real partial normal form to complex
    partial normal form by complexifying elliptic coordinate pairs. This enables
    the use of complex analysis techniques on the partially normalized dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in real partial normal form, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to complexify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in complex partial normal form with name "complex_partial_normal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are complexified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are complexified since all directions are elliptic.

    The partial normal form retains only terms that contribute to center manifold
    dynamics while eliminating non-resonant terms. The complexification process
    converts real variables to complex variables while preserving the normalized
    structure, enabling the application of complex analysis techniques.

    This transformation is typically used when complex analysis methods are
    needed on the partially normalized dynamics, such as for further normal
    form computations or complex center manifold analysis.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Underlying complexification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._complex_partial_normal_to_real_partial_normal`
        Inverse transformation back to real partial normal form.
    """
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="complex_partial_normal")


@register_conversion("complex_partial_normal", "center_manifold_complex", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_partial_normal_to_center_manifold_complex(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Restrict Hamiltonian to center manifold by eliminating hyperbolic terms.

    Project a partial normal form Hamiltonian onto the center manifold by
    removing all terms that depend on hyperbolic variables, retaining only
    the dynamics within the center-stable/center-unstable subspace.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex partial normal form, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining the manifold structure and hyperbolic directions.
        - tol : float, default 1e-14
            Numerical tolerance for zeroing small coefficients during restriction.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Restricted Hamiltonian on center manifold with name "center_manifold_complex".

    Notes
    -----
    For collinear points, the first canonical pair corresponds to the hyperbolic
    direction, so all terms with non-zero exponents in these variables are
    eliminated. This reduces the 6D phase space to a 4D center manifold.

    For triangular points, all directions are elliptic (center-type), so the
    function returns the original Hamiltonian without restriction.

    This restriction is fundamental to center manifold theory, which enables
    dimensional reduction by focusing on neutrally stable directions while
    eliminating exponentially growing/decaying hyperbolic modes.

    The resulting center manifold Hamiltonian captures the long-term dynamics
    and is used for constructing periodic orbits, invariant tori, and other
    dynamical structures that persist in the full system.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._restrict_poly_to_center_manifold`
        Underlying center manifold restriction function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._center_manifold_complex_to_center_manifold_real`
        Conversion to real center manifold coordinates.

    References
    ----------
    Carr, J. (1981). Applications of Centre Manifold Theory. Springer-Verlag.
    """
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-14)
    new_poly = _restrict_poly_to_center_manifold(point, ham.dynamics.poly_H, ham.dynamics.clmo, tol=tol)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="center_manifold_complex")


@register_conversion("center_manifold_complex", "center_manifold_real", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _center_manifold_complex_to_center_manifold_real(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from complex center manifold to real center manifold.

    Convert a polynomial Hamiltonian from complex center manifold representation
    to real center manifold representation by realifying complexified elliptic
    coordinate pairs. This provides a real representation of the center manifold dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex center manifold representation, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to realify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in real center manifold representation with name "center_manifold_real".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are realified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are realified since all directions are elliptic.

    The center manifold Hamiltonian captures the long-term dynamics by eliminating
    hyperbolic terms and focusing on neutrally stable directions. The realification
    process converts complex variables back to real variables while preserving
    the center manifold structure, making the dynamics more interpretable in
    physical terms.

    This transformation is typically used after complex center manifold computations
    to obtain a real representation suitable for physical interpretation or
    further real-coordinate analysis of the center manifold dynamics.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Underlying realification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._center_manifold_real_to_center_manifold_complex`
        Inverse transformation back to complex center manifold representation.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="center_manifold_real")


@register_conversion("center_manifold_real", "center_manifold_complex", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _center_manifold_real_to_center_manifold_complex(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from real center manifold to complex center manifold.

    Convert a polynomial Hamiltonian from real center manifold representation
    to complex center manifold representation by complexifying elliptic coordinate
    pairs. This enables the use of complex analysis techniques on center manifold dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in real center manifold representation, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to complexify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in complex center manifold representation with name "center_manifold_complex".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are complexified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are complexified since all directions are elliptic.

    The center manifold Hamiltonian captures the long-term dynamics by eliminating
    hyperbolic terms and focusing on neutrally stable directions. The complexification
    process converts real variables to complex variables while preserving the
    center manifold structure, enabling the application of complex analysis techniques.

    This transformation is typically used when complex analysis methods are
    needed on the center manifold dynamics, such as for further normal form
    computations or complex center manifold analysis.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Underlying complexification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._center_manifold_complex_to_center_manifold_real`
        Inverse transformation back to real center manifold representation.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="center_manifold_complex")


@register_conversion("complex_modal", "complex_full_normal", 
                    required_context=["point"],
                    default_params={"tol_lie": 1e-30, "resonance_tol": 1e-14})
def _complex_modal_to_complex_full_normal(ham: Hamiltonian, **kwargs) -> tuple[Hamiltonian, "LieGeneratingFunction"]:
    """Transform Hamiltonian to full normal form via Lie series method.

    Apply complete normal form transformation to eliminate all non-resonant terms
    from a complex modal Hamiltonian, achieving the maximum possible simplification
    of the dynamics through canonical transformations.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex modal coordinates, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or 
            :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point providing eigenvalue information for resonance analysis.
        - tol_lie : float, default 1e-30
            Numerical tolerance for Lie series computations and coefficient cleaning.
        - resonance_tol : float, default 1e-14
            Tolerance for detecting resonance conditions between eigenvalues.

    Returns
    -------
    tuple of (:class:`~hiten.system.hamiltonian.Hamiltonian`, :class:`~hiten.system.hamiltonian.LieGeneratingFunction`)
        - Transformed Hamiltonian in full normal form with name "complex_full_normal"
        - Generating functions used in the transformation, containing both the
          total generating function and eliminated terms

    Notes
    -----
    The full normal form eliminates all non-resonant terms through a sequence
    of canonical transformations, achieving the maximum possible simplification
    of the Hamiltonian while preserving the essential dynamics. This is the
    most complete form of normalization possible.

    The Lie series method implements transformations of the form:
    H' = exp(L_G) H
    where L_G is the Lie operator associated with generating function G.

    Resonance conditions are detected based on the eigenvalues of the libration
    point, and only resonant terms are preserved in the final normal form.
    The resulting Hamiltonian contains only terms that cannot be eliminated
    due to resonance conditions.

    The returned generating functions contain the complete transformation
    history and can be used for coordinate transformations or analysis of
    the eliminated dynamics.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.normal._lie._lie_transform`
        Underlying full normal form computation.
    :func:`~hiten.algorithms.hamiltonian.wrappers._complex_modal_to_complex_partial_normal`
        Partial normal form transformation.
    :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
        Container for generating function data.
    """
    point = kwargs["point"]
    tol_lie = kwargs.get("tol_lie", 1e-30)
    resonance_tol = kwargs.get("resonance_tol", 1e-14)
    new_poly, poly_G_total, poly_elim_total = _lie_transform_full(point, ham.dynamics.poly_H, ham.dynamics.psi, ham.dynamics.clmo, ham.dynamics.degree, tol=tol_lie, resonance_tol=resonance_tol)
    
    new_ham = Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="complex_full_normal")
    generating_functions = LieGeneratingFunction(poly_G=poly_G_total, poly_elim=poly_elim_total, degree=ham.dynamics.degree, ndof=ham.dynamics.ndof, name="generating_functions_full")
    
    return new_ham, generating_functions


@register_conversion("complex_full_normal", "real_full_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_full_normal_to_real_full_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from complex full normal to real full normal form.

    Convert a polynomial Hamiltonian from complex full normal form to real
    full normal form by realifying complexified elliptic coordinate pairs.
    This provides a real representation of the completely normalized dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in complex full normal form, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or 
            :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to realify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in real full normal form with name "real_full_normal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are realified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are realified since all directions are elliptic.

    The full normal form represents the maximum possible simplification of the
    Hamiltonian through canonical transformations, eliminating all non-resonant
    terms. The realification process converts complex variables back to real
    variables while preserving the normalized structure, making the dynamics
    more interpretable in physical terms.

    This transformation is typically used after complex full normal form
    computations to obtain a real representation suitable for physical
    interpretation or further real-coordinate analysis of the completely
    normalized dynamics.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_real`
        Underlying realification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._real_full_normal_to_complex_full_normal`
        Inverse transformation back to complex full normal form.
    """
    point = kwargs["point"]
    from hiten.system.libration.collinear import CollinearPoint
    from hiten.system.libration.triangular import TriangularPoint
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="real_full_normal")


@register_conversion("real_full_normal", "complex_full_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _real_full_normal_to_complex_full_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    """Transform Hamiltonian from real full normal to complex full normal form.

    Convert a polynomial Hamiltonian from real full normal form to complex
    full normal form by complexifying elliptic coordinate pairs. This enables
    the use of complex analysis techniques on the completely normalized dynamics.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        Source Hamiltonian in real full normal form, with polynomial
        coefficients in nondimensional energy units.
    kwargs
        Conversion context and parameters:
        
        - point : :class:`~hiten.system.libration.collinear.CollinearPoint` or :class:`~hiten.system.libration.triangular.TriangularPoint`
            Libration point determining which coordinate pairs to complexify.
        - tol : float, default 1e-14
            Numerical tolerance for cleaning small coefficients.

    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        Transformed Hamiltonian in complex full normal form with name "complex_full_normal".

    Notes
    -----
    For collinear points, coordinate pairs (1, 2) are complexified, corresponding
    to the elliptic directions. For triangular points, all coordinate pairs
    (0, 1, 2) are complexified since all directions are elliptic.

    The full normal form represents the maximum possible simplification of the
    Hamiltonian through canonical transformations, eliminating all non-resonant
    terms. The complexification process converts real variables to complex
    variables while preserving the normalized structure, enabling the application
    of complex analysis techniques.

    This transformation is typically used when complex analysis methods are
    needed on the completely normalized dynamics, such as for further analysis
    or complex coordinate transformations.

    See Also
    --------
    :func:`~hiten.algorithms.hamiltonian.transforms._substitute_complex`
        Underlying complexification function.
    :func:`~hiten.algorithms.hamiltonian.wrappers._complex_full_normal_to_real_full_normal`
        Inverse transformation back to real full normal form.
    """
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.dynamics.poly_H, ham.dynamics.degree, ham.dynamics.psi, ham.dynamics.clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.dynamics.degree, ham.dynamics.ndof, name="complex_full_normal")