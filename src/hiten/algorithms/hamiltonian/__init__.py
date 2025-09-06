"""Hamiltonian normal form algorithms for CR3BP equilibrium point analysis.

This package provides comprehensive tools for constructing polynomial normal forms
of Hamiltonian systems around equilibrium points in the Circular Restricted Three-Body
Problem (CR3BP). It implements the complete pipeline from physical Hamiltonians
through coordinate transformations to normal forms and center manifold reductions.

Key Features
------------
- Polynomial Hamiltonian construction around collinear and triangular points
- Lie series transformations for canonical coordinate changes
- Coordinate system transformations (physical, modal, complex)
- Partial and full normal form computations
- Center manifold dimensional reduction
- Registry-based conversion system for automatic transformations

The package supports analysis around all five libration points (L1-L5) with
appropriate handling of hyperbolic directions at collinear points and elliptic
directions at triangular points.

All computations use nondimensional CR3BP units where:
- Distance: primary-secondary separation = 1
- Time: normalized by orbital period / (2*pi)
- Mass: total system mass = 1
- Energy: normalized by characteristic energy scale

Modules
-------
hamiltonian
    Polynomial Hamiltonian construction using Chebyshev and Legendre expansions.
lie
    Lie series transformations for canonical coordinate changes.
transforms
    Coordinate system transformations and complexification utilities.
wrappers
    Registry-based conversion system for automatic Hamiltonian transformations.
center
    Partial normal form computations for center manifold analysis.
normal
    Full normal form computations for complete dynamical reduction.

See Also
--------
:mod:`hiten.system.libration`
    Libration point classes providing equilibrium point data.
:mod:`hiten.algorithms.polynomial`
    Polynomial algebra operations used throughout the package.
:mod:`hiten.system.hamiltonians`
    Base Hamiltonian classes and data structures.

References
----------
Jorba, A. (1999). A methodology for the numerical computation of normal forms,
centre manifolds and first integrals of Hamiltonian systems. Experimental
Mathematics, 8(2), 155-195.

Meyer, K.R., Hall, G.R. (1992). Introduction to Hamiltonian Dynamical Systems
and the N-Body Problem. Springer-Verlag.

Gomez, G., Llibre, J., Martinez, R., Simo, C. (2001). Dynamics and Mission Design
Near Libration Points. World Scientific.
"""

from .wrappers import _CONVERSION_REGISTRY

__all__ = ["_CONVERSION_REGISTRY"]