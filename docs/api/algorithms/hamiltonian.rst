Hamiltonian Module
==================

The Hamiltonian module provides comprehensive tools for constructing polynomial normal forms
of Hamiltonian systems around equilibrium points in the Circular Restricted Three-Body
Problem (CR3BP). It implements the complete pipeline from physical Hamiltonians
through coordinate transformations to normal forms and center manifold reductions.

The package supports analysis around all five libration points (L1-L5) with
appropriate handling of hyperbolic directions at collinear points and elliptic
directions at triangular points.

.. currentmodule:: hiten.algorithms.hamiltonian

hamiltonian.py
~~~~~~~~~~~~~~

The hamiltonian module provides polynomial Hamiltonian construction using Chebyshev and Legendre expansions.

.. currentmodule:: hiten.algorithms.hamiltonian.hamiltonian

.. autofunction:: _build_T_polynomials
   :noindex:

Compute Chebyshev polynomials of the first kind for collinear point expansions.
Generates the sequence T_n(r) where r = x / sqrt(x^2 + y^2 + z^2) using the classical
Chebyshev recurrence relation adapted for Cartesian coordinates in nondimensional CR3BP units.

.. autofunction:: _build_R_polynomials
   :noindex:

Generate auxiliary R_n polynomials for Lindstedt-Poincare formulation.
Computes the sequence R_n required for the Lindstedt-Poincare right-hand side
polynomials in collinear point normal form computations.

.. autofunction:: _build_potential_U
   :noindex:

Assemble gravitational potential expansion for collinear points.
Constructs the effective potential U = -sum_{n>=2} c_n * T_n(r) using
Chebyshev polynomial expansions and libration point coefficients.

.. autofunction:: _build_kinetic_energy_terms
   :noindex:

Build kinetic energy polynomial T = (1/2) * (px^2 + py^2 + pz^2).
Constructs the kinetic energy contribution to the Hamiltonian using
canonical momentum polynomials in nondimensional CR3BP units.

.. autofunction:: _build_rotational_terms
   :noindex:

Construct Coriolis (rotational) terms C = y*px - x*py for rotating frame.
Builds the Coriolis contribution to the Hamiltonian that arises from the
transformation to the rotating synodic reference frame in the CR3BP.

.. autofunction:: _build_physical_hamiltonian_collinear
   :noindex:

Build complete rotating-frame Hamiltonian H = T + U + C for collinear points.
Constructs the full CR3BP Hamiltonian around collinear equilibrium points
by combining kinetic energy, gravitational potential, and Coriolis terms.

.. autofunction:: _build_A_polynomials
   :noindex:

Generate Legendre-type polynomials A_n for triangular point expansions.
Computes the sequence A_n used for inverse distance expansions around
triangular equilibrium points (L4/L5) in the CR3BP using a Legendre-type
recurrence relation.

.. autofunction:: _build_physical_hamiltonian_triangular
   :noindex:

Build rotating-frame Hamiltonian for triangular points (L4/L5).
Constructs the complete CR3BP Hamiltonian around triangular equilibria
using Legendre-type polynomial expansions for the inverse distance terms
to the primary masses.

.. autofunction:: _build_lindstedt_poincare_rhs_polynomials
   :noindex:

Compute right-hand side polynomials for Lindstedt-Poincare method.
Generates the polynomial right-hand sides for the x, y, z equations in
the first iteration of the Lindstedt-Poincare method for constructing
periodic solutions around collinear equilibrium points.

lie.py
~~~~~~

The lie module provides Lie series transformations for canonical coordinate changes.

.. currentmodule:: hiten.algorithms.hamiltonian.lie

.. autofunction:: _solve_homological_equation
   :noindex:

Solve homological equation for polynomial generating function coefficients.
Computes the generating function coefficients that eliminate specified terms
from a polynomial Hamiltonian through canonical transformation.

.. autofunction:: _apply_poly_transform
   :noindex:

Apply Lie series transformation to polynomial Hamiltonian.
Transforms a polynomial Hamiltonian using a canonical transformation generated
by a polynomial generating function via the Lie series method.

transforms.py
~~~~~~~~~~~~~

The transforms module provides coordinate system transformations and complexification utilities.

.. currentmodule:: hiten.algorithms.hamiltonian.transforms

.. autofunction:: _build_complexification_matrix
   :noindex:

Build complexification transformation matrix for coordinate pairs.
Creates the matrix that transforms real canonical coordinate pairs to complex coordinates.

.. autofunction:: _M
   :noindex:

Return complexification transformation matrix for canonical coordinate pairs.
Provides the matrix M that transforms real coordinates to complex coordinates
for elliptic directions in the CR3BP.

.. autofunction:: _M_inv
   :noindex:

Return inverse complexification matrix for real coordinate recovery.
Provides the inverse matrix M^{-1} that transforms complex coordinates back
to real coordinates.

.. autofunction:: _substitute_complex
   :noindex:

Transform polynomial from real to complex coordinates.
Converts polynomial coefficients from real modal coordinates to complex
coordinates using the complexification transformation.

.. autofunction:: _substitute_real
   :noindex:

Transform polynomial from complex to real coordinates.
Converts polynomial coefficients from complex coordinates back to real
modal coordinates using the inverse complexification transformation.

.. autofunction:: _solve_complex
   :noindex:

Transform real coordinates to complex coordinates.
Applies the complexification transformation to coordinate vectors.

.. autofunction:: _solve_real
   :noindex:

Transform complex coordinates to real coordinates.
Applies the inverse complexification transformation to coordinate vectors.

.. autofunction:: _polylocal2realmodal
   :noindex:

Transform polynomial from local frame to real modal frame.
Converts polynomial coefficients from local coordinates centered at the
equilibrium point to real modal coordinates aligned with eigenvectors.

.. autofunction:: _polyrealmodal2local
   :noindex:

Transform polynomial from real modal frame to local frame.
Converts polynomial coefficients from real modal coordinates back to
local coordinates centered at the equilibrium point.

.. autofunction:: _coordrealmodal2local
   :noindex:

Transform coordinates from real modal to local frame.
Converts coordinate vectors from real modal coordinates to local coordinates.

.. autofunction:: _coordlocal2realmodal
   :noindex:

Transform coordinates from local to real modal frame.
Converts coordinate vectors from local coordinates to real modal coordinates.

.. autofunction:: _local2synodic_collinear
   :noindex:

Transform coordinates from local to synodic frame for collinear points.
Converts coordinates from the local frame centered at a collinear libration
point to the synodic rotating frame coordinates.

.. autofunction:: _synodic2local_collinear
   :noindex:

Transform coordinates from synodic to local frame for collinear points.
Converts coordinates from the synodic rotating frame to the local frame
centered at a collinear libration point.

.. autofunction:: _local2synodic_triangular
   :noindex:

Transform coordinates from local to synodic frame for triangular points.
Converts coordinates from the local frame centered at a triangular libration
point to the synodic rotating frame coordinates.

.. autofunction:: _synodic2local_triangular
   :noindex:

Transform coordinates from synodic to local frame for triangular points.
Converts coordinates from the synodic rotating frame to the local frame
centered at a triangular libration point.

.. autofunction:: _restrict_poly_to_center_manifold
   :noindex:

Restrict polynomial Hamiltonian to center manifold by eliminating hyperbolic terms.
Removes all terms that depend on hyperbolic variables, effectively restricting
the Hamiltonian to the center-stable manifold subspace.

wrappers.py
~~~~~~~~~~~

The wrappers module provides a registry-based conversion system for automatic Hamiltonian transformations.

.. currentmodule:: hiten.algorithms.hamiltonian.wrappers

.. autofunction:: register_conversion
   :noindex:

Register a conversion function in the Hamiltonian conversion registry.
Decorator function that registers conversion functions between different
Hamiltonian representations in the automatic conversion system.

.. autofunction:: _physical_to_real_modal
   :noindex:

Transform Hamiltonian from physical to real modal coordinates.
Converts the physical Hamiltonian from local coordinates to real modal
coordinates aligned with the linear stability eigenvectors.

.. autofunction:: _real_modal_to_physical
   :noindex:

Transform Hamiltonian from real modal to physical coordinates.
Converts the Hamiltonian from real modal coordinates back to physical
local coordinates centered at the equilibrium point.

.. autofunction:: _real_modal_to_complex_modal
   :noindex:

Transform Hamiltonian from real modal to complex modal coordinates.
Converts the Hamiltonian from real modal coordinates to complex modal
coordinates for elliptic directions using complexification.

.. autofunction:: _complex_modal_to_real_modal
   :noindex:

Transform Hamiltonian from complex modal to real modal coordinates.
Converts the Hamiltonian from complex modal coordinates back to real
modal coordinates using inverse complexification.

.. autofunction:: _complex_modal_to_complex_partial_normal
   :noindex:

Transform Hamiltonian to partial normal form via Lie series method.
Applies partial Lie series normalization to eliminate non-resonant terms
while preserving resonant terms for center manifold analysis.

.. autofunction:: _complex_partial_normal_to_real_partial_normal
   :noindex:

Transform Hamiltonian from complex partial normal to real partial normal form.
Converts the complex partial normal form back to real coordinates while
maintaining the partial normalization structure.

.. autofunction:: _real_partial_normal_to_complex_partial_normal
   :noindex:

Transform Hamiltonian from real partial normal to complex partial normal form.
Converts the real partial normal form to complex coordinates for further
processing in the normalization pipeline.

.. autofunction:: _complex_partial_normal_to_center_manifold_complex
   :noindex:

Restrict Hamiltonian to center manifold by eliminating hyperbolic terms.
Removes all terms that depend on hyperbolic variables, effectively restricting
the Hamiltonian to the center-stable manifold subspace.

.. autofunction:: _center_manifold_complex_to_center_manifold_real
   :noindex:

Transform center manifold Hamiltonian from complex to real coordinates.
Converts the center manifold Hamiltonian from complex coordinates back
to real coordinates while maintaining the center manifold restriction.

.. autofunction:: _center_manifold_real_to_center_manifold_complex
   :noindex:

Transform center manifold Hamiltonian from real to complex coordinates.
Converts the center manifold Hamiltonian from real coordinates to complex
coordinates for further analysis or processing.

.. autofunction:: _complex_modal_to_complex_full_normal
   :noindex:

Transform Hamiltonian to full normal form via complete Lie series method.
Applies full Lie series normalization to eliminate all non-resonant terms,
producing the maximally simplified canonical form.

.. autofunction:: _complex_full_normal_to_real_full_normal
   :noindex:

Transform Hamiltonian from complex full normal to real full normal form.
Converts the complex full normal form back to real coordinates while
maintaining the complete normalization structure.

.. autofunction:: _real_full_normal_to_complex_full_normal
   :noindex:

Transform Hamiltonian from real full normal to complex full normal form.
Converts the real full normal form to complex coordinates for further
analysis or processing in the normalization pipeline.

center/
~~~~~~~

The center module provides partial normal form computations for center manifold analysis.

.. currentmodule:: hiten.algorithms.hamiltonian.center

.. autofunction:: _lie_transform
   :noindex:

Perform Lie series normalization of polynomial Hamiltonian for center manifold.
Implements the partial normal form algorithm that systematically eliminates
non-resonant terms from a polynomial Hamiltonian using Lie series transformations.

.. autofunction:: _get_homogeneous_terms
   :noindex:

Extract homogeneous terms of specified degree from polynomial.
JIT-compiled function that extracts the coefficient array corresponding
to homogeneous terms of a specific degree from a polynomial representation.

.. autofunction:: _select_terms_for_elimination
   :noindex:

Identify non-resonant terms for elimination in Lie normalization.
JIT-compiled function that selects polynomial terms to be eliminated
based on resonance conditions for center manifold analysis.

.. autofunction:: _lie_expansion
   :noindex:

Compute coordinate transformations using Lie series expansions.
Performs Lie series transformations to compute polynomial expansions
that relate center manifold coordinates to the original coordinate system.

.. autofunction:: _apply_coord_transform
   :noindex:

Apply Lie series transformation to single coordinate polynomial.
JIT-compiled function that applies a Lie series transformation to transform
a coordinate polynomial using a generating function.

.. autofunction:: _evaluate_transform
   :noindex:

Evaluate coordinate transformation at specific center manifold point.
JIT-compiled function that evaluates six polynomial expansions representing
coordinate transformations at a given point in center manifold coordinates.

.. autofunction:: _zero_q1p1
   :noindex:

Restrict polynomial expansions to center manifold subspace.
Eliminates all terms in coordinate expansions that depend on the hyperbolic
variables q1 or p1, effectively restricting the expansions to the center manifold.

normal/
~~~~~~~

The normal module provides full normal form computations for complete dynamical reduction.

.. currentmodule:: hiten.algorithms.hamiltonian.normal

.. autofunction:: _lie_transform
   :noindex:

Perform full Lie series normalization of polynomial Hamiltonian.
Eliminates all non-resonant terms using the complete resonance condition,
producing the maximally simplified canonical form.

.. autofunction:: _select_nonresonant_terms
   :noindex:

Identify non-resonant terms using full resonance condition.
JIT-compiled function that selects terms for elimination based on
the complete frequency analysis for full normal form computation.

