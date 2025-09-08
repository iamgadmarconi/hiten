"""Polynomial algebra for the Circular Restricted Three-Body Problem.

This module provides comprehensive tools for polynomial operations in the CR3BP,
including multivariate polynomial manipulation, coefficient array operations,
and efficient evaluation routines optimized for dynamical systems analysis.

The module implements efficient storage and manipulation of multivariate
polynomials in the 6D phase space (q1, q2, q3, p1, p2, p3) of the circular
restricted three-body problem using compressed monomial ordering.

See Also
--------
:mod:`~hiten.algorithms.hamiltonian`
    Hamiltonian normal form computations using polynomial algebra
:mod:`~hiten.algorithms.fourier`
    Fourier analysis that builds on polynomial operations
"""