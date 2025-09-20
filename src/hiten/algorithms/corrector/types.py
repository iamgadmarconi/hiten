"""
Types for the corrector module.



"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

#: Type alias for residual function signatures.
#:
#: Functions of this type compute residual vectors from parameter vectors,
#: representing the nonlinear equations to be solved. The residual should
#: approach zero as the parameter vector approaches the solution.
#:
#: In dynamical systems contexts, the residual typically represents:
#: - Constraint violations for periodic orbits
#: - Boundary condition errors for invariant manifolds
#: - Fixed point equations for equilibrium solutions
#:
#: Parameters
#: ----------
#: x : ndarray
#:     Parameter vector at which to evaluate the residual.
#:
#: Returns
#: -------
#: residual : ndarray
#:     Residual vector of the same shape as the input.
#:
#: Notes
#: -----
#: The residual function should be well-defined and continuous in
#: the neighborhood of the expected solution. For best convergence
#: properties, it should also be differentiable with a non-singular
#: Jacobian at the solution.
ResidualFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for Jacobian function signatures.
#:
#: Functions of this type compute Jacobian matrices (first derivatives)
#: of residual functions with respect to parameter vectors. The Jacobian
#: is essential for Newton-type methods and provides information about
#: the local linearization of the nonlinear system.
#:
#: Parameters
#: ----------
#: x : ndarray
#:     Parameter vector at which to evaluate the Jacobian.
#:
#: Returns
#: -------
#: jacobian : ndarray
#:     Jacobian matrix with shape (n, n) where n is the length of x.
#:     Element (i, j) contains the partial derivative of residual[i]
#:     with respect to x[j].
#:
#: Notes
#: -----
#: For Newton methods to converge quadratically, the Jacobian should
#: be continuous and non-singular in a neighborhood of the solution.
#: When analytic Jacobians are not available, finite-difference
#: approximations can be used at the cost of reduced convergence rate.
JacobianFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for norm function signatures.
#:
#: Functions of this type compute scalar norms from vectors, providing
#: a measure of vector magnitude used for convergence assessment and
#: step-size control. The choice of norm can affect convergence behavior
#: and numerical stability.
#:
#: Parameters
#: ----------
#: vector : ndarray
#:     Vector to compute the norm of.
#:
#: Returns
#: -------
#: norm : float
#:     Scalar norm value (non-negative).
#:
#: Notes
#: -----
#: Common choices include:
#: - L2 norm (Euclidean): Good general-purpose choice
#: - Infinity norm: Emphasizes largest component
#: - Weighted norms: Account for different scales in components
#:
#: The norm should be consistent across all uses within a single
#: correction process to ensure proper convergence assessment.
NormFn = Callable[[np.ndarray], float]


class CorrectionResult(NamedTuple):

    converged: bool
    x_corrected: np.ndarray
    residual_norm: float
    iterations: int


class _CorrectionProblem(NamedTuple):
    pass