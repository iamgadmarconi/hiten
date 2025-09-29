"""
Types for the corrector module.

This module provides the types for the corrector module.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from hiten.algorithms.corrector.config import _LineSearchConfig

if TYPE_CHECKING:
    from hiten.algorithms.corrector.protocols import CorrectorStepProtocol

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

StepperFactory = Callable[[ResidualFn, NormFn, float | None], "CorrectorStepProtocol"]


@dataclass
class CorrectionResult:
    """Standardized result for a backend correction run.
    
    Attributes
    ----------
    converged : bool
        Whether the correction converged.
    x_corrected : ndarray
        Corrected parameter vector.
    residual_norm : float
        Final residual norm.
    iterations : int
        Number of iterations performed.
    """
    converged: bool
    x_corrected: np.ndarray
    residual_norm: float
    iterations: int


@dataclass
class OrbitCorrectionResult(CorrectionResult):
    """Result for an orbit correction run.
    
    Attributes
    ----------
    half_period : float
        Half-period associated with the corrected orbit.
    """
    half_period: float


@dataclass
class _CorrectionProblem:
    """Defines the inputs for a backend correction run.

    Attributes
    ----------
    initial_guess : ndarray
        Initial parameter vector.
    residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
        Residual function R(x).
    jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` | None
        Optional analytical Jacobian.
    norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` | None
        Optional norm function for convergence checks.
    max_attempts : int
        Maximum number of Newton iterations to attempt.
    tol : float
        Convergence tolerance for the residual norm.
    max_delta : float
        Maximum allowed infinity norm of Newton steps.
    line_search_config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig` | bool | None
        Configuration for line search behavior.
    finite_difference : bool
        Force finite-difference approximation of Jacobians.
    fd_step : float
        Finite-difference step size.
    method : str
        Integration method for trajectory computation.
    order : int
        Integration order for numerical methods.
    steps : int
        Number of integration steps.
    forward : int
        Integration direction (1 for forward, -1 for backward).
    stepper_factory : callable or None
        Optional factory producing a stepper compatible with the backend.
    """
    initial_guess: np.ndarray
    residual_fn: ResidualFn
    jacobian_fn: Optional[JacobianFn]
    norm_fn: Optional[NormFn]
    max_attempts: int
    tol: float
    max_delta: float
    line_search_config: _LineSearchConfig | bool | None
    finite_difference: bool
    fd_step: float
    method: str
    order: int
    steps: int
    forward: int
    stepper_factory: Optional[StepperFactory]


@dataclass
class _OrbitCorrectionProblem(_CorrectionProblem):
    """Defines the inputs for a backend orbit correction run.
    
    Attributes
    ----------
    domain_obj: Any
        Orbit to be corrected.
    residual_indices : tuple of int
        State components used to build the residual vector.
    control_indices : tuple of int
        State components allowed to change during correction.
    extra_jacobian : callable or None
        Additional Jacobian contribution function.
    target : tuple of float
        Target values for the residual components.
    event_func : callable
        Function to detect Poincare section crossings.
    """
    domain_obj: Any
    residual_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    extra_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    target: tuple[float, ...]
    event_func: Callable[..., tuple[float, np.ndarray]]
