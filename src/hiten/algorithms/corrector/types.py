"""
Types for the corrector module.

This module provides the types for the corrector module.
"""

from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol,
                    Sequence)

import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.corrector.protocols import CorrectorStepProtocol

from hiten.algorithms.types.core import _DomainPayload

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


class ResidualCallable(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


class JacobianCallable(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


class NormCallable(Protocol):
    def __call__(self, residual: np.ndarray) -> float:
        ...


@dataclass
class CorrectorInput:
    """Structured request describing a single backend correction solve."""
    initial_guess: np.ndarray
    residual_fn: ResidualCallable
    jacobian_fn: Optional[JacobianCallable]
    norm_fn: Optional[NormCallable]
    max_attempts: int
    tol: float
    max_delta: float | None
    fd_step: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectorOutput:
    """Structured response produced by a backend correction solve."""

    x_corrected: np.ndarray
    iterations: int
    residual_norm: float
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class OrbitCorrectionDomainPayload(_DomainPayload):
    """Domain payload describing updates produced by orbit correction interfaces."""

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> "OrbitCorrectionDomainPayload":
        return cls(data=data)

    @property
    def x_full(self) -> np.ndarray:
        return np.asarray(self.require("x_full"), dtype=float)

    @property
    def half_period(self) -> float:
        return float(self.require("half_period"))

    @property
    def iterations(self) -> int:
        return int(self.require("iterations"))

    @property
    def residual_norm(self) -> float:
        return float(self.require("residual_norm"))


@dataclass(frozen=True)
class MultipleShootingDomainPayload(OrbitCorrectionDomainPayload):
    """Domain payload with additional diagnostics for multiple-shooting runs."""

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> "MultipleShootingDomainPayload":
        return cls(data=data)

    @property
    def patch_states(self) -> Sequence[np.ndarray]:
        raw = self.require("patch_states")
        return tuple(np.asarray(p, dtype=float) for p in raw)

    @property
    def patch_times(self) -> np.ndarray:
        return np.asarray(self.require("patch_times"), dtype=float)

    @property
    def continuity_errors(self) -> np.ndarray:
        errors = self.get("continuity_errors", None)
        if errors is None:
            return np.asarray([], dtype=float)
        return np.asarray(errors, dtype=float)


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


@dataclass
class MultipleShootingResult(CorrectionResult):
    pass

@dataclass
class _MultipleShootingProblem(_CorrectionProblem):
    pass