"""Types for the continuation module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.corrector.options import OrbitCorrectionOptions


@dataclass(frozen=True)
class ContinuationResult:
    """Standardized result for a continuation run.
    
    Attributes
    ----------
    accepted_count : int
        The number of accepted solutions.
    rejected_count : int
        The number of rejected solutions.
    success_rate : float
        The success rate.
    family : Tuple[object, ...]
        The family of solutions.
    parameter_values : Tuple[np.ndarray, ...]
        The parameter values.
    iterations : int
        The number of iterations.
    """

    accepted_count: int
    rejected_count: int
    success_rate: float
    family: Tuple[object, ...]
    parameter_values: Tuple[np.ndarray, ...]
    iterations: int


@dataclass(frozen=True)
class _ContinuationProblem:
    """Defines the inputs for a continuation run.
    
    Attributes
    ----------
    initial_solution : object
        Starting solution for the continuation.
    parameter_getter : callable
        Function that extracts continuation parameter(s) from a solution object.
    target : sequence
        Target parameter range(s) for continuation. For 1D: (min, max).
        For multi-dimensional: (2, m) array where each column specifies
        (min, max) for one parameter.
    step : float or sequence of float
        Initial step size(s) for continuation parameters. If scalar,
        uses same step for all parameters.
    max_members : int
        Maximum number of accepted solutions to generate.
    max_retries_per_step : int
        Maximum number of retries per failed continuation step.
    representation_of : callable or None
        Function to convert solution objects to vector representations.
    shrink_policy : callable or None
        Policy for shrinking step sizes when continuation fails.
    step_min : float
        Minimum allowed step size.
    step_max : float
        Maximum allowed step size.
    stepper : str
        The stepper to use.
    state_indices : Optional[np.ndarray]
        The state indices.
    corrector_tol : float
        Convergence tolerance for the corrector residual norm.
    corrector_max_attempts : int
        Maximum number of corrector iterations.
    corrector_max_delta : float
        Maximum allowed infinity norm of corrector Newton steps.
    corrector_order : int
        Integration order for corrector.
    corrector_steps : int
        Number of integration steps for corrector.
    corrector_forward : int
        Integration direction for corrector (1 for forward, -1 for backward).
    corrector_fd_step : float
        Finite difference step size for corrector.
    corrector_options_override : Optional["OrbitCorrectionOptions"]
        Full corrector options object that overrides all individual parameters if provided.
    """

    initial_solution: object
    parameter_getter: Callable[[np.ndarray], np.ndarray]
    target: np.ndarray
    step: np.ndarray
    max_members: int
    max_retries_per_step: int
    representation_of: Optional[Callable[[np.ndarray], np.ndarray]] = None
    shrink_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None
    step_min: float = 1e-10
    step_max: float = 1.0
    stepper: str = "natural"
    state_indices: Optional[np.ndarray] = None
    corrector_tol: float = 1e-12
    corrector_max_attempts: int = 50
    corrector_max_delta: float = 1e-2
    corrector_order: int = 8
    corrector_steps: int = 500
    corrector_forward: int = 1
    corrector_fd_step: float = 1e-8