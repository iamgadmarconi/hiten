"""Types for the continuation module."""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from hiten.algorithms.continuation.config import _ContinuationConfig


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    """Standardized result for a continuation run."""

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
    corrector_kwargs : dict
        Additional keyword arguments passed to the corrector method.
    """

    initial_solution: object
    parameter_getter: Callable[[np.ndarray], np.ndarray]
    target: np.ndarray
    step: np.ndarray
    max_members: int
    max_retries_per_step: int
    corrector_kwargs: dict
    representation_of: Callable[[np.ndarray], np.ndarray] | None = None
    shrink_policy: Callable[[np.ndarray], np.ndarray] | None = None
    step_min: float = 1e-10
    step_max: float = 1.0
    cfg: _ContinuationConfig | None = None
