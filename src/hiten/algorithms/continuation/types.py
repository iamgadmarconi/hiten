"""
Types for the continuation module.

Defines the standardized Result and Problem objects used by continuation
engines, following the shared architecture used across algorithms.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np


class ContinuationResult(NamedTuple):
    """Standardized result for a continuation run.
    
    Attributes
    ----------
    accepted_count : int
        Whether the continuation converged.
    rejected_count : int
        Number of rejected solutions.
    success_rate : float
        Success rate of the continuation.
    family : list[object]
        List of accepted solutions.
    parameter_values : tuple[np.ndarray, ...]
        Tuple of parameter values for each solution in the family.
    iterations : int
        Total predict-correct iterations attempted.
    """

    accepted_count: int
    rejected_count: int
    success_rate: float
    family: list[object]
    parameter_values: tuple[np.ndarray, ...]
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
    set_tangent: Callable[[np.ndarray | None], None] | None = None
    shrink_policy: Callable[[np.ndarray], np.ndarray] | None = None
    step_min: float = 1e-10
    step_max: float = 1.0
