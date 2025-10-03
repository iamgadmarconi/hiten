"""Types for the continuation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol,
                    Sequence, Tuple, Union)

import numpy as np

from hiten.algorithms.types.core import _BackendCall, _DomainPayload


class PredictorCallable(Protocol):
    def __call__(self, last: np.ndarray, step: np.ndarray) -> np.ndarray:
        ...


class CorrectorCallable(Protocol):
    def __call__(self, prediction: np.ndarray) -> tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict]:
        ...


class StepperFnCallable(Protocol):
    def __call__(self, last: np.ndarray, step: np.ndarray) -> Any:
        ...


@dataclass
class ContinuationBackendRequest:
    seed_repr: np.ndarray
    stepper_fn: StepperFnCallable
    predictor_fn: PredictorCallable
    parameter_getter: Callable[[np.ndarray], np.ndarray]
    corrector: CorrectorCallable
    step: np.ndarray
    target: np.ndarray
    max_members: int
    max_retries_per_step: int
    shrink_policy: Callable[[np.ndarray], np.ndarray] | None
    step_min: float
    step_max: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuationBackendResponse:
    family_repr: list[np.ndarray]
    info: dict[str, Any]

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
class ContinuationDomainPayload(_DomainPayload):
    """Domain payload containing continuation family data."""

    @classmethod
    def _from_mapping(cls, data: Mapping[str, object]) -> "ContinuationDomainPayload":
        return cls(data=data)

    @property
    def family(self) -> Tuple[object, ...]:
        return tuple(self.require("family"))

    @property
    def family_repr(self) -> Tuple[np.ndarray, ...]:
        return tuple(np.asarray(vec, dtype=float) for vec in self.require("family_repr"))

    @property
    def accepted_count(self) -> int:
        return int(self.require("accepted_count"))

    @property
    def rejected_count(self) -> int:
        return int(self.require("rejected_count"))

    @property
    def success_rate(self) -> float:
        return float(self.require("success_rate"))

    @property
    def iterations(self) -> int:
        return int(self.require("iterations"))

    @property
    def parameter_values(self) -> Tuple[np.ndarray, ...]:
        return tuple(np.asarray(val, dtype=float) for val in self.require("parameter_values"))

    @property
    def info(self) -> dict[str, object]:
        info = self.get("info", {})
        return dict(info) if isinstance(info, Mapping) else dict()


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