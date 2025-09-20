"""Abstract base class for continuation backends."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class _ContinuationBackend(ABC):
    
    @abstractmethod
    def solve(
        self,
        *,
        seed_repr: np.ndarray,
        stepper: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        parameter_getter: Callable[[np.ndarray], np.ndarray],
        corrector: Callable[[np.ndarray], tuple[np.ndarray, float, bool]],
        representation_of: Callable[[np.ndarray], np.ndarray] | None,
        set_tangent: Callable[[np.ndarray | None], None] | None,
        step: np.ndarray,
        target: np.ndarray,
        max_members: int,
        max_retries_per_step: int,
        shrink_policy: Callable[[np.ndarray], np.ndarray] | None,
        step_min: float,
        step_max: float,
    ) -> tuple[list[np.ndarray], dict]:
        """Run continuation using purely numerical inputs and callables.

        Parameters
        ----------
        seed_repr : ndarray
            Numerical representation of the seed solution.
        stepper : callable
            stepper(last_repr, step) -> (next_prediction: ndarray, step_hint: ndarray)
        parameter_getter : callable
            parameter_getter(repr) -> ndarray of continuation parameters.
        corrector : callable
            corrector(prediction_repr) -> (corrected_repr, residual_norm, converged).
        representation_of : callable, optional
            Maps a domain solution to its numerical representation (for secant updates).
        set_tangent : callable, optional
            Setter to update the unit tangent vector maintained by the backend.
        step : ndarray
            Initial step vector (m,).
        target : ndarray
            Bounds array shaped (2, m): [mins; maxs].
        max_members : int
            Maximum number of accepted members (including the seed).
        max_retries_per_step : int
            Maximum retries allowed when correction fails at a step.
        shrink_policy : callable, optional
            Function to produce a reduced step on failure.
        step_min : float
            Minimum allowed |step| magnitude.
        step_max : float
            Maximum allowed |step| magnitude.

        Returns
        -------
        family_repr : list of ndarray
            Accepted member representations in order (including seed as first).
        info : dict
            Backend-specific telemetry (e.g., parameter history, counts, timings).
        """
        ...

    def on_iteration(self, k: int, x: np.ndarray, r_norm: float) -> None:
        """Called after each iteration. Default: no-op."""
        return

    def on_accept(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Called when the backend detects convergence. Default: no-op."""
        return

    def on_failure(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Called when the backend completes without converging. Default: no-op."""
        return

    def on_success(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Called by the Engine after final acceptance. Default: no-op."""
        return
