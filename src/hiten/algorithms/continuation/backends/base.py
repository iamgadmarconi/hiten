"""Abstract base class for continuation backends."""

from abc import abstractmethod
from typing import Callable

import numpy as np

from hiten.algorithms.types.core import _HitenBaseBackend


class _ContinuationBackend(_HitenBaseBackend):
    
    @abstractmethod
    def run(
        self,
        *,
        seed_repr: np.ndarray,
        stepper: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        parameter_getter: Callable[[np.ndarray], np.ndarray],
        corrector: Callable[[np.ndarray], tuple[np.ndarray, float, bool]],
        representation_of: Callable[[np.ndarray], np.ndarray] | None,
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
        seed_repr : np.ndarray
            Numerical representation of the seed solution.
        stepper : callable
            stepper(last_repr, step) -> (next_prediction: np.ndarray, step_hint: np.ndarray)
        parameter_getter : callable
            parameter_getter(repr) -> np.ndarray of continuation parameters.
        corrector : callable
            corrector(prediction_repr) -> (corrected_repr, residual_norm, converged).
        representation_of : callable, optional
            Maps a domain solution to its numerical representation (for secant updates).
        step : np.ndarray
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
        family_repr : list of np.ndarray
            Accepted member representations in order (including seed as first).
        info : dict
            Backend-specific telemetry (e.g., parameter history, counts, timings).
        """
        ...

    def get_tangent(self) -> np.ndarray | None:
        """Return the current tangent vector maintained by the backend.

        Stateless backends may return None.
        """
        return None

    def seed_tangent(self, tangent: np.ndarray | None) -> None:
        """Seed the backend with an initial tangent vector prior to 
        :meth:`~hiten.algorithms.continuation.backends.base._ContinuationBackend.run`.

        Engines may call this at most once. Default implementation is a no-op.
        """
        return
