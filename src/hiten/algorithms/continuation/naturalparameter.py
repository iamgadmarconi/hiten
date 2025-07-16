from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine


class _NaturalParameter(_ContinuationEngine, ABC):
    """Abstract base class for natural-parameter continuation algorithms."""

    def __init__(self, *args, **kwargs):
        """Initialise the underlying continuation engine and enforce natural-parameter
        policies (monotone parameter advance toward the target interval and
        interval-based stopping criterion)."""

        super().__init__(*args, **kwargs)

        # Ensure the initial step points from the current parameter value toward
        # the target interval.  If it does not, flip its sign component-wise.
        current_param = self._param_history[-1]
        for i in range(current_param.size):
            if (current_param[i] < self._target_min[i] and self._step[i] < 0) or (
                current_param[i] > self._target_max[i] and self._step[i] > 0
            ):
                self._step[i] = -self._step[i]

    @abstractmethod
    def _predict(self, last_solution: object, step: np.ndarray) -> np.ndarray:
        """Return a predicted representation for the next solution."""
        pass

    def _stop_condition(self) -> bool:
        """Terminate when the parameter leaves the prescribed [min, max] window."""

        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)