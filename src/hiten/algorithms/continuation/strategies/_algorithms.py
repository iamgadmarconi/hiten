from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine
from hiten.algorithms.continuation.strategies._stepping import _SecantStep


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

    def _stop_condition(self) -> bool:
        """Terminate when the parameter leaves the prescribed [min, max] window."""

        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    def _make_stepper(self):
        raise NotImplementedError(
            "Natural-parameter continuations must define a StepStrategy by "
            "overriding _make_stepper() or assigning self._stepper before "
            "calling super().__init__."
        )


class _SecantArcLength(_ContinuationEngine, ABC):
    """Abstract base class for pseudo arclength continuation algorithms"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Build and assign secant stepper strategy
        self._stepper = _SecantStep(
            representation_fn=self._representation,
            parameter_fn=lambda obj: np.asarray(self._parameter(obj), dtype=float),
        )

        # Notify strategy of initialisation
        if hasattr(self._stepper, "on_initialisation"):
            self._stepper.on_initialisation(self._family[0])

    @abstractmethod
    def _representation(self, obj: object) -> np.ndarray:
        pass

    def _stop_condition(self) -> bool:
        """Terminate when the parameter leaves the prescribed target window."""

        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    # _on_accept may still be used for extra bookkeeping but tangent handled by strategy
    def _on_accept(self, member: object) -> None:
        pass

    def _make_stepper(self):
        raise NotImplementedError(
            "Secant-based continuations must define a StepStrategy by "
            "overriding _make_stepper() or assigning self._stepper before "
            "calling super().__init__."
        )