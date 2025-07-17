from typing import Callable, Protocol

import numpy as np


class _StepStrategy(Protocol):

    def __call__(
        self,
        last_solution: object,
        step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def on_accept(self, *args, **kwargs) -> None: ...

    def on_iteration(self, *args, **kwargs) -> None: ...

    def on_reject(self, *args, **kwargs) -> None: ...

    def on_failure(self, *args, **kwargs) -> None: ...

    def on_success(self, *args, **kwargs) -> None: ...

    def on_initialisation(self, *args, **kwargs) -> None: ...


class _NaturalParameterStep:
    """Generic strategy that forwards prediction to a user-supplied callable.

    It keeps the step vector unchanged and provides no-op hooks.  All domain-
    specific logic (state indices, amplitude manipulations, etc.) lives in the
    predictor function passed at construction time.
    """

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        self._predictor = predictor

    def __call__(self, last_solution: object, step: np.ndarray):
        return self._predictor(last_solution, step), step

    # Optional hooks, kept as no-ops
    def on_accept(self, *_, **__):
        pass

    def on_iteration(self, *_, **__):
        pass

    def on_reject(self, *_, **__):
        pass

    def on_failure(self, *_, **__):
        pass

    def on_success(self, *_, **__):
        pass

    def on_initialisation(self, *_, **__):
        pass


class _PseudoArcLengthStep:
    pass