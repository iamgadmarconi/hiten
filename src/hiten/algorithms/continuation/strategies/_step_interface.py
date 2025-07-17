from typing import Callable, Protocol

import numpy as np


class _ContinuationStep(Protocol):

    def __call__(self, last_solution: object, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class _PlainStep:

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        self._predictor = predictor

    def __call__(self, last_solution, step):
        return self._predictor(last_solution, step), step