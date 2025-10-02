"""Simple stepping strategy using a provided predictor function."""

from typing import Callable, Optional

import numpy as np

from hiten.algorithms.continuation.stepping.base import (
    _StepProposal,
    _ContinuationStepBase,
)


class _ContinuationPlainStep(_ContinuationStepBase):
    """Implement a simple stepping strategy using a provided predictor function."""

    def __init__(
        self,
        predictor: Callable[[object, np.ndarray], np.ndarray],
        *,
        step_min: float = 1e-10,
        step_max: float = 1.0,
        shrink_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        super().__init__(step_min=step_min, step_max=step_max, shrink_policy=shrink_policy)
        self._predictor = predictor

    def predict(self, last_solution: object, step: np.ndarray) -> _StepProposal:
        prediction = self._predictor(last_solution, step)
        return _StepProposal(np.asarray(prediction, dtype=float), np.asarray(step, dtype=float))

    def on_accept(
        self,
        *,
        last_solution: object,
        new_solution: np.ndarray,
        step: np.ndarray,
        proposal: _StepProposal,
    ) -> Optional[np.ndarray]:
        return proposal.step_hint