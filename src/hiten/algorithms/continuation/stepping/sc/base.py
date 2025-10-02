"""Stateless secant stepping strategy."""

from typing import Callable, Optional

import numpy as np

from hiten.algorithms.continuation.stepping.base import (
    _StepProposal,
    _ContinuationStepBase,
)


class _SecantStep(_ContinuationStepBase):
    """Stateless secant step using an external tangent provider."""

    def __init__(
        self,
        representation_fn: Callable[[object], np.ndarray],
        tangent_provider: Callable[[], np.ndarray | None],
    ) -> None:
        self._repr_fn = representation_fn
        self._tangent_provider = tangent_provider

    def predict(self, last_solution: object, step: np.ndarray) -> _StepProposal:
        r_last = self._repr_fn(last_solution)
        tan = self._tangent_provider()

        ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))

        if tan is None:
            n = np.asarray(r_last, dtype=float).copy()
            if n.size > 0:
                n[0] = n[0] + ds_scalar
            return _StepProposal(n, np.asarray(step, dtype=float))

        dr = np.asarray(tan, dtype=float).reshape(r_last.shape) * ds_scalar
        prediction = np.asarray(r_last, dtype=float) + dr
        return _StepProposal(prediction, np.asarray(step, dtype=float))
