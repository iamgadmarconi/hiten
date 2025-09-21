"""Stateless secant stepping strategy."""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.stepping.base import _ContinuationStepBase


class _SecantStep(_ContinuationStepBase):
    """Stateless secant step using an external tangent provider.

    The backend owns history and tangent computation. This stepper simply
    uses the provided tangent to form a prediction and returns the step
    hint unchanged.

    Parameters
    ----------
    representation_fn : callable
        Maps a solution object to its numerical representation.
    tangent_provider : callable
        Returns the current unit tangent vector in representation space.
        Should return ndarray with shape matching the representation.
    """

    def __init__(
        self,
        representation_fn: Callable[[object], np.ndarray],
        tangent_provider: Callable[[], np.ndarray | None],
    ) -> None:
        self._repr_fn = representation_fn
        self._tangent_provider = tangent_provider

    def __call__(self, last_solution: object, step) -> tuple[np.ndarray, np.ndarray]:
        r_last = self._repr_fn(last_solution)
        tan = self._tangent_provider()

        # Magnitude from step (supports scalar or vector)
        ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))

        if tan is None:
            # Fallback: perturb first component by |step|
            n = r_last.copy()
            if n.size == 0:
                return n, step
            n[0] = n[0] + ds_scalar
            return n, step

        dr = tan.reshape(r_last.shape) * ds_scalar
        return r_last + dr, step
