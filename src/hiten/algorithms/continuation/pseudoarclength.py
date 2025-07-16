from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine


class _PseudoArcLength(_ContinuationEngine, ABC):
    """Abstract base class for pseudo arclength continuation algorithms"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        seed_repr = self._representation(self._family[0])
        self._repr_history: list[np.ndarray] = [np.asarray(seed_repr, dtype=float).copy()]

        self._tangent: "np.ndarray | None" = None

        step_mag = float(np.linalg.norm(self._step))
        self._step = step_mag if step_mag != 0.0 else 1e-4

    @abstractmethod
    def _representation(self, obj: object) -> np.ndarray:  # pragma: no cover
        """Return the *np.ndarray* representation of *obj*.

        This representation is what :py:meth:`_instantiate` consumes.  For a
        periodic orbit this would typically be its 6-component initial state
        vector.  Sub-classes *must* override this method.
        """
        pass

    def _update_tangent(self) -> None:
        """Compute secant-based unit tangent using the last two family members."""

        if len(self._repr_history) < 2:
            self._tangent = None
            return

        dr = self._repr_history[-1] - self._repr_history[-2]
        dp = self._param_history[-1] - self._param_history[-2]
        vec = np.concatenate((dr.ravel(), dp.ravel()))

        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return
        self._tangent = vec / norm

    def _predict(self, last_solution: object, step: "float | np.ndarray") -> np.ndarray:
        """Secant-based tangent predictor.

        Once a valid tangent has been established (two or more accepted
        members), the predictor moves a distance *step* along that tangent in
        the combined (representation, parameter) space and returns only the
        *representation* component.  The continuation parameter component is
        implicitly adjusted by the corrector.

        If the tangent is not yet defined (first step) this base
        implementation raises *RuntimeError* so that the concrete algorithm
        can provide an alternative initial predictor.
        """

        if self._tangent is None:
            raise RuntimeError("Tangent is undefined: sub-class must provide initial predictor.")

        ds = float(step)  # ensured scalar in __init__
        n_repr = self._repr_history[-1].size
        dr = self._tangent[:n_repr] * ds
        return self._repr_history[-1] + dr

    def _stop_condition(self) -> bool:
        """Terminate when the parameter leaves the prescribed target window."""

        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    def _on_accept(self, member: object) -> None:
        """Bookkeeping for accepted members."""
        self._repr_history.append(self._representation(member))
        self._update_tangent()