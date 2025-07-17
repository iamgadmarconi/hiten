from typing import Callable, Protocol

import numpy as np


class _StepStrategy(Protocol):

    def __call__(
        self,
        last_solution: object,
        step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

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
    def on_success(self, *_, **__):
        pass

    def on_iteration(self, *_, **__):
        pass

    def on_reject(self, *_, **__):
        pass

    def on_failure(self, *_, **__):
        pass

    def on_initialisation(self, *_, **__):
        pass


class _SecantStep:
    """Secant predictor + tangent maintenance for pseudo-arclength engines.
    """

    def __init__(
        self,
        representation_fn: Callable[[object], np.ndarray],
        parameter_fn: Callable[[object], np.ndarray],
    ) -> None:

        self._repr_fn = representation_fn
        self._param_fn = parameter_fn

        # History buffers (updated in on_success)
        self._repr_hist: list[np.ndarray] = []
        self._param_hist: list[np.ndarray] = []

        self._tangent: np.ndarray | None = None

    def __call__(self, last_solution: object, step) -> tuple[np.ndarray, np.ndarray]:
        # Ensure history contains last solution
        if not self._repr_hist:
            self._repr_hist.append(self._repr_fn(last_solution))
            self._param_hist.append(self._param_fn(last_solution))

        # If we have a valid tangent use it, otherwise perform small natural step
        if self._tangent is None:
            # Fallback: small natural-parameter step of magnitude |step|.
            # Ensure both scalar and vector ``step`` inputs are handled consistently.
            ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))
            n = self._repr_hist[-1].copy()
            n[0] += ds_scalar  # naive perturb of the first component by |step|
            return n, step

        ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))
        n_repr = self._repr_hist[-1].size
        dr = self._tangent[:n_repr] * ds_scalar
        new_repr = self._repr_hist[-1] + dr
        return new_repr, step

    def _update_history_and_tangent(self, accepted_solution: object):
        """Private helper that records representation/parameter history and
        (re)computes the secant tangent vector."""

        r = self._repr_fn(accepted_solution)
        p = self._param_fn(accepted_solution)

        self._repr_hist.append(r)
        self._param_hist.append(p)

        if len(self._repr_hist) < 2:
            self._tangent = None
            return

        dr = self._repr_hist[-1] - self._repr_hist[-2]
        dp = self._param_hist[-1] - self._param_hist[-2]
        vec = np.concatenate((dr.ravel(), dp.ravel()))
        norm = np.linalg.norm(vec)
        self._tangent = None if norm == 0 else vec / norm

    def on_success(self, accepted_solution: object):
        """Callback invoked by the continuation engine after a successful step."""

        self._update_history_and_tangent(accepted_solution)

    def on_iteration(self, *_, **__):
        pass

    def on_reject(self, *_, **__):
        pass

    def on_failure(self, *_, **__):
        pass

    def on_initialisation(self, first_solution: object):
        # Prime history with seed when engine notifies initialisation
        self._repr_hist.append(self._repr_fn(first_solution))
        self._param_hist.append(self._param_fn(first_solution))