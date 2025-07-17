from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from hiten.algorithms.continuation.strategies._step_interface import \
    _ContinuationStep
from hiten.utils.log_config import logger


class _ContinuationEngine(ABC):

    def __init__(self, *,  initial_solution: object, parameter_getter: Callable[[object], "np.ndarray | float"],
                target: Sequence[Sequence[float] | float], step: float | Sequence[float] = 1e-4,
                corrector_kwargs: dict | None = None, max_iters: int = 256) -> None:

        self._getter = parameter_getter
        target_arr = np.asarray(target, dtype=float)
        if target_arr.ndim == 1:
            if target_arr.size != 2:
                raise ValueError("target must be (min,max) for 1-D or (2,m) for multi-D continuation")
            target_arr = target_arr.reshape(2, 1)
        elif not (target_arr.ndim == 2 and target_arr.shape[0] == 2):
            raise ValueError("target must be iterable shaped (2,) or (2,m)")

        current_param = np.asarray(self._getter(initial_solution), dtype=float)
        if current_param.ndim == 0:
            current_param = current_param.reshape(1)

        step_arr = np.asarray(step, dtype=float)
        if step_arr.size == 1:
            step_arr = np.full_like(current_param, float(step_arr))
        elif step_arr.size != current_param.size:
            raise ValueError("step length does not match number of continuation parameters")

        if target_arr.shape[1] != current_param.size:
            if target_arr.shape[1] == 1:
                target_arr = np.repeat(target_arr, current_param.size, axis=1)
            else:
                raise ValueError("target dimensionality mismatch with continuation parameter")

        self._target_min = np.minimum(target_arr[0], target_arr[1])
        self._target_max = np.maximum(target_arr[0], target_arr[1])

        self._step = step_arr.astype(float)

        self._family: list[object] = [initial_solution]
        self._param_history: list[np.ndarray] = [current_param.copy()]

        self._corrector_kwargs = corrector_kwargs or {}
        self._max_iters = int(max_iters)

        # Build stepper strategy (must be provided by subclass or mix-in)
        self._stepper: _ContinuationStep = self._make_stepper()
        # Notify strategy initialisation hook if present
        if hasattr(self._stepper, "on_initialisation"):
            try:
                self._stepper.on_initialisation(initial_solution)
            except Exception as exc:
                logger.debug("stepper on_initialisation hook error: %s", exc)

        logger.info(
            "Continuation initialised: parameter=%s, target=[%s - %s], step=%s, max_iters=%d",
            current_param,
            self._target_min,
            self._target_max,
            self._step,
            self._max_iters,
        )

    @property
    def family(self) -> Sequence[object]:  
        """Read-only view of the generated solution list (seed is index 0)."""

        return tuple(self._family)

    @property
    def parameter_values(self) -> Sequence[np.ndarray]:
        """Parameter value associated with each family member."""

        return tuple(self._param_history)

    def run(self) -> list[object]:
        logger.info("Starting continuation loop ...")
        attempts_at_current_step = 0

        while not self._stop_condition():
            if len(self._family) >= self._max_iters:
                logger.warning("Reached max_iters=%d, terminating continuation.", self._max_iters)
                break

            last_sol = self._family[-1]

            predicted_repr, next_step = self._stepper(last_sol, self._step)
            self._step = next_step.copy()

            candidate = self._instantiate(predicted_repr)

            try:
                candidate = self._correct(candidate, **self._corrector_kwargs)
            except Exception as exc:
                logger.debug(
                    "Correction failed at step %s (attempt %d): %s",
                    self._step,
                    attempts_at_current_step + 1,
                    exc,
                    exc_info=exc,
                )
                # Notify strategy of failure via _update_step fallback for now
                self._step = self._update_step(self._step, success=False)
                attempts_at_current_step += 1
                if attempts_at_current_step > 10:
                    logger.error("Too many failed attempts at current step; aborting continuation.")
                    break
                continue  # retry with reduced step

            attempts_at_current_step = 0  # reset counter on success
            self._family.append(candidate)

            param_val = self._parameter(candidate)
            self._param_history.append(np.asarray(param_val, dtype=float).copy())

            logger.info("Accepted member #%d, parameter=%s", len(self._family) - 1, param_val)

            # Call optional hook for subclasses/callbacks after successful acceptance
            try:
                self._on_accept(candidate)
            except Exception as exc:
                logger.warning("_on_accept hook raised exception: %s", exc)

            self._step = self._update_step(self._step, success=True)

            if hasattr(self._stepper, "on_success"):
                try:
                    self._stepper.on_success(candidate)
                except Exception as exc:
                    logger.debug("stepper on_success hook error: %s", exc)

        logger.info("Continuation finished : generated %d members.", len(self._family))
        return self._family

    def _instantiate(self, representation: np.ndarray):
        """Instantiate a domain object from the predicted representation."""

        raise NotImplementedError("_instantiate must be provided by a domain mix-in")

    def _correct(self, obj: object, **kwargs):  
        """Apply a problem-specific corrector returning the refined object."""

        raise NotImplementedError("_correct must be implemented by a domain mix-in")

    def _parameter(self, obj: object) -> np.ndarray:  
        """Return the continuation parameter value for the given object."""

        raise NotImplementedError("_parameter must be implemented by a domain mix-in")

    def _update_step(self, current_step: np.ndarray, *, success: bool) -> np.ndarray:  
        """Default component-wise multiplicative adaption (*2 on success, *0.5 on failure)."""

        factor = 2.0 if success else 0.5
        new_step = current_step * factor
        clipped_mag = np.clip(np.abs(new_step), 1e-10, 1.0)
        return np.sign(new_step) * clipped_mag

    @abstractmethod
    def _stop_condition(self) -> bool:  
        """Return True to terminate continuation."""

        raise NotImplementedError("_stop_condition must be provided by a sub-class")

    @staticmethod
    def _clamp_step(
        step_value: float,
        *,
        reference_value: float = 1.0,
        min_relative: float = 1e-6,
        min_absolute: float = 1e-8,
    ) -> float:
        """Sign-preserving clamp that enforces a minimum step magnitude."""

        if step_value == 0:
            return min_absolute

        ref_mag = abs(reference_value)
        min_step = max(min_absolute, ref_mag * min_relative) if ref_mag > min_absolute else min_absolute
        return np.sign(step_value) * max(min_step, abs(step_value))

    @staticmethod
    def _clamp_scale(scale_value: float, *, min_scale: float = 1e-3, max_scale: float = 1e3) -> float:
        """Clamp multiplicative scaling factors into a safe range."""

        return float(np.clip(scale_value, min_scale, max_scale))

    def __repr__(self) -> str:  
        return (
            f"{self.__class__.__name__}(n_members={len(self._family)}, step={self._step}, "
            f"target=[[{self._target_min}], [{self._target_max}]])"
        )

    def _on_accept(self, candidate: object) -> None:
        """Hook executed after a candidate is accepted into the family.

        Subclasses can override this to perform custom bookkeeping (e.g.,
        updating tangents for pseudo-arclength continuation) without having to
        reimplement the entire *run()* loop.
        """

        pass

    @abstractmethod
    def _make_stepper(self) -> _ContinuationStep:  # noqa: N802
        """Return the `StepStrategy` for this continuation run.

        Subclasses or mix-ins **must** implement this method (or assign
        ``self._stepper`` before calling ``super().__init__``) so that the
        engine knows how to predict the next candidate and possibly adapt the
        step length.  The deprecated fallback that wrapped ``_predict`` has
        been removed to enforce the new strategy-based architecture.
        """
        raise NotImplementedError










