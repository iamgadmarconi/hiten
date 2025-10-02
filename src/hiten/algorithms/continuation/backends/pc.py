"""Predict-correct continuation backend implementation."""

import warnings
from typing import Callable

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.stepping.support import (
    _ContinuationStepSupport, _VectorSpaceSecantSupport)
from hiten.algorithms.continuation.stepping.base import _ContinuationStepBase


class _PCContinuationBackend(_ContinuationBackend):
    """Implement a predict-correct continuation backend."""

    def __init__(
        self,
        *,
        stepper_factory: Callable[[
            _ContinuationStepSupport | None
        ], "_ContinuationStepBase"] | None = None,
        support_factory: Callable[[], _ContinuationStepSupport] | None = None,
    ) -> None:
        super().__init__(
            stepper_factory=stepper_factory,
            support_factory=support_factory or _VectorSpaceSecantSupport,
        )
        self._support = None
        self._last_residual: float = float("nan")

    def _reset_state(self) -> None:
        self._last_residual = float("nan")

    def make_step_support(self) -> _ContinuationStepSupport:
        self._support = super().make_step_support()
        return self._support

    def run(
        self,
        *,
        seed_repr: np.ndarray,
        predictor_fn: Callable[[object, np.ndarray], np.ndarray],
        parameter_getter: Callable[[np.ndarray], np.ndarray],
        corrector: Callable[[np.ndarray], tuple[np.ndarray, float, bool]],
        representation_of: Callable[[object], np.ndarray] | None,
        step: np.ndarray,
        target: np.ndarray,
        max_members: int,
        max_retries_per_step: int,
        shrink_policy: Callable[[np.ndarray], np.ndarray] | None,
        step_min: float,
        step_max: float,
    ) -> tuple[list[np.ndarray], dict]:
        self._reset_state()

        support_obj = self.make_step_support()
        
        # Seed initial tangent for secant steppers
        from hiten.algorithms.continuation.stepping.support import _SecantSupport
        if isinstance(support_obj, _SecantSupport):
            try:
                repr_seed = representation_of(seed_repr) if representation_of is not None else seed_repr
                pred0 = predictor_fn(repr_seed, np.asarray(step, dtype=float))
                diff0 = (np.asarray(pred0, dtype=float) - np.asarray(repr_seed, dtype=float)).ravel()
                norm0 = float(np.linalg.norm(diff0))
                tangent0 = None if norm0 == 0.0 else diff0 / norm0
            except Exception:
                tangent0 = None
            support_obj.seed(tangent0)
        
        stepper = self._stepper_factory(predictor_fn, representation_of, support_obj)

        family: list[np.ndarray] = [np.asarray(seed_repr, dtype=float).copy()]
        params_history: list[np.ndarray] = [np.asarray(parameter_getter(seed_repr), dtype=float).copy()]

        accepted_count = 1
        rejected_count = 0
        iterations = 0

        step_vec = np.asarray(step, dtype=float).copy()
        target_min = np.asarray(target[0], dtype=float)
        target_max = np.asarray(target[1], dtype=float)

        def _clamp_step(vec: np.ndarray) -> np.ndarray:
            mag = np.clip(np.abs(vec), step_min, step_max)
            return np.sign(vec) * mag

        converged = False
        failed_to_continue = False
        while accepted_count < int(max_members) and not failed_to_continue:
            last = family[-1]

            attempt = 0
            while True:
                proposal = stepper.predict(last, step_vec)
                prediction = proposal.prediction
                step_hint = proposal.step_hint if proposal.step_hint is not None else step_vec
                iterations += 1
                try:
                    corrected, res_norm, converged = corrector(prediction)
                except Exception as e:
                    converged = False
                    res_norm = np.nan
                    warnings.warn(f"Continuation correction failed: {str(e)[:100]}", stacklevel=2)

                try:
                    self.on_iteration(iterations, prediction, float(res_norm))
                except Exception:
                    pass

                if converged:
                    family.append(corrected)
                    params = np.asarray(parameter_getter(corrected), dtype=float).copy()
                    params_history.append(params)
                    accepted_count += 1
                    try:
                        self.on_accept(corrected, iterations=iterations, residual_norm=float(res_norm))
                    except Exception:
                        pass

                    if support_obj is not None and len(family) >= 2:
                        prev = family[-2]
                        curr = family[-1]
                        try:
                            support_obj.on_accept(np.asarray(prev, dtype=float), np.asarray(curr, dtype=float))
                        except Exception:
                            pass

                    next_step = stepper.on_accept(
                        last_solution=last,
                        new_solution=np.asarray(corrected, dtype=float),
                        step=step_vec,
                        proposal=proposal,
                    )
                    step_vec = _clamp_step(np.asarray(next_step if next_step is not None else step_hint, dtype=float))
                    self._last_residual = float(res_norm)
                    converged = True

                    current_params = params_history[-1]
                    if np.any(current_params < target_min) or np.any(current_params > target_max):
                        break
                    break

                rejected_count += 1
                attempt += 1

                if shrink_policy is not None:
                    try:
                        new_step = np.asarray(shrink_policy(step_vec), dtype=float)
                    except Exception:
                        new_step = step_vec * 0.5
                else:
                    new_step = step_vec * 0.5

                next_step = stepper.on_reject(
                    last_solution=last,
                    step=step_vec,
                    proposal=proposal,
                )
                if next_step is not None:
                    new_step = np.asarray(next_step, dtype=float)

                step_vec = _clamp_step(new_step)

                if support_obj is not None:
                    try:
                        support_obj.on_reject(np.asarray(last, dtype=float), step_vec)
                    except Exception:
                        pass

                if attempt > int(max_retries_per_step):
                    try:
                        self.on_failure(prediction, iterations=iterations, residual_norm=float(res_norm))
                    except Exception:
                        pass
                    converged = False
                    failed_to_continue = True
                    break

            if accepted_count >= int(max_members):
                break

        info = {
            "accepted_count": int(accepted_count),
            "rejected_count": int(rejected_count),
            "iterations": int(iterations),
            "parameter_values": tuple(np.asarray(p, dtype=float).copy() for p in params_history),
            "final_step": np.asarray(step_vec, dtype=float).copy(),
            "residual_norm": float(self._last_residual) if np.isfinite(self._last_residual) else float("nan"),
        }

        return family, info