"""Predict-correct continuation backend implementation."""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.stepping.support import (
    _ContinuationStepSupport, _VectorSpaceSecantSupport)
from hiten.algorithms.continuation.stepping.base import _ContinuationStepBase


class _PredictorCorrectorContinuationBackend(_ContinuationBackend):
    """Implement a predict-correct continuation backend."""

    def __init__(
        self,
        *,
        stepper_factory: Callable[[
            _ContinuationStepSupport | None
        ], "_ContinuationStepBase"] = None,
        support_factory: Callable[[], _ContinuationStepSupport]= None,
    ) -> None:
        super().__init__(
            stepper_factory=stepper_factory,
            support_factory=support_factory,
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
        stepper_fn: Callable,
        predictor_fn: Callable[[object, np.ndarray], np.ndarray],
        parameter_getter: Callable[[np.ndarray], np.ndarray],
        corrector: Callable[[np.ndarray], tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict]],
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
        
        stepper = self._stepper_factory(
            stepper_fn,
            support_obj,
            seed_repr,
            step,
            predictor_fn,
            step_min,
            step_max,
            shrink_policy,
        )

        family: list[np.ndarray] = [np.asarray(seed_repr, dtype=float).copy()]
        params_history: list[np.ndarray] = [np.asarray(parameter_getter(seed_repr), dtype=float).copy()]
        aux_history: list[dict] = []

        accepted_count = 1
        rejected_count = 0
        iterations = 0

        step_vec = np.asarray(step, dtype=float).copy()
        target_min = np.asarray(target[0], dtype=float)
        target_max = np.asarray(target[1], dtype=float)

        converged = False
        failed_to_continue = False
        while accepted_count < int(max_members) and not failed_to_continue:
            last = family[-1]

            attempt = 0
            while True:
                proposal = stepper.predict(last, step_vec)
                prediction = proposal.prediction
                iterations += 1
                try:
                    out = corrector(prediction)
                    corrected, res_norm, converged, *rest = out
                    aux = rest[0] if rest and isinstance(rest[0], dict) else {}
                except Exception as e:
                    converged = False
                    res_norm = np.nan
                try:
                    self.on_iteration(iterations, prediction, float(res_norm))
                except Exception:
                    pass

                if converged:
                    family.append(corrected)
                    params = np.asarray(parameter_getter(corrected), dtype=float).copy()
                    params_history.append(params)
                    accepted_count += 1
                    aux_history.append(dict(aux))
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

                    step_vec = stepper.on_accept(
                        last_solution=last,
                        new_solution=np.asarray(corrected, dtype=float),
                        step=step_vec,
                        proposal=proposal,
                    )
                    self._last_residual = float(res_norm)
                    converged = True

                    current_params = params_history[-1]
                    if np.any(current_params < target_min) or np.any(current_params > target_max):
                        break
                    break

                rejected_count += 1
                attempt += 1

                step_vec = stepper.on_reject(
                    last_solution=last,
                    step=step_vec,
                    proposal=proposal,
                )

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
            "aux": tuple(aux_history),
            "final_step": np.asarray(step_vec, dtype=float).copy(),
            "residual_norm": float(self._last_residual) if np.isfinite(self._last_residual) else float("nan"),
        }

        return family, info