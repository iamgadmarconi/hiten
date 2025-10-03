"""Predict-correct continuation backend implementation."""

from typing import Any, Callable

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.types import (
    ContinuationBackendRequest,
    ContinuationBackendResponse,
)
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
        request: ContinuationBackendRequest,
    ) -> ContinuationBackendResponse:
        self._reset_state()

        support_obj = self.make_step_support()

        stepper = self._stepper_factory(
            request.stepper_fn,
            support_obj,
            request.seed_repr,
            request.step,
            request.predictor_fn,
            request.step_min,
            request.step_max,
            request.shrink_policy,
        )

        family: list[np.ndarray] = [np.asarray(request.seed_repr, dtype=float).copy()]
        params_history: list[np.ndarray] = [
            np.asarray(request.parameter_getter(request.seed_repr), dtype=float).copy()
        ]
        aux_history: list[dict] = []

        accepted_count = 1
        rejected_count = 0
        iterations = 0

        step_vec = np.asarray(request.step, dtype=float).copy()
        target_min = np.asarray(request.target[0], dtype=float)
        target_max = np.asarray(request.target[1], dtype=float)

        converged = False
        failed_to_continue = False
        while accepted_count < int(request.max_members) and not failed_to_continue:
            last = family[-1]

            attempt = 0
            while True:
                proposal = stepper.predict(last, step_vec)
                prediction = proposal.prediction
                iterations += 1
                try:
                    out = request.corrector(prediction)
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
                    params = np.asarray(request.parameter_getter(corrected), dtype=float).copy()
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

                if attempt > int(request.max_retries_per_step):
                    try:
                        self.on_failure(prediction, iterations=iterations, residual_norm=float(res_norm))
                    except Exception:
                        pass
                    converged = False
                    failed_to_continue = True
                    break

            if accepted_count >= int(request.max_members):
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
        return ContinuationBackendResponse(family_repr=family, info=info)