"""Predict-correct continuation backend implementation."""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend


class _PCContinuationBackend(_ContinuationBackend):
    """Implement a predict-correct continuation backend.

    This backend drives a simple predict-correct-accept loop using a
    user-provided predictor and corrector, adapting the step size based
    on success/failure and stopping when either the member limit is
    reached or parameters exit the configured bounds.
    """

    def solve(
        self,
        *,
        seed_repr: np.ndarray,
        stepper: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        parameter_getter: Callable[[np.ndarray], np.ndarray],
        corrector: Callable[[np.ndarray], tuple[np.ndarray, float, bool]],
        representation_of: Callable[[np.ndarray], np.ndarray] | None,
        set_tangent: Callable[[np.ndarray | None], None] | None,
        step: np.ndarray,
        target: np.ndarray,
        max_members: int,
        max_retries_per_step: int,
        shrink_policy: Callable[[np.ndarray], np.ndarray] | None,
        step_min: float,
        step_max: float,
    ) -> tuple[list[np.ndarray], dict]:
        """Run continuation using purely numerical inputs and callables.
        
        This backend implements a simple predict-correct-accept loop using a
        user-provided predictor and corrector, adapting the step size based
        on success/failure and stopping when either the member limit is
        reached or parameters exit the configured bounds.

        Parameters
        ----------
        seed_repr : np.ndarray
            Numerical representation of the seed solution.
        stepper : Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            Stepper function that takes the last representation and step vector and returns the next prediction and step hint.
        parameter_getter : Callable[[np.ndarray], np.ndarray]
            Function that takes a representation and returns the parameter values.
        corrector : Callable[[np.ndarray], tuple[np.ndarray, float, bool]]
            Function that takes a prediction and returns the corrected representation, residual norm, and convergence flag.
        representation_of : Callable[[np.ndarray], np.ndarray] | None
            Function that takes a representation and returns the numerical representation.
        set_tangent : Callable[[np.ndarray | None], None] | None
            Function that takes a tangent vector and sets it.
        step : np.ndarray
            Initial step vector.
        target : np.ndarray
            Target parameter range.
        max_members : int
            Maximum number of accepted members.
        max_retries_per_step : int
            Maximum number of retries per step.
        shrink_policy : Callable[[np.ndarray], np.ndarray] | None
            Function that takes a step vector and returns a shrunk step vector.
        step_min : float
            Minimum step size.
        step_max : float
            Maximum step size.
        
        Returns
        -------
        family : list[np.ndarray]
            List of accepted member representations.
        info : dict
            Dictionary containing the accepted count, rejected count, iterations, parameter values, and final step vector.
        """
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

        while accepted_count < int(max_members):
            last = family[-1]

            attempt = 0
            while True:
                # Compute a fresh prediction for the current step size
                prediction, step_hint = stepper(last, step_vec)
                if isinstance(step_hint, np.ndarray) and step_hint.shape == step_vec.shape:
                    # On success we will adopt step_hint; for attempts we keep current
                    proposed_step_after_success = _clamp_step(step_hint.astype(float))
                else:
                    proposed_step_after_success = step_vec
                iterations += 1
                try:
                    corrected, res_norm, converged = corrector(prediction)
                except Exception:
                    converged = False
                    res_norm = np.nan

                try:
                    self.on_iteration(iterations, prediction, float(res_norm))
                except Exception:
                    pass

                if converged:
                    family.append(corrected)
                    params_history.append(np.asarray(parameter_getter(corrected), dtype=float).copy())
                    accepted_count += 1
                    try:
                        self.on_accept(corrected, iterations=iterations, residual_norm=float(res_norm))
                    except Exception:
                        pass

                    # Always take stepper's hint on success
                    step_vec = proposed_step_after_success

                    # Optional: update tangent if interfaces provided (secant)
                    if representation_of is not None and set_tangent is not None and len(family) >= 2:
                        # Compute secant in representation space
                        r_prev = representation_of(family[-2])
                        r_curr = representation_of(family[-1])
                        diff = (r_curr - r_prev).ravel()
                        norm = float(np.linalg.norm(diff))
                        set_tangent(None if norm == 0.0 else diff / norm)

                    current_params = params_history[-1]
                    if np.any(current_params < target_min) or np.any(current_params > target_max):
                        break
                    break

                rejected_count += 1
                attempt += 1
                # On failure, use injected shrink policy if provided, else halve, then retry with new prediction
                if shrink_policy is not None:
                    try:
                        step_vec = _clamp_step(np.asarray(shrink_policy(step_vec), dtype=float))
                    except Exception:
                        step_vec = _clamp_step(step_vec * 0.5)
                else:
                    step_vec = _clamp_step(step_vec * 0.5)

                if attempt > int(max_retries_per_step):
                    try:
                        self.on_failure(prediction, iterations=iterations, residual_norm=float(res_norm))
                    except Exception:
                        pass
                    accepted_count = max_members
                    break

            if accepted_count >= int(max_members):
                break

        info = {
            "accepted_count": int(accepted_count),
            "rejected_count": int(rejected_count),
            "iterations": int(iterations),
            "parameter_values": tuple(params_history),
            "final_step": step_vec.copy(),
        }

        try:
            last_repr = family[-1]
            res_norm_final = float(np.linalg.norm(params_history[-1], ord=2))
            self.on_success(last_repr, iterations=int(iterations), residual_norm=res_norm_final)
        except Exception:
            pass

        return family, info