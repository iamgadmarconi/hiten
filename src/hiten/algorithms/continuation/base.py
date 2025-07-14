"""
hiten.algorithms.continuation.base
====================================

Abstract predictor-corrector continuation engine for families of periodic
orbits in the Circular Restricted Three-Body Problem (CR3BP).

The class provides the *infrastructure* - bookkeeping, generic loop, step-size
control, logging - but purposely delegates the actual *prediction strategy* to
sub-classes via the :pyfunc:`_predict` hook.

Optionally override ``_update_step`` and ``_stop_condition`` for fancier
strategies (pseudo-arclength, MIL continuation, ...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Sequence

import numpy as np

from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger


class _PeriodicOrbitContinuationEngine(ABC):
    """Generic predictor-corrector engine for periodic-orbit continuation.

    Parameters
    ----------
    initial_orbit : PeriodicOrbit
        *Seed* orbit **already** corrected (i.e. `period` is not *None*).
    parameter_getter : Callable[[PeriodicOrbit], "np.ndarray | float"]
        Function that extracts the continuation parameter from an orbit
        instance (e.g. ``lambda o: o.amplitude_z``).
    target : tuple[float, float]
        Inclusive lower/upper bounds that delimit the continuation range.  The
        engine stops once the parameter value leaves this interval.
    step : float, default 1e-4
        Initial step size for the predictor (sign included).
    corrector_kwargs : dict, optional
        Keyword arguments forwarded to
        :pyfunc:`PeriodicOrbit.correct`.
    max_orbits : int, default 256
        Hard limit on the number of family members to generate (safety brake).
    """

    def __init__(
        self,
        *,
        initial_orbit: PeriodicOrbit,
        parameter_getter: Callable[[PeriodicOrbit], "np.ndarray | float"],
        target: Sequence[Sequence[float] | float],
        step: float | Sequence[float] = 1e-4,
        corrector_kwargs: dict | None = None,
        max_orbits: int = 256,
    ) -> None:
        if not isinstance(initial_orbit, PeriodicOrbit):
            raise TypeError("initial_orbit must be a PeriodicOrbit instance")
        if initial_orbit.period is None:
            raise ValueError(
                "initial_orbit must be corrected before launching continuation "
                "(period attribute is None)."
            )
        # normalise *target* to 2-by-m array
        target_arr = np.asarray(target, dtype=float)
        if target_arr.ndim == 1:
            # classic (min,max) specification for 1-D continuation
            if target_arr.size != 2:
                raise ValueError("target must be (min,max) for 1-D or (2,m) for multi-D continuation")
            target_arr = target_arr.reshape(2, 1)  #  -> 2x1 matrix
        elif target_arr.ndim == 2 and target_arr.shape[0] == 2:
            # Already in correct layout  (2,m)
            pass
        else:
            raise ValueError("target must be iterable shaped (2,) or (2,m)")

        self._initial_orbit = initial_orbit
        self._orbit_class = type(initial_orbit)
        self._libration_point = initial_orbit.libration_point
        self._getter = parameter_getter

        current_param = np.asarray(self._getter(initial_orbit), dtype=float)
        if current_param.ndim == 0:
            current_param = current_param.reshape(1)

        # convert step to array now and broadcast
        step_arr = np.asarray(step, dtype=float)
        if step_arr.size == 1:
            step_arr = np.full_like(current_param, float(step_arr))
        elif step_arr.size != current_param.size:
            raise ValueError("step length does not match number of continuation parameters")

        # Broadcast target rows to match parameter dimensionality
        if target_arr.shape[1] != current_param.size:
            if target_arr.shape[1] == 1:
                target_arr = np.repeat(target_arr, current_param.size, axis=1)
            else:
                raise ValueError("target dimensionality mismatch with continuation parameter")

        self._target_min = np.minimum(target_arr[0], target_arr[1])
        self._target_max = np.maximum(target_arr[0], target_arr[1])

        for i in range(current_param.size):
            if (current_param[i] < self._target_min[i] and step_arr[i] < 0) or (
                current_param[i] > self._target_max[i] and step_arr[i] > 0
            ):
                step_arr[i] = -step_arr[i]
        self._step = step_arr.astype(float)

        self._corrector_kwargs = corrector_kwargs or {}
        self._max_orbits = int(max_orbits)

        self._family: List[PeriodicOrbit] = [initial_orbit]
        self._param_history: List[np.ndarray] = [current_param.copy()]

        logger.info(
            "Continuation initialised: parameter=%s, target=[%s - %s], step=%s, max_orbits=%d",
            current_param,
            self._target_min,
            self._target_max,
            self._step,
            self._max_orbits,
        )

    @property
    def family(self) -> Sequence[PeriodicOrbit]:
        """Read-only view of the generated orbit list (first element is the seed)."""
        return tuple(self._family)

    @property
    def parameter_values(self) -> Sequence[np.ndarray]:
        """Parameter value associated with each family member."""
        return tuple(self._param_history)

    def run(self) -> List[PeriodicOrbit]:
        """Run the predictor-corrector loop until the stop criterion is met.

        Returns
        -------
        list[PeriodicOrbit]
            The generated family, *including* the initial orbit (index 0).
        """
        logger.info("Starting continuation loop ...")
        attempts_at_current_step = 0
        while not self._stop_condition():
            if len(self._family) >= self._max_orbits:
                logger.warning("Reached max_orbits=%d, terminating continuation.", self._max_orbits)
                break

            last_orbit = self._family[-1]
            predicted_state = self._predict(last_orbit, self._step)
            trial_orbit = self._instantiate_orbit(predicted_state)

            try:
                trial_orbit.correct(**self._corrector_kwargs)
            except Exception as exc:
                logger.debug(
                    "Correction failed at step %s (attempt %d): %s",
                    self._step,
                    attempts_at_current_step + 1,
                    exc,
                    exc_info=exc,
                )
                self._step = self._update_step(self._step, success=False)
                attempts_at_current_step += 1
                if attempts_at_current_step > 10:
                    logger.error("Too many failed attempts at current step; aborting continuation.")
                    break
                continue  # retry with reduced step

            attempts_at_current_step = 0  # reset counter
            self._family.append(trial_orbit)
            param_val = self._getter(trial_orbit)
            self._param_history.append(param_val.copy())
            logger.info("Accepted orbit #%d, parameter=%s", len(self._family) - 1, param_val)

            # Adapt step for next iteration
            self._step = self._update_step(self._step, success=True)

        logger.info("Continuation finished - generated %d orbits.", len(self._family))
        return self._family

    @abstractmethod
    def _predict(self, last_orbit: PeriodicOrbit, step: float) -> np.ndarray:
        """Return a 6-component state vector predicted for the next orbit."""
        raise NotImplementedError

    def _update_step(self, current_step: np.ndarray, *, success: bool) -> np.ndarray:
        """Simple adaptive strategy applied component-wise. Preserve sign while clamping magnitude."""
        factor = 2.0 if success else 0.5
        new_step = current_step * factor
        clipped_mag = np.clip(np.abs(new_step), 1e-10, 1.0)
        return np.sign(new_step) * clipped_mag

    def _stop_condition(self) -> bool:
        """Default stop condition: parameter value left target interval."""
        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    def _instantiate_orbit(self, state: np.ndarray) -> PeriodicOrbit:
        """Create a new orbit instance from the provided 6-state vector."""
        return self._orbit_class(
            libration_point=self._libration_point,
            initial_state=state,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_orbits={len(self._family)}, "
            f"step={self._step}, target=[{self._target_min}, {self._target_max}])"
        )

    @staticmethod
    def _clamp_step(step_value: float, reference_value: float = 1.0, min_relative: float = 1e-6, min_absolute: float = 1e-8) -> float:
        """
        Apply robust step clamping that preserves sign and allows adaptive reduction.
        
        This method prevents pathological step values while respecting the continuation
        engine's adaptive step reduction logic. It scales the minimum step based on
        the reference value to handle different parameter magnitudes appropriately.
        
        Parameters
        ----------
        step_value : float
            The proposed step value (can be positive or negative)
        reference_value : float, optional
            Reference value to scale the minimum step (e.g., current state component).
            Default is 1.0.
        min_relative : float, optional
            Minimum step as a fraction of the reference value. Default is 1e-6.
        min_absolute : float, optional
            Absolute minimum step size. Default is 1e-8.
            
        Returns
        -------
        float
            Clamped step value that preserves sign and respects minimum bounds
        """
        if step_value == 0:
            return min_absolute
            
        # Compute adaptive minimum based on reference value
        ref_magnitude = abs(reference_value)
        if ref_magnitude > min_absolute:
            min_step = max(min_absolute, ref_magnitude * min_relative)
        else:
            min_step = min_absolute
            
        # Apply sign-aware clamping
        if abs(step_value) < min_step:
            return np.sign(step_value) * min_step
        else:
            return step_value

    @staticmethod
    def _clamp_scale(scale_value: float, min_scale: float = 1e-3, max_scale: float = 1e3) -> float:
        """
        Apply robust scaling factor clamping for multiplicative predictors.
        
        This method ensures scaling factors remain within reasonable bounds to
        prevent pathological orbit predictions while allowing adaptive step reduction.
        
        Parameters
        ----------
        scale_value : float
            The proposed scaling factor
        min_scale : float, optional
            Minimum allowed scaling factor. Default is 1e-3.
        max_scale : float, optional
            Maximum allowed scaling factor. Default is 1e3.
            
        Returns
        -------
        float
            Clamped scaling factor within reasonable bounds
        """
        return np.clip(scale_value, min_scale, max_scale)

