"""Provide configuration classes for domain-specific continuation algorithms.

This module provides configuration classes for domain-specific continuation
algorithms. These classes encapsulate the parameters required for different
types of continuation methods (natural parameter, pseudo-arclength, etc.).
"""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from hiten.algorithms.utils.types import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


@dataclass(frozen=True, slots=True)
class _ContinuationConfig:
    """Define configuration parameters for continuation algorithms."""
    target: np.ndarray
    step: np.ndarray
    max_members: int
    max_retries_per_step: int
    step_min: float
    step_max: float
    shrink_policy: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        # Normalize target to shape (2, m)
        target_arr = np.asarray(self.target, dtype=float)
        if target_arr.ndim == 1:
            if target_arr.size != 2:
                raise ValueError("target must be (min,max) for 1-D or (2,m) for multi-D continuation")
            target_arr = target_arr.reshape(2, 1)
        elif not (target_arr.ndim == 2 and target_arr.shape[0] == 2):
            raise ValueError("target must be array-like shaped (2,) or (2,m)")

        # Ensure row 0 is min and row 1 is max component-wise
        target_min = np.minimum(target_arr[0], target_arr[1])
        target_max = np.maximum(target_arr[0], target_arr[1])
        target_norm = np.stack((target_min, target_max), axis=0)

        # Normalize step to shape (m,)
        step_arr = np.asarray(self.step, dtype=float)
        m = target_norm.shape[1]
        if step_arr.ndim == 0:
            step_arr = np.full(m, float(step_arr))
        elif step_arr.ndim == 1:
            if step_arr.size == 1:
                step_arr = np.full(m, float(step_arr[0]))
            elif step_arr.size != m:
                raise ValueError("step length does not match number of continuation parameters (columns of target)")
        else:
            raise ValueError("step must be scalar or 1-D array")

        # Validate counts
        if not isinstance(self.max_members, int) or self.max_members <= 0:
            raise ValueError("max_members must be a positive integer")
        if not isinstance(self.max_retries_per_step, int) or self.max_retries_per_step < 0:
            raise ValueError("max_retries_per_step must be a non-negative integer")

        if not (isinstance(self.step_min, float) and self.step_min > 0.0):
            raise ValueError("step_min must be a positive float")
        if not (isinstance(self.step_max, float) and self.step_max > self.step_min):
            raise ValueError("step_max must be a float > step_min")

        # Validate step magnitudes against bounds (preserve sign)
        step_mag = np.abs(step_arr)
        if np.any(step_mag < self.step_min) or np.any(step_mag > self.step_max):
            raise ValueError("each |step| must satisfy step_min <= |step| <= step_max")

        # Assign normalized arrays back (frozen dataclass requires object.__setattr__)
        object.__setattr__(self, "target", target_norm)
        object.__setattr__(self, "step", step_arr.astype(float))


@dataclass(frozen=True, slots=True)
class _OrbitContinuationConfig(_ContinuationConfig):
    """Define configuration parameters for periodic orbit continuation.

    This dataclass encapsulates configuration options specific to
    periodic orbit continuation, including state initialization,
    parameter extraction, and additional correction settings.

    Parameters
    ----------
    state : :class:`~hiten.algorithms.utils.types.SynodicState` or None
        Initial state for orbit construction. If None, uses default
        state from the orbit class.
    amplitude : bool, default False
        Whether to use amplitude-based continuation instead of
        natural parameter continuation.
    getter : callable or None
        Function to extract continuation parameter from periodic orbit.
        Should take a :class:`~hiten.system.orbits.base.PeriodicOrbit` and return float.
        If None, uses default parameter extraction.
    extra_params : dict or None
        Additional parameters passed to orbit correction methods.
        Common keys include tolerances, maximum iterations, etc.
    """
    state: SynodicState | None = None
    amplitude: bool = False
    getter: Callable[["PeriodicOrbit"], float] | None = None
    extra_params: dict | None = None
    stepper: Literal["natural", "secant"] = "natural"
