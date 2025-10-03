"""Runtime options for continuation algorithms.

These classes define runtime tuning parameters that control HOW WELL the
continuation algorithm runs. They can vary between method calls without changing
the algorithm structure.

For compile-time configuration (algorithm structure), see config.py.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from hiten.algorithms.corrector.options import OrbitCorrectionOptions
from hiten.algorithms.types.options import _HitenBaseOptions


@dataclass(frozen=True)
class ContinuationOptions(_HitenBaseOptions):
    """Runtime options for continuation algorithms.
    
    These parameters tune HOW WELL the continuation runs and can vary
    between method calls without changing the algorithm structure.
    
    Parameters
    ----------
    target : np.ndarray
        Target parameter range for continuation. For 1D: (min, max).
        For multi-dimensional: (2, m) array where each column specifies
        (min, max) for one parameter.
    step : np.ndarray
        Initial step size(s) for continuation parameters. If scalar,
        uses same step for all parameters.
    max_members : int, default=100
        Maximum number of accepted solutions to generate.
    max_retries_per_step : int, default=50
        Maximum number of retries when correction fails at a step.
    step_min : float, default=1e-10
        Minimum allowed step size magnitude.
    step_max : float, default=1.0
        Maximum allowed step size magnitude.
    shrink_policy : callable or None, default=None
        Function to reduce step size on failure: ``shrink_policy(step) -> new_step``
        If None, uses default halving strategy.
    
    Notes
    -----
    For algorithm structure parameters like `stepper`, `state`, `getter`,
    see ContinuationConfig and OrbitContinuationConfig instead.
    
    The target and step sizes are runtime parameters because they control
    HOW WELL the continuation explores parameter space, not WHAT algorithm
    is used.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Default options
    >>> options = ContinuationOptions(
    ...     target=np.array([0.0, 1.0]),
    ...     step=np.array([0.01])
    ... )
    >>> 
    >>> # More conservative stepping
    >>> careful_options = options.merge(
    ...     step_min=1e-12,
    ...     max_retries_per_step=100
    ... )
    >>> 
    >>> # Larger family
    >>> large_options = options.merge(max_members=500)
    """
    target: np.ndarray = None
    step: np.ndarray = None
    max_members: int = 100
    max_retries_per_step: int = 50
    step_min: float = 1e-10
    step_max: float = 1.0
    shrink_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        """Post-initialization hook with target/step normalization."""
        # Normalize target to shape (2, m)
        if self.target is not None:
            target_arr = np.asarray(self.target, dtype=float)
            if target_arr.ndim == 1:
                if target_arr.size != 2:
                    raise ValueError(
                        "target must be (min,max) for 1-D or (2,m) for multi-D continuation"
                    )
                target_arr = target_arr.reshape(2, 1)
            elif not (target_arr.ndim == 2 and target_arr.shape[0] == 2):
                raise ValueError("target must be array-like shaped (2,) or (2,m)")

            # Ensure row 0 is min and row 1 is max component-wise
            target_min = np.minimum(target_arr[0], target_arr[1])
            target_max = np.maximum(target_arr[0], target_arr[1])
            target_norm = np.stack((target_min, target_max), axis=0)
            object.__setattr__(self, "target", target_norm)

        # Normalize step to shape (m,)
        if self.step is not None:
            step_arr = np.asarray(self.step, dtype=float)
            if self.target is not None:
                m = self.target.shape[1]
                if step_arr.ndim == 0:
                    step_arr = np.full(m, float(step_arr))
                elif step_arr.ndim == 1:
                    if step_arr.size == 1:
                        step_arr = np.full(m, float(step_arr[0]))
                    elif step_arr.size != m:
                        raise ValueError(
                            "step length does not match number of continuation "
                            "parameters (columns of target)"
                        )
                else:
                    raise ValueError("step must be scalar or 1-D array")
            object.__setattr__(self, "step", step_arr.astype(float))

        # Call parent validation
        super().__post_init__()

    def _validate(self) -> None:
        """Validate the options."""
        if self.target is None:
            raise ValueError("target must be provided")
        if self.step is None:
            raise ValueError("step must be provided")
            
        if not isinstance(self.max_members, int) or self.max_members <= 0:
            raise ValueError("max_members must be a positive integer")
        if not isinstance(self.max_retries_per_step, int) or self.max_retries_per_step < 0:
            raise ValueError("max_retries_per_step must be a non-negative integer")

        if not (isinstance(self.step_min, float) and self.step_min > 0.0):
            raise ValueError("step_min must be a positive float")
        if not (isinstance(self.step_max, float) and self.step_max > self.step_min):
            raise ValueError("step_max must be a float > step_min")

        # Validate step magnitudes against bounds (preserve sign)
        step_mag = np.abs(self.step)
        if np.any(step_mag < self.step_min) or np.any(step_mag > self.step_max):
            raise ValueError("each |step| must satisfy step_min <= |step| <= step_max")


@dataclass(frozen=True)
class OrbitContinuationOptions(ContinuationOptions):
    """Runtime options for periodic orbit continuation.
    
    Extends ContinuationOptions with orbit-specific runtime parameters.
    
    Parameters
    ----------
    extra_params : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions` or None, default=None
        Additional keyword arguments passed to orbit.correct() at runtime.
        Common keys include tolerances, maximum iterations, etc.
        These are runtime tuning parameters for the correction step.
    target : np.ndarray
        Target parameter range.
    step : np.ndarray
        Initial step size(s).
    max_members : int, default=100
        Maximum number of solutions.
    max_retries_per_step : int, default=50
        Maximum retries per step.
    step_min : float, default=1e-10
        Minimum step size.
    step_max : float, default=1.0
        Maximum step size.
    shrink_policy : callable or None, default=None
        Step shrinking policy.
    
    Notes
    -----
    For algorithm structure parameters like `state`, `getter`, `stepper`,
    see OrbitContinuationConfig instead.
    """
    extra_params: OrbitCorrectionOptions = field(default_factory=OrbitCorrectionOptions)

    def _validate(self) -> None:
        """Validate the options."""
        super()._validate()
        if not isinstance(self.extra_params, OrbitCorrectionOptions):
            raise ValueError("extra_params must be a OrbitCorrectionOptions")

