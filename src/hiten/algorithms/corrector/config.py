"""Provide configuration classes for iterative correction algorithms.

This module provides the compile-time configuration classes for iterative 
correction algorithms used throughout the hiten framework. These classes 
encapsulate algorithm structure parameters that define WHAT algorithm is used.

For runtime tuning parameters (HOW WELL it runs), see CorrectionOptions in 
hiten.algorithms.types.options.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from hiten.algorithms.poincare.singlehit.backend import _y_plane_crossing
from hiten.algorithms.types.configs import CorrectionConfig


@dataclass(frozen=True)
class OrbitCorrectionConfig(CorrectionConfig):
    """Configuration for periodic orbit correction (compile-time structure).

    Extends the base correction configuration with orbit-specific structural
    parameters for constraint selection and event detection. These define
    WHAT problem is being solved, not HOW WELL it is solved.

    Parameters
    ----------
    residual_indices : tuple of int, default=()
        State components used to build the residual vector.
        Defines the structure of the correction problem.
    control_indices : tuple of int, default=()
        State components allowed to change during correction.
        Defines the structure of the correction problem.
    extra_jacobian : callable or None, default=None
        Additional Jacobian contribution function.
        Defines the structure of the correction problem.
    target : tuple of float, default=(0.0,)
        Target values for the residual components.
        Defines the problem to solve.
    event_func : callable, default=_y_plane_crossing
        Function to detect Poincare section crossings.
        Defines the problem structure.

    Notes
    -----
    For runtime tuning like `tol`, `max_attempts`, use CorrectionOptions.

    Examples
    --------
    >>> # Compile-time: Define problem structure
    >>> config = OrbitCorrectionConfig(
    ...     method="adaptive",
    ...     residual_indices=(2, 3, 4),  # z, vz, vx
    ...     control_indices=(0, 2, 4),    # x, z, vx
    ...     target=(0.0, 0.0, 0.0),
    ...     event_func=_y_plane_crossing,
    ...     finite_difference=False
    ... )
    >>> # Runtime: Tune convergence per call
    >>> from hiten.algorithms.types.options import CorrectionOptions
    >>> options = CorrectionOptions()
    """
    residual_indices: tuple[int, ...] = ()
    control_indices: tuple[int, ...] = ()
    extra_jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    target: tuple[float, ...] = (0.0,)
    event_func: Callable[..., tuple[float, np.ndarray]] = _y_plane_crossing

    def _validate(self) -> None:
        """Validate the configuration."""
        super()._validate()
        if len(self.residual_indices) != len(self.target):
            raise ValueError(
                f"Length mismatch: residual_indices has {len(self.residual_indices)} "
                f"elements but target has {len(self.target)} elements."
            )
        if not all(isinstance(i, int) and i >= 0 for i in self.residual_indices):
            raise ValueError("residual_indices must contain non-negative integers.")
        if not all(isinstance(i, int) and i >= 0 for i in self.control_indices):
            raise ValueError("control_indices must contain non-negative integers.")


@dataclass(frozen=True)
class MultipleShootingOrbitCorrectionConfig(OrbitCorrectionConfig):
    pass