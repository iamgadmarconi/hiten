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
    """Configuration for multiple shooting correction (compile-time structure).

    Extends OrbitCorrectionConfig with multiple shooting-specific structural
    parameters for patch management, continuity enforcement, and boundary
    condition handling.

    Multiple shooting divides a trajectory into N segments (patches) and
    treats each patch's initial state as an independent variable. This
    approach is more robust than single shooting for long-period orbits,
    unstable systems, or poor initial guesses.

    Parameters
    ----------
    patch_strategy : {"uniform", "adaptive", "manual"}, default="uniform"
        Strategy for distributing patch points along the trajectory.
        This is a structural algorithm choice.
        
        - "uniform": Evenly spaced in time (simplest, most common)
        - "adaptive": Concentrate patches near instabilities (future)
        - "manual": User-provided times via `manual_patch_times`
        
    manual_patch_times : tuple[float, ...] or None, default=None
        Explicit patch point times for `patch_strategy="manual"`.
        Defines the problem structure.
    continuity_indices : tuple[int, ...], default=()
        State components to enforce continuity at patch junctions.
        Defines the problem structure.
    enforce_all_continuity : bool, default=True
        If True, enforces continuity on all `control_indices`.
        Defines the problem structure.
    boundary_only_indices : tuple[int, ...], default=()
        State components with boundary conditions but no continuity
        enforcement at internal patches.
        Defines the problem structure.
    use_sparse_jacobian : bool, default=False
        If True, assemble the Jacobian matrix directly in sparse format.
        This is a structural algorithm choice for large problems.

    Notes
    -----
    For runtime tuning like `n_patches`, `tol`, `max_attempts`, use 
    MultipleShootingCorrectionOptions instead.

    The parameter vector for multiple shooting has dimension:
    
    .. math::
        
        n_{\\text{params}} = n_{\\text{patches}} \\times n_{\\text{control}}

    The residual vector has dimension:
    
    .. math::
        
        n_{\\text{residual}} = (n_{\\text{patches}} - 1) \\times n_{\\text{continuity}} + n_{\\text{boundary}}

    Examples
    --------
    >>> # Compile-time: Define algorithm structure
    >>> config = MultipleShootingOrbitCorrectionConfig(
    ...     method="adaptive",
    ...     residual_indices=(1, 2, 3, 4, 5),
    ...     control_indices=(0, 1, 2, 3, 4, 5),
    ...     target=(0.0, 0.0, 0.0, 0.0, 0.0),
    ...     patch_strategy="uniform",
    ...     use_sparse_jacobian=False
    ... )
    >>> # Runtime: Tune per call
    >>> from hiten.algorithms.corrector.options import MultipleShootingCorrectionOptions
    >>> options = MultipleShootingCorrectionOptions(n_patches=7)

    See Also
    --------
    :class:`OrbitCorrectionConfig`
        Base configuration for orbit correction (single shooting).
    :class:`CorrectionConfig`
        Base configuration for all correction algorithms.
    """
    patch_strategy: Literal["uniform", "adaptive", "manual"] = "uniform"
    manual_patch_times: Optional[tuple[float, ...]] = None
    continuity_indices: tuple[int, ...] = ()
    enforce_all_continuity: bool = True
    boundary_only_indices: tuple[int, ...] = ()
    use_sparse_jacobian: bool = False
    # Optional: specify which control indices the extra_jacobian columns map to
    # If None, extra_jacobian is assumed to already match the full set of
    # control_indices columns. When provided, the extra_jacobian matrix must
    # have shape (len(residual_indices_at_boundary), len(extra_jacobian_control_indices)),
    # and will be embedded into the full control space at these columns.
    extra_jacobian_control_indices: Optional[tuple[int, ...]] = None

    def _validate(self) -> None:
        """Validate the configuration."""
        super()._validate()
        if self.patch_strategy not in ["uniform", "adaptive", "manual"]:
            raise ValueError(
                f"Invalid patch_strategy: {self.patch_strategy}. "
                "Must be 'uniform', 'adaptive', or 'manual'."
            )
        if self.patch_strategy == "manual" and self.manual_patch_times is None:
            raise ValueError(
                "manual_patch_times must be provided when patch_strategy='manual'."
            )
        if self.manual_patch_times is not None:
            if len(self.manual_patch_times) < 2:
                raise ValueError(
                    "manual_patch_times must have at least 2 elements (start and end)."
                )
        if self.extra_jacobian is not None and self.extra_jacobian_control_indices is not None:
            # basic sanity: indices non-negative
            if not all(isinstance(i, int) and i >= 0 for i in self.extra_jacobian_control_indices):
                raise ValueError("extra_jacobian_control_indices must contain non-negative integers.")
            # they should be a subset of control_indices
            missing = [i for i in self.extra_jacobian_control_indices if i not in self.control_indices]
            if missing:
                raise ValueError(
                    "extra_jacobian_control_indices must be a subset of control_indices; "
                    f"missing from control_indices: {missing}"
                )
