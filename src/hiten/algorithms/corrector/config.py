"""Provide configuration classes for iterative correction algorithms.

This module provides the configuration classes for iterative correction
algorithms used throughout the hiten framework. These classes encapsulate
the parameters for the correction algorithms and are used to configure
the correction algorithms.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Optional

import numpy as np

from hiten.algorithms.poincare.singlehit.backend import _y_plane_crossing

if TYPE_CHECKING:
    from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn


class _LineSearchConfig(NamedTuple):
    """Define configuration parameters for Armijo line search.
    
    Parameters
    ----------
    norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` or None, default=None
        Function to compute residual norm. Uses L2 norm if None.
    residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn` or None, default=None
        Function to compute residual vector. Must be provided.
    jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` or None, default=None
        Jacobian function (currently unused).
    max_delta : float, default=1e-2
        Maximum allowed step size (infinity norm).
    alpha_reduction : float, default=0.5
        Factor to reduce step size in backtracking.
    min_alpha : float, default=1e-4
        Minimum step size before giving up.
    armijo_c : float, default=0.1
        Armijo parameter for sufficient decrease condition.
    """
    norm_fn: Optional["NormFn"] = None
    residual_fn: Optional["ResidualFn"] = None
    jacobian_fn: Optional["JacobianFn"] = None
    max_delta: float = 1e-2
    alpha_reduction: float = 0.5
    min_alpha: float = 1e-4
    armijo_c: float = 0.1


@dataclass(frozen=True, slots=True)
class _BaseCorrectionConfig:
    """Define a base configuration class for correction algorithm parameters.

    This dataclass encapsulates the common configuration parameters used
    by correction algorithms throughout the hiten framework. It provides
    sensible defaults while allowing customization for specific problem
    requirements and numerical considerations.

    The configuration is designed to be immutable (frozen) for thread safety
    and to prevent accidental modification during algorithm execution. The
    slots optimization reduces memory overhead when many configuration
    objects are created.

    Parameters
    ----------
    max_attempts : int, default=50
        Maximum number of Newton iterations to attempt before declaring
        convergence failure. This prevents infinite loops in cases where
        the algorithm fails to converge.
    tol : float, default=1e-10
        Convergence tolerance for the residual norm. The algorithm terminates
        successfully when the norm of the residual falls below this value.
        Should be chosen based on the required precision and numerical
        conditioning of the problem.
    max_delta : float, default=1e-2
        Maximum allowed infinity norm of Newton steps. This serves as a
        safeguard against excessively large steps that could cause numerical
        overflow or move far from the solution. Particularly important for
        poorly conditioned problems or bad initial guesses.
    line_search_config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig`, bool, or None, default=True
        Configuration for line search behavior:
        - True: Enable line search with default parameters
        - False or None: Disable line search (use full Newton steps)
        - :class:`~hiten.algorithms.corrector.config._LineSearchConfig`: Enable line search with custom parameters
        Line search improves robustness for challenging problems at the
        cost of additional function evaluations.
    finite_difference : bool, default=False
        Force finite-difference approximation of Jacobians even when
        analytic Jacobians are available. Useful for debugging, testing,
        or when analytic Jacobians are suspected to be incorrect.
        Generally results in slower convergence but can be more robust.
    fd_step : float, default=1e-8
        Finite-difference step size used when computing Jacobians via
        central differences. Scaled internally per-parameter by
        max(1, |x[i]|) to maintain relative step size.
    method : str, default="adaptive"
        Integration method for trajectory computation.
    order : int, default=8
        Integration order for numerical methods.
    steps : int, default=500
        Number of integration steps.
    forward : int, default=1
        Integration direction (1 for forward, -1 for backward).

    Notes
    -----
    The default parameters are chosen to work well for typical problems
    in astrodynamics and dynamical systems, particularly in the context
    of the Circular Restricted Three-Body Problem (CR3BP).
    """
    max_attempts: int = 50
    tol: float = 1e-10
    max_delta: float = 1e-2
    line_search_config: _LineSearchConfig | bool | None = True
    finite_difference: bool = False
    fd_step: float = 1e-8
    method: Literal["fixed", "adaptive", "symplectic"] = "adaptive"
    order: int = 8
    steps: int = 500
    forward: int = 1


@dataclass(frozen=True, slots=True)
class _OrbitCorrectionConfig(_BaseCorrectionConfig):
    """Define a configuration for periodic orbit correction.

    Extends the base correction configuration with orbit-specific parameters
    for constraint selection, integration settings, and event detection.

    Parameters
    ----------
    residual_indices : tuple of int, default=()
        State components used to build the residual vector.
    control_indices : tuple of int, default=()
        State components allowed to change during correction.
    extra_jacobian : callable or None, default=None
        Additional Jacobian contribution function.
    target : tuple of float, default=(0.0,)
        Target values for the residual components.
    event_func : callable, default=:class:`~hiten.algorithms.poincare.singlehit.backend._y_plane_crossing`
        Function to detect Poincare section crossings.
    """

    residual_indices: tuple[int, ...] = ()  # Components used to build R(x)
    control_indices: tuple[int, ...] = ()   # Components allowed to change
    extra_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    target: tuple[float, ...] = (0.0,)  # Desired residual values
    event_func: Callable[..., tuple[float, np.ndarray]] = _y_plane_crossing


@dataclass(frozen=True, slots=True)
class _MultipleShootingOrbitCorrectionConfig(_OrbitCorrectionConfig):
    """Define configuration for multiple shooting correction.

    Extends :class:`_OrbitCorrectionConfig` with multiple shooting-specific
    parameters for patch management, continuity enforcement, and boundary
    condition handling.

    Multiple shooting divides a trajectory into N segments (patches) and
    treats each patch's initial state as an independent variable. This
    approach is more robust than single shooting for long-period orbits,
    unstable systems, or poor initial guesses.

    Parameters
    ----------
    n_patches : int, default=3
        Number of shooting segments. More patches generally improve robustness
        at the cost of increased computational cost. Recommended ranges:
        
        - Lyapunov orbits: 3-5 patches
        - Halo orbits: 4-7 patches
        - Long-period orbits: 7-15 patches
        - Unstable orbits: 5-10 patches
        
    patch_strategy : {"uniform", "adaptive", "manual"}, default="uniform"
        Strategy for distributing patch points along the trajectory:
        
        - "uniform": Evenly spaced in time (simplest, most common)
        - "adaptive": Concentrate patches near instabilities (future feature)
        - "manual": User-provided times via `manual_patch_times`
        
    manual_patch_times : tuple[float, ...] or None, default=None
        Explicit patch point times for `patch_strategy="manual"`.
        Must have length `n_patches + 1` (includes initial and final times).
        Example: `(0.0, 0.5, 1.2, 2.0)` for 3 patches.
        
    continuity_indices : tuple[int, ...], default=()
        State components to enforce continuity at patch junctions.
        If empty, uses `control_indices` (enforces continuity on all
        free variables). Common choices:
        
        - `()`: Use control_indices (default)
        - `(0, 1, 2, 3, 4, 5)`: Enforce full state continuity
        - `(0, 2, 4)`: Only position continuity (relax velocity)
        
    enforce_all_continuity : bool, default=True
        If True, enforces continuity on all `control_indices`.
        If False, only enforces on `continuity_indices`.
        
    boundary_only_indices : tuple[int, ...], default=()
        State components with boundary conditions but no continuity
        enforcement at internal patches. Useful for problems with
        endpoint constraints that don't apply to patch junctions.

    Notes
    -----
    The parameter vector for multiple shooting has dimension:
    
    .. math::
        
        n_{\\text{params}} = n_{\\text{patches}} \\times n_{\\text{control}}

    The residual vector has dimension:
    
    .. math::
        
        n_{\\text{residual}} = (n_{\\text{patches}} - 1) \\times n_{\\text{continuity}} + n_{\\text{boundary}}

    Uniform spacing works well for most periodic orbits in the CR3BP.
    Adaptive spacing (future feature) concentrates patches near regions
    of high sensitivity (large STM condition numbers).

    Examples
    --------
    Basic configuration for Lyapunov orbit:

    >>> config = _MultipleShootingOrbitCorrectionConfig(
    ...     n_patches=3,
    ...     control_indices=(0, 2, 4),  # x, z, vx
    ...     residual_indices=(2, 3, 4),  # z, vz, vx
    ...     target=(0.0, 0.0, 0.0),
    ...     tol=1e-10
    ... )

    Halo orbit with more patches for robustness:

    >>> config = _MultipleShootingOrbitCorrectionConfig(
    ...     n_patches=7,
    ...     control_indices=(0, 1, 2, 3, 4, 5),
    ...     residual_indices=(1, 2, 3, 4, 5),
    ...     target=(0.0, 0.0, 0.0, 0.0, 0.0),
    ...     tol=1e-12,
    ...     max_attempts=50
    ... )

    Manual patch placement:

    >>> config = _MultipleShootingOrbitCorrectionConfig(
    ...     n_patches=4,
    ...     patch_strategy="manual",
    ...     manual_patch_times=(0.0, 0.3, 0.8, 1.5, 2.0),
    ...     control_indices=(0, 2, 4)
    ... )

    Partial continuity enforcement:

    >>> config = _MultipleShootingOrbitCorrectionConfig(
    ...     n_patches=5,
    ...     control_indices=(0, 1, 2, 3, 4, 5),
    ...     continuity_indices=(0, 1, 2),  # Only position continuity
    ...     boundary_indices=(2, 3, 4),     # Full velocity at boundary
    ...     enforce_all_continuity=False
    ... )

    See Also
    --------
    :class:`_OrbitCorrectionConfig`
        Base configuration for orbit correction (single shooting).
    :class:`_BaseCorrectionConfig`
        Base configuration for all correction algorithms.
    :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
        Problem definition using this configuration.
    """
    n_patches: int = 3
    patch_strategy: Literal["uniform", "adaptive", "manual"] = "uniform"
    manual_patch_times: tuple[float, ...] | None = None
    continuity_indices: tuple[int, ...] = ()
    enforce_all_continuity: bool = True
    boundary_only_indices: tuple[int, ...] = ()
