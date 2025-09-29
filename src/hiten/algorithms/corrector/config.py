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

