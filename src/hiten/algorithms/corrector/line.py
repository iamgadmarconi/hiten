"""Provide line search implementations for robust Newton-type methods.

This module provides Armijo line search with backtracking for Newton-type
correction algorithms. Line search ensures sufficient decrease in the residual
norm, providing robustness for challenging nonlinear systems.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from hiten.utils.log_config import logger

# Type aliases for function signatures
NormFn = Callable[[np.ndarray], float]
ResidualFn = Callable[[np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]


def _default_norm(r: np.ndarray) -> float:
    """Compute L2 norm of residual vector.

    Parameters
    ----------
    r : ndarray
        Residual vector.
        
    Returns
    -------
    float
        L2 norm of the residual.
        
    Notes
    -----
    Uses L2 norm as default because most invariance residuals
    are already normalized by the number of components.
    """
    return float(np.linalg.norm(r))

def _infinity_norm(r: np.ndarray) -> float:
    """Compute infinity norm of residual vector.

    Parameters
    ----------
    r : ndarray
        Residual vector.
        
    Returns
    -------
    float
        Maximum absolute component of the residual.
    """
    return float(np.linalg.norm(r, ord=np.inf))

class _LineSearchConfig(NamedTuple):
    """Define configuration parameters for Armijo line search.
    
    Parameters
    ----------
    norm_fn : NormFn or None, default=None
        Function to compute residual norm. Uses L2 norm if None.
    residual_fn : ResidualFn or None, default=None
        Function to compute residual vector. Must be provided.
    jacobian_fn : JacobianFn or None, default=None
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
    norm_fn: Optional[NormFn] = None
    residual_fn: Optional[ResidualFn] = None
    jacobian_fn: Optional[JacobianFn] = None
    max_delta: float = 1e-2
    alpha_reduction: float = 0.5
    min_alpha: float = 1e-4
    armijo_c: float = 0.1


class _ArmijoLineSearch:
    """Implement Armijo line search with backtracking for Newton methods.
    
    Implements the Armijo rule for sufficient decrease, ensuring that
    each step reduces the residual norm by a sufficient amount proportional
    to the step size. Includes step size capping and fallback strategies.
    
    Parameters
    ----------
    config : _LineSearchConfig
        Configuration parameters for the line search.
    """

    def __init__(self, *, config: _LineSearchConfig) -> None:
        self.norm_fn = _default_norm if config.norm_fn is None else config.norm_fn
        self.residual_fn = config.residual_fn
        self.jacobian_fn = config.jacobian_fn
        self.max_delta = config.max_delta
        self.alpha_reduction = config.alpha_reduction
        self.min_alpha = config.min_alpha
        self.armijo_c = config.armijo_c

    def __call__(
        self,
        *,
        x0: np.ndarray,
        delta: np.ndarray,
        current_norm: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Execute Armijo line search with backtracking.

        Finds step size satisfying Armijo condition:
        ||R(x + alpha * delta)|| <= (1 - c * alpha) * ||R(x)||
        
        Starts with full Newton step and reduces by backtracking until
        sufficient decrease is achieved or minimum step size is reached.

        Parameters
        ----------
        x0 : ndarray
            Current parameter vector.
        delta : ndarray
            Newton step direction.
        current_norm : float
            Norm of residual at current point.

        Returns
        -------
        x_new : ndarray
            Updated parameter vector.
        r_norm_new : float
            Norm of residual at new point.
        alpha_used : float
            Step size scaling factor that was accepted.
            
        Raises
        ------
        ValueError
            If residual function is not provided in configuration.
        RuntimeError
            If line search fails to find any productive step.
        """
        if self.residual_fn is None:
            raise ValueError("residual_fn must be provided in _LineSearchConfig")

        if (self.max_delta is not None) and (not np.isinf(self.max_delta)):
            delta_norm = np.linalg.norm(delta, ord=np.inf)
            if delta_norm > self.max_delta:
                delta = delta * (self.max_delta / delta_norm)
                logger.info(
                    "Capping Newton step (|delta|=%.2e > %.2e)",
                    delta_norm,
                    self.max_delta,
                )

        alpha = 1.0
        best_x = x0
        best_norm = current_norm
        best_alpha = 0.0

        # Backtracking line search loop
        while alpha >= self.min_alpha:
            x_trial = x0 + alpha * delta
            r_trial = self.residual_fn(x_trial)
            norm_trial = self.norm_fn(r_trial)

            # Check Armijo sufficient decrease condition
            if norm_trial <= (1.0 - self.armijo_c * alpha) * current_norm:
                logger.debug(
                    "Armijo success: alpha=%.3e, |r|=%.3e (was |r0|=%.3e)",
                    alpha,
                    norm_trial,
                    current_norm,
                )
                return x_trial, norm_trial, alpha

            # Track best point for fallback
            if norm_trial < best_norm:
                best_x = x_trial
                best_norm = norm_trial
                best_alpha = alpha

            alpha *= self.alpha_reduction

        # Fallback to best point found if Armijo condition never satisfied
        if best_alpha > 0:
            logger.warning(
                "Line search exhausted; using best found step (alpha=%.3e, |r|=%.3e)",
                best_alpha,
                best_norm,
            )
            return best_x, best_norm, best_alpha

        # Complete failure case
        logger.warning(
            "Armijo line search failed to find any step that reduces the residual "
            "for min_alpha=%.2e",
            self.min_alpha,
        )
        raise RuntimeError("Armijo line search failed to find a productive step.")
