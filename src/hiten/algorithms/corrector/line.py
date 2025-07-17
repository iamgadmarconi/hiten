from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from hiten.utils.log_config import logger

NormFn = Callable[[np.ndarray], float]
ResidualFn = Callable[[np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]


def _default_norm(r: np.ndarray) -> float:
    """Return 2-norm of *r*.

    Uses 2-norm because most invariance residuals already divide by N.
    """
    return float(np.linalg.norm(r))

def _infinity_norm(r: np.ndarray) -> float:
    """Return infinity-norm of *r*."""
    return float(np.linalg.norm(r, ord=np.inf))

class _LineSearchConfig(NamedTuple):
    norm_fn: Optional[NormFn] = None
    residual_fn: Optional[ResidualFn] = None
    jacobian_fn: Optional[JacobianFn] = None
    max_delta: float = 1e-2
    alpha_reduction: float = 0.5
    min_alpha: float = 1e-4
    armijo_c: float = 0.1


class _ArmijoLineSearch:

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
        """
        Execute the line-search for the provided Newton step *delta*.

        Parameters
        ----------
        x0 : ndarray
            Current parameter vector.
        delta : ndarray
            Newton step direction.
        current_norm : float
            Norm of residual at x0.

        Returns
        -------
        x_new : ndarray
            Updated parameter vector.
        r_norm_new : float
            Norm of the new residual.
        alpha_used : float
            Step-size scaling that was accepted.
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

        while alpha >= self.min_alpha:
            x_trial = x0 + alpha * delta
            r_trial = self.residual_fn(x_trial)
            norm_trial = self.norm_fn(r_trial)

            # Armijo / sufficient-decrease condition
            if norm_trial <= (1.0 - self.armijo_c * alpha) * current_norm:
                logger.debug(
                    "Armijo success: alpha=%.3e, |r|=%.3e (was |r0|=%.3e)",
                    alpha,
                    norm_trial,
                    current_norm,
                )
                return x_trial, norm_trial, alpha

            # Keep best point seen for possible fallback
            if norm_trial < best_norm:
                best_x = x_trial
                best_norm = norm_trial
                best_alpha = alpha

            alpha *= self.alpha_reduction

        if best_alpha > 0:
            logger.warning(
                "Line search exhausted; using best found step (alpha=%.3e, |r|=%.3e)",
                best_alpha,
                best_norm,
            )
            return best_x, best_norm, best_alpha

        logger.warning(
            "Armijo line search failed to find any step that reduces the residual "
            "for min_alpha=%.2e",
            self.min_alpha,
        )
        raise RuntimeError("Armijo line search failed to find a productive step.")
