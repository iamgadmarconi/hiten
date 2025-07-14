from typing import Callable, Tuple

import numpy as np

from hiten.utils.log_config import logger

NormFn = Callable[[np.ndarray], float]
ResidualFn = Callable[[np.ndarray], np.ndarray]


def _default_norm(r: np.ndarray) -> float:
    """Return 2-norm of *r*.

    Uses 2-norm because most invariance residuals already divide by N.
    """

    return float(np.linalg.norm(r))


def armijo_line_search(
    x0: np.ndarray,
    delta: np.ndarray,
    residual_fn: ResidualFn,
    current_norm: float,
    *,
    norm_fn: NormFn | None = None,
    max_delta: float | None = 1e-2,
    alpha_reduction: float = 0.5,
    min_alpha: float = 1e-4,
    armijo_c: float = 0.1,
) -> Tuple[np.ndarray, float, float]:
    r"""
    Apply *step-size cap* and Armijo back-tracking line search.

    Parameters
    ----------
    x0 : ndarray
        Current parameter vector (flattened).
    delta : ndarray
        Proposed Newton step of same shape as *x0*.
    residual_fn : callable
        Function *r = residual_fn(x)* returning the residual vector for *x*.
    current_norm : float
        Norm of the current residual ``||r(x0)||`` used for the Armijo test.
    norm_fn : callable, optional
        Function to compute a norm from the residual vector.  Defaults to
        2-norm.
    max_delta, alpha_reduction, min_alpha, armijo_c : float, optional
        Standard Armijo parameters (see :func:`PeriodicOrbit.correct`).

    Returns
    -------
    x_new : ndarray
        Updated parameter vector after line search.
    new_norm : float
        Norm of the residual at *x_new*.
    alpha_used : float
        Scaling factor actually applied to *delta*.
    """

    if norm_fn is None:
        norm_fn = _default_norm

    if (max_delta is not None) and (not np.isinf(max_delta)):
        delta_norm = np.linalg.norm(delta, ord=np.inf)
        if delta_norm > max_delta:
            delta = delta * (max_delta / delta_norm)
            logger.info("Capping Newton step (|delta|=%.2e > %.2e)", delta_norm, max_delta)

    alpha = 1.0
    best_x = x0
    best_norm = current_norm
    best_alpha = 0.0

    while alpha >= min_alpha:
        x_trial = x0 + alpha * delta
        r_trial = residual_fn(x_trial)
        norm_trial = norm_fn(r_trial)

        # Armijo / sufficient decrease condition
        if norm_trial <= (1.0 - armijo_c * alpha) * current_norm:
            logger.debug(
                "Armijo success: alpha=%.3e, |r|=%.3e (was |r0|=%.3e)",
                alpha,
                norm_trial,
                current_norm,
            )
            return x_trial, norm_trial, alpha

        # Keep track of best point (and corresponding alpha) encountered for fallback
        if norm_trial < best_norm:
            best_x = x_trial
            best_norm = norm_trial
            best_alpha = alpha

        alpha *= alpha_reduction

    # Fallback: return the best point found (may be the original)
    logger.warning(
        "Line search exhausted; using best \u03b1 found (\u03b1=%.3e, |r|=%.3e)",
        best_alpha,
        best_norm,
    )
    return best_x, best_norm, best_alpha