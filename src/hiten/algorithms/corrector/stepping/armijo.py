"""Define the Armijo step interface for step-size control strategies.

This module provides the Armijo step interface for step-size control strategies.
"""

from typing import Optional, Tuple

import numpy as np

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.stepping.base import _CorrectorStepBase
from hiten.algorithms.corrector.stepping.norm import _default_norm
from hiten.algorithms.corrector.types import NormFn, ResidualFn
from hiten.algorithms.types.exceptions import BackendError
from hiten.utils.log_config import logger


class _ArmijoLineSearch:
    """Implement Armijo line search with backtracking for Newton methods.
    
    Implements the Armijo rule for sufficient decrease, ensuring that
    each step reduces the residual norm by a sufficient amount proportional
    to the step size. Includes step size capping and fallback strategies.
    
    Parameters
    ----------
    residual_fn : ResidualFn
        Function to compute residual vectors.
    norm_fn : NormFn, optional
        Function to compute residual norms. Uses L2 norm if None.
    max_delta : float, optional
        Maximum allowed step size (infinity norm).
    alpha_reduction : float, default=0.5
        Factor to reduce step size in backtracking.
    min_alpha : float, default=1e-4
        Minimum step size before giving up.
    armijo_c : float, default=0.1
        Armijo parameter for sufficient decrease condition.
    """

    def __init__(
        self,
        *,
        residual_fn: ResidualFn,
        norm_fn: Optional[NormFn] = None,
        max_delta: Optional[float] = 1e-2,
        alpha_reduction: float = 0.5,
        min_alpha: float = 1e-4,
        armijo_c: float = 0.1,
    ) -> None:
        self.norm_fn = _default_norm if norm_fn is None else norm_fn
        self.residual_fn = residual_fn
        self.max_delta = max_delta
        self.alpha_reduction = alpha_reduction
        self.min_alpha = min_alpha
        self.armijo_c = armijo_c

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
        x0 : np.ndarray
            Current parameter vector.
        delta : np.ndarray
            Newton step direction.
        current_norm : float
            Norm of residual at current point.

        Returns
        -------
        x_new : np.ndarray
            Updated parameter vector.
        r_norm_new : float
            Norm of residual at new point.
        alpha_used : float
            Step size scaling factor that was accepted.
            
        Raises
        ------
        ValueError
            If residual function is not provided.
        :class:`~hiten.algorithms.types.exceptions.BackendError`
            If line search fails to find any productive step.
        """
        if self.residual_fn is None:
            raise ValueError("residual_fn must be provided")

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
            try:
                r_trial = self.residual_fn(x_trial)
                norm_trial = self.norm_fn(r_trial)
            except Exception as exc:
                logger.debug(
                    "Residual evaluation failed at alpha=%.3e: %s. Trying smaller step.",
                    alpha,
                    exc,
                )
                alpha *= self.alpha_reduction
                continue

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
        raise BackendError("Armijo line search failed to find a productive step.")


class _ArmijoStep(_CorrectorStepBase):
    """Provide a step interface with Armijo line search for robust convergence.

    This class extends the plain step interface with Armijo line
    search capabilities. It provides a more robust stepping strategy that
    can handle poorly conditioned problems, bad initial guesses, and
    nonlinear systems where full Newton steps might diverge.

    Parameters
    ----------
    alpha_reduction : float, default=0.5
        Factor to reduce step size in backtracking.
    min_alpha : float, default=1e-4
        Minimum step size before giving up.
    armijo_c : float, default=0.1
        Armijo parameter for sufficient decrease condition.
    **kwargs
        Additional arguments passed to parent classes.

    Notes
    -----
    The Armijo condition requires that the residual norm decrease by a
    sufficient amount proportional to the step size, providing a balance
    between convergence speed and robustness.

    Examples
    --------
    >>> # Default line search parameters
    >>> interface = _ArmijoStep()
    >>>
    >>> # Custom line search parameters
    >>> interface = _ArmijoStep(alpha_reduction=0.7, min_alpha=1e-5)

    See Also
    --------
    :class:`~hiten.algorithms.corrector._step_interface._CorrectorPlainStep`
        Parent class providing plain Newton step capabilities.
    :class:`~hiten.algorithms.corrector.line._ArmijoLineSearch`
        Line search implementation used by this interface.
    """

    def __init__(
        self,
        *,
        alpha_reduction: float = 0.5,
        min_alpha: float = 1e-4,
        armijo_c: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha_reduction = alpha_reduction
        self.min_alpha = min_alpha
        self.armijo_c = armijo_c

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> CorrectorStepProtocol:
        """Build Armijo line search stepper.

        Creates an Armijo line search stepper with the configured parameters.

        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual vectors.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed step size.

        Returns
        -------
        stepper : :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
            Armijo line search step transformation function.
        """
        searcher = _ArmijoLineSearch(
            residual_fn=residual_fn,
            norm_fn=norm_fn,
            max_delta=max_delta,
            alpha_reduction=self.alpha_reduction,
            min_alpha=self.min_alpha,
            armijo_c=self.armijo_c,
        )

        def _armijo_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            """Armijo line search step transformation."""
            return searcher(x0=x, delta=delta, current_norm=current_norm)

        return _armijo_step