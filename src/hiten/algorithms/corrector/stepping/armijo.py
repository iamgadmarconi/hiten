"""Define the Armijo step interface for step-size control strategies.

This module provides the Armijo step interface for step-size control strategies.
"""

from typing import Optional, Tuple

import numpy as np

from hiten.algorithms.corrector.config import _LineSearchConfig
from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.stepping.base import _CorrectorStepBase
from hiten.algorithms.corrector.types import NormFn, ResidualFn
from hiten.algorithms.utils.exceptions import BackendError
from hiten.utils.log_config import logger


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


class _ArmijoLineSearch:
    """Implement Armijo line search with backtracking for Newton methods.
    
    Implements the Armijo rule for sufficient decrease, ensuring that
    each step reduces the residual norm by a sufficient amount proportional
    to the step size. Includes step size capping and fallback strategies.
    
    Parameters
    ----------
    config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig`
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
        BackendError
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

    This class extends the plain step interface with optional Armijo line
    search capabilities. It provides a more robust stepping strategy that
    can handle poorly conditioned problems, bad initial guesses, and
    nonlinear systems where full Newton steps might diverge.

    The interface supports both plain Newton steps (for efficiency) and
    Armijo line search (for robustness), with the choice determined by
    configuration. This flexibility allows algorithms to adapt their
    stepping strategy based on problem characteristics or user preferences.

    Features:
    - Optional Armijo line search with backtracking
    - Configurable line search parameters
    - Fallback to plain Newton steps when line search is disabled
    - Automatic sufficient decrease condition checking
    - Robust convergence for challenging problems

    The Armijo line search ensures sufficient decrease in the residual norm,
    providing theoretical convergence guarantees under appropriate conditions.
    This makes it suitable for:

    - Poorly conditioned nonlinear systems
    - Problems with bad initial guesses
    - Situations requiring guaranteed convergence properties
    - Production code where robustness is critical

    Attributes
    ----------
    _line_search_config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig` or None
        Configuration object for line search parameters.
    _use_line_search : bool
        Flag indicating whether line search should be used.

    Parameters
    ----------
    line_search_config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig`, bool, or None, optional
        Line search configuration. Can be:
        - None: Disable line search (use plain Newton steps)
        - True: Enable line search with default parameters
        - False: Explicitly disable line search
        - :class:`~hiten.algorithms.corrector.config._LineSearchConfig`: Enable line search with custom parameters
    **kwargs
        Additional arguments passed to parent classes.

    Notes
    -----
    The interface inherits plain Newton step capabilities from its parent
    class, ensuring that it can fall back to simple stepping when line
    search is not needed or fails to improve convergence.

    The Armijo condition requires that the residual norm decrease by a
    sufficient amount proportional to the step size, providing a balance
    between convergence speed and robustness.

    Examples
    --------
    >>> # Enable line search with default parameters
    >>> interface = _ArmijoStep(line_search_config=True)
    >>>
    >>> # Disable line search (use plain Newton)
    >>> interface = _ArmijoStep(line_search_config=False)
    >>>
    >>> # Custom line search configuration
    >>> config = _LineSearchConfig(c1=1e-4, rho=0.5, max_iter=20)
    >>> interface = _ArmijoStep(line_search_config=config)

    See Also
    --------
    :class:`~hiten.algorithms.corrector._step_interface._CorrectorPlainStep`
        Parent class providing plain Newton step capabilities.
    :class:`~hiten.algorithms.corrector.line._ArmijoLineSearch`
        Line search implementation used by this interface.
    :class:`~hiten.algorithms.corrector.config._LineSearchConfig`
        Configuration class for line search parameters.
    """

    _line_search_config: Optional[_LineSearchConfig]
    _use_line_search: bool

    def __init__(
        self,
        *,
        line_search_config: _LineSearchConfig | bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if line_search_config is None:
            self._line_search_config = None
            self._use_line_search = False
        elif isinstance(line_search_config, bool):
            if line_search_config:
                self._line_search_config = _LineSearchConfig()
                self._use_line_search = True
            else:
                self._line_search_config = None
                self._use_line_search = False
        else:
            self._line_search_config = line_search_config
            self._use_line_search = True

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> CorrectorStepProtocol:
        """Build a step transformation function with optional line search.

        This method creates either a plain Newton stepper or an Armijo
        line search stepper based on the configuration. The choice is
        made at stepper creation time and remains fixed for the lifetime
        of the stepper.

        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual vectors.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed step size (used by plain stepper fallback).

        Returns
        -------
        stepper : :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
            Step transformation function, either plain Newton or Armijo
            line search based on configuration.

        Notes
        -----
        When line search is disabled, this method falls back to the
        plain Newton stepper from the parent class, ensuring consistent
        behavior and maintaining the step size capping safeguard.

        When line search is enabled, the method creates an Armijo line
        search object and wraps it in a stepper function that matches
        the expected interface.
        """
        if not getattr(self, "_use_line_search", False):
            return self._make_plain_stepper(residual_fn, norm_fn, max_delta)

        cfg = self._line_search_config
        searcher = _ArmijoLineSearch(
            config=cfg._replace(residual_fn=residual_fn, norm_fn=norm_fn)
        )

        def _armijo_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            """Armijo line search step transformation.
            
            This closure wraps the Armijo line search object to provide
            the standard stepper interface expected by Newton algorithms.
            """
            return searcher(x0=x, delta=delta, current_norm=current_norm)

        return _armijo_step