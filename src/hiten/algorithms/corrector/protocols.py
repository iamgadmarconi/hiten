from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn


@runtime_checkable
class StepStrategyProtocol(Protocol):
    """Protocol for a step-size control strategy used by backends.

    Transforms a Newton step into an accepted update and returns the
    new point, new residual norm, and effective step scale.
    """

    def __call__(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        current_norm: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Transform a Newton step into an accepted update.
        
        Parameters
        ----------
        x : np.ndarray
            Current iterate in the Newton method.
        delta : np.ndarray
            Newton step direction (typically from solving J*delta = -F).
        current_norm : float
            Norm of the residual at the current iterate *x*.
            
        Returns
        -------
        x_new : np.ndarray
            Updated iterate after applying the step transformation.
        r_norm_new : float
            Norm of the residual at the new iterate *x_new*.
        alpha_used : float
            Step-size scaling factor actually employed.
        """
        ...


@runtime_checkable
class CorrectorBackendProtocol(Protocol):
    """Protocol for backend correctors (e.g., Newton).
    
    Attributes
    ----------
    correct : Callable
        Correct method for the backend.
    """

    def correct(
        self,
        x0: np.ndarray,
        residual_fn: ResidualFn,
        *,
        jacobian_fn: JacobianFn | None = None,
        norm_fn: NormFn | None = None,
        tol: float = 1e-10,
        max_attempts: int = 25,
        max_delta: float | None = 1e-2,
        fd_step: float = 1e-8,
    ) -> tuple[np.ndarray, dict]:
        """Correct the initial guess for the residual function.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the parameter vector.
        residual_fn : ResidualFn
            Residual function R(x).
        jacobian_fn : JacobianFn | None
            Optional analytical Jacobian.
        norm_fn : NormFn | None
            Optional norm function for convergence checks.
        tol : float
            Convergence tolerance on residual norm.
        max_attempts : int
            Maximum Newton iterations.
        max_delta : float | None
            Optional cap on infinity-norm of Newton step.
        fd_step : float
            Finite-difference step if Jacobian is not provided.
            
        Returns
        -------
        x_corrected : np.ndarray
            Corrected parameter vector.
        info : dict
            Convergence information with keys 'iterations' and 'residual_norm'.
        """
        ...


class StepProtocol(Protocol):
    """Define the protocol for step transformation functions in Newton-type methods.

    This protocol defines the interface for functions that transform a
    computed Newton step into an accepted update. Different implementations
    can provide various step-size control strategies, from simple full
    steps to sophisticated line search and trust region methods.

    The protocol enables separation of concerns between:
    - Newton step computation (direction finding)
    - Step size control (distance along direction)
    - Convergence monitoring (residual evaluation)

    Implementations typically handle:
    - Step size scaling for convergence control
    - Safeguards against excessive step sizes
    - Line search for sufficient decrease conditions
    - Trust region constraints for robustness

    Parameters
    ----------
    x : ndarray
        Current iterate in the Newton method.
    delta : ndarray
        Newton step direction (typically from solving J*delta = -F).
    current_norm : float
        Norm of the residual at the current iterate *x*.

    Returns
    -------
    x_new : ndarray
        Updated iterate after applying the step transformation.
    r_norm_new : float
        Norm of the residual at the new iterate *x_new*.
    alpha_used : float
        Step-size scaling factor actually employed. A value of 1.0
        indicates the full Newton step was taken, while smaller values
        indicate step size reduction for convergence control.

    Notes
    -----
    The protocol allows for flexible step-size control strategies:
    
    - **Full Newton steps**: alpha_used = 1.0, x_new = x + delta
    - **Scaled steps**: alpha_used < 1.0, x_new = x + alpha * delta
    - **Line search**: alpha chosen to satisfy decrease conditions
    - **Trust region**: delta modified to stay within trust region
    
    Implementations should ensure that r_norm_new is computed consistently
    with the norm function used in the overall Newton algorithm.

    Examples
    --------
    >>> # Simple full-step implementation
    >>> def full_step(x, delta, current_norm):
    ...     x_new = x + delta
    ...     r_norm_new = norm_fn(residual_fn(x_new))
    ...     return x_new, r_norm_new, 1.0
    >>>
    >>> # Scaled step implementation
    >>> def scaled_step(x, delta, current_norm):
    ...     alpha = 0.5  # Half step
    ...     x_new = x + alpha * delta
    ...     r_norm_new = norm_fn(residual_fn(x_new))
    ...     return x_new, r_norm_new, alpha

    See Also
    --------
    :class:`~hiten.algorithms.corrector._step_interface._PlainStepInterface`
        Simple implementation with optional step size capping.
    :class:`~hiten.algorithms.corrector._step_interface._ArmijoStepInterface`
        Line search implementation using Armijo conditions.
    """

    def __call__(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        current_norm: float,
    ) -> tuple[np.ndarray, float, float]:
        """Transform Newton step into accepted update.

        Parameters
        ----------
        x : ndarray
            Current iterate.
        delta : ndarray
            Newton step direction.
        current_norm : float
            Norm of residual at current iterate.

        Returns
        -------
        x_new : ndarray
            Updated iterate.
        r_norm_new : float
            Norm of residual at new iterate.
        alpha_used : float
            Step scaling factor employed.
        """
        ...