"""Protocol definitions for the corrector module.

This module provides the protocol definitions for the corrector module.
"""

from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn


@runtime_checkable
class CorrectorStepProtocol(Protocol):
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
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Residual function R(x).
        jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` | None
            Optional analytical Jacobian.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` | None
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
