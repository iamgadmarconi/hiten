"""Provide a Newton-Raphson correction algorithm with robust linear algebra.

This module provides the core Newton-Raphson implementation with automatic
handling of ill-conditioned systems, finite-difference Jacobians, and
extensible hooks for customization.
"""

from typing import Any

import numpy as np

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.types import (CorrectorInput, CorrectorOutput,
                                              JacobianCallable, NormCallable,
                                              ResidualCallable, StepperFactory)
from hiten.algorithms.types.exceptions import ConvergenceError
from hiten.utils.log_config import logger


class _NewtonBackend(_CorrectorBackend):
    """Implement the Newton-Raphson algorithm with robust linear algebra and step control.
    
    Combines Newton-Raphson iteration with Armijo line search, automatic
    handling of ill-conditioned Jacobians, and extensible hooks for
    customization. Uses multiple inheritance to separate step control
    from core Newton logic.

    Parameters
    ----------
    stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory`
        The stepper factory to use.

    Notes
    -----
    This class is designed to be mixed with :class:`~hiten.algorithms.corrector.stepping.armijo._ArmijoStep`
    to provide a robust Newton-Raphson algorithm with Armijo line search.
    """

    def run(
        self,
        *,
        request: CorrectorInput,
        stepper_factory: StepperFactory | None = None,
    ) -> CorrectorOutput:
        """Solve nonlinear system using Newton-Raphson method.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual vector R(x).
        jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` or None, optional
            Function to compute Jacobian dR/dx. Uses finite-difference if None.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` or None, optional
            Function to compute residual norm. Uses L2 norm if None.
        tol : float, default=1e-10
            Convergence tolerance for residual norm.
        max_attempts : int, default=25
            Maximum number of Newton iterations.
        max_delta : float or None, default=1e-2
            Maximum step size for numerical stability.
        fd_step : float, default=1e-8
            Step size for finite-difference Jacobian.
            
        Returns
        -------
        x_solution : np.ndarray
            Converged solution vector.
        iterations : int
            Number of iterations performed.
        residual_norm : float
            Final residual norm achieved.
            
        Raises
        ------
        RuntimeError
            If Newton method fails to converge within max_attempts.
        """
        tol = request.tol
        max_attempts = request.max_attempts
        fd_step = request.fd_step
        max_delta = request.max_delta
        residual_fn: ResidualCallable = request.residual_fn
        jacobian_fn: JacobianCallable | None = request.jacobian_fn
        norm_fn: NormCallable | None = request.norm_fn

        if norm_fn is None:
            norm_callable: NormCallable = lambda r: float(np.linalg.norm(r))
        else:
            norm_callable = norm_fn

        x = request.initial_guess.copy()

        factory = self._stepper_factory if stepper_factory is None else stepper_factory
        stepper = factory(residual_fn, norm_callable, max_delta)

        metadata: dict[str, Any] = dict(request.metadata)

        for k in range(max_attempts):
            r = self._compute_residual(x, residual_fn)
            r_norm = self._compute_norm(r, norm_callable)

            try:
                self.on_iteration(k, x, r_norm)
            except Exception:
                pass

            if r_norm < tol:
                logger.info("Newton converged after %d iterations (|R|=%.2e)", k, r_norm)
                try:
                    self.on_accept(x, iterations=k, residual_norm=r_norm)
                except Exception:
                    pass
                metadata.update({"iterations": k, "residual_norm": r_norm})
                return CorrectorOutput(x_corrected=x, iterations=k, residual_norm=r_norm, metadata=metadata)

            J = self._compute_jacobian(x, residual_fn, jacobian_fn, fd_step)
            delta = self._solve_delta_dense(J, r)

            try:
                x_new, r_norm_new, alpha_used = stepper(x, delta, r_norm)
            except Exception as exc:
                # Map step strategy failures to convergence errors at backend level
                raise ConvergenceError(
                    f"Step strategy failed to produce an update at iter {k}: {exc}"
                ) from exc

            x = x_new

        r_final = self._compute_residual(x, residual_fn)
        r_final_norm = self._compute_norm(r_final, norm_callable)

        if r_final_norm < tol:
            self.on_accept(x, iterations=max_attempts, residual_norm=r_final_norm)
            metadata.update({"iterations": max_attempts, "residual_norm": r_final_norm})
            return CorrectorOutput(x_corrected=x, iterations=max_attempts, residual_norm=r_final_norm, metadata=metadata)

        self.on_failure(x, iterations=max_attempts, residual_norm=r_final_norm)

        raise ConvergenceError(
            f"Newton did not converge after {max_attempts} iterations (|R|={r_final_norm:.2e})."
        ) from None
