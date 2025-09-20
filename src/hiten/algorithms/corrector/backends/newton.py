"""Provide a Newton-Raphson correction algorithm with robust linear algebra.

This module provides the core Newton-Raphson implementation with automatic
handling of ill-conditioned systems, finite-difference Jacobians, and
extensible hooks for customization.
"""

from typing import Any, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.protocols import StepProtocol
from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn
from hiten.algorithms.utils.exceptions import ConvergenceError
from hiten.utils.log_config import logger


class _NewtonBackend(_CorrectorBackend):
    """Implement the Newton-Raphson algorithm with robust linear algebra and step control.
    
    Combines Newton-Raphson iteration with Armijo line search, automatic
    handling of ill-conditioned Jacobians, and extensible hooks for
    customization. Uses multiple inheritance to separate step control
    from core Newton logic.

    Dependency injection
    --------------------
    The backend receives a stepper factory that builds a step strategy per
    problem. If not provided, a safe default plain-capped step is used.

    Notes
    -----
    This class is designed to be mixed with :class:`~hiten.algorithms.corrector.stepping.armijo._ArmijoStep`
    to provide a robust Newton-Raphson algorithm with Armijo line search.
    """

    def __init__(
        self,
        *,
        stepper_factory: Callable[[ResidualFn, NormFn, float | None], StepProtocol] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Dependency-injected factory building the stepper per problem
        self._stepper_factory: Callable[[ResidualFn, NormFn, float | None], StepProtocol]
        if stepper_factory is None:
            # Default to a simple capped plain stepper, keeping backend decoupled
            def _default_factory(res_fn: ResidualFn, nrm_fn: NormFn, max_del: float | None) -> StepProtocol:
                def _plain_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
                    if (max_del is not None) and (not np.isinf(max_del)):
                        delta_norm = float(np.linalg.norm(delta, ord=np.inf))
                        if delta_norm > max_del:
                            scale = max_del / delta_norm
                            delta = delta * scale
                            x_new_local = x + delta
                            r_norm_new_local = nrm_fn(res_fn(x_new_local))
                            return x_new_local, r_norm_new_local, float(scale)
                    x_new_local = x + delta
                    r_norm_new_local = nrm_fn(res_fn(x_new_local))
                    return x_new_local, r_norm_new_local, 1.0
                return _plain_step
            self._stepper_factory = _default_factory
        else:
            self._stepper_factory = stepper_factory

    def _on_iteration(self, k: int, x: np.ndarray, r_norm: float) -> None:
        """Hook called after each iteration for custom processing.

        Override for custom bookkeeping, adaptive strategies, or detailed
        logging without modifying the core solver.

        Parameters
        ----------
        k : int
            Current iteration index (starting at 0).
        x : ndarray
            Current solution estimate.
        r_norm : float
            Residual norm at current estimate.
        """
        pass

    def _on_accept(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Hook called once after successful convergence.

        Override for post-processing that should happen only once after
        convergence (caching, statistics, cleanup).

        Parameters
        ----------
        x : ndarray
            Converged solution vector.
        iterations : int
            Total iterations performed.
        residual_norm : float
            Final residual norm (<= tolerance).
        """
        pass

    def _on_failure(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Hook called once after convergence failure.

        Override for post-processing that should happen only once after
        failure (cleanup, diagnostics, fallback strategies).

        Parameters
        ----------
        x : ndarray
            Final solution vector.
        iterations : int
            Total iterations performed.
        residual_norm : float
            Final residual norm (>= tolerance).
        """
        pass

    def _compute_residual(self, x: np.ndarray, residual_fn: ResidualFn) -> np.ndarray:
        """Compute residual vector R(x).

        Separated for easy overriding or acceleration (e.g., with numba).

        Parameters
        ----------
        x : ndarray
            Current parameter vector.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual.
            
        Returns
        -------
        ndarray
            Residual vector R(x).
        """
        return residual_fn(x)

    def _compute_norm(self, residual: np.ndarray, norm_fn: NormFn) -> float:
        """Compute residual norm for convergence checking.

        Parameters
        ----------
        residual : ndarray
            Residual vector.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute norm.
            
        Returns
        -------
        float
            Scalar norm value.
        """
        return norm_fn(residual)

    def _compute_jacobian(
        self,
        x: np.ndarray,
        residual_fn: ResidualFn,
        jacobian_fn: JacobianFn | None,
        fd_step: float,
    ) -> np.ndarray:
        """Compute Jacobian matrix J(x) = dR/dx.

        Uses analytical Jacobian if provided, otherwise computes central
        finite-difference approximation with O(h^2) accuracy.

        Parameters
        ----------
        x : ndarray
            Current parameter vector.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual.
        jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` or None
            Analytical Jacobian function, if available.
        fd_step : float
            Step size for finite-difference approximation.
            
        Returns
        -------
        ndarray
            Jacobian matrix with shape (m, n) where m is residual size
            and n is parameter size.
        """
        if jacobian_fn is not None:
            return jacobian_fn(x)

        # Finite-difference approximation (central diff, O(h**2))
        n = x.size
        r0 = residual_fn(x)
        J = np.zeros((r0.size, n))
        for i in range(n):
            x_p = x.copy(); x_m = x.copy()
            h_i = fd_step * max(1.0, abs(x[i]))
            x_p[i] += h_i
            x_m[i] -= h_i
            J[:, i] = (residual_fn(x_p) - residual_fn(x_m)) / (2.0 * h_i)
        return J

    def _solve_delta(self, J: np.ndarray, r: np.ndarray, cond_threshold: float = 1e8) -> np.ndarray:
        """Solve linear Newton system J * delta = -r.

        Handles ill-conditioned and rectangular systems automatically:
        - Applies Tikhonov regularization for ill-conditioned square systems
        - Uses least-squares for rectangular systems
        - Falls back to SVD for singular systems

        Parameters
        ----------
        J : ndarray
            Jacobian matrix.
        r : ndarray
            Residual vector.
        cond_threshold : float, default=1e8
            Condition number threshold for regularization.
            
        Returns
        -------
        ndarray
            Newton step vector delta.
        """
        try:
            cond_J = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            cond_J = np.inf

        lambda_reg = 0.0
        if J.shape[0] == J.shape[1]:
            if np.isnan(cond_J) or cond_J > cond_threshold:
                lambda_reg = 1e-12
                J_reg = J + np.eye(J.shape[0]) * lambda_reg
            else:
                J_reg = J

            logger.debug("Jacobian cond=%.2e, lambda_reg=%.1e", cond_J, lambda_reg)
            try:
                delta = np.linalg.solve(J_reg, -r)
            except np.linalg.LinAlgError:
                logger.warning("Jacobian singular; switching to SVD least-squares update")
                delta = np.linalg.lstsq(J_reg, -r, rcond=None)[0]
        else:
            logger.debug("Rectangular Jacobian (%dx%d); solving via Tikhonov least-squares", *J.shape)
            lambda_reg = 1e-12 if (np.isnan(cond_J) or cond_J > cond_threshold) else 0.0
            JTJ = J.T @ J + lambda_reg * np.eye(J.shape[1])
            JTr = J.T @ r
            logger.debug("Jacobian cond=%.2e, lambda_reg=%.1e", cond_J, lambda_reg)
            try:
                delta = np.linalg.solve(JTJ, -JTr)
            except np.linalg.LinAlgError:
                logger.warning("Normal equations singular; falling back to SVD lstsq")
                delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        return delta

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
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Solve nonlinear system using Newton-Raphson method.

        Parameters
        ----------
        x0 : ndarray
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
        x_solution : ndarray
            Converged solution vector.
        info : dict
            Convergence information with keys 'iterations' and 'residual_norm'.
            
        Raises
        ------
        RuntimeError
            If Newton method fails to converge within max_attempts.
        """
        if norm_fn is None:
            norm_fn = lambda r: float(np.linalg.norm(r))

        x = x0.copy()
        info: dict[str, Any] = {}

        # Obtain the stepper callable from the injected factory
        stepper = self._stepper_factory(residual_fn, norm_fn, max_delta)

        for k in range(max_attempts):
            r = self._compute_residual(x, residual_fn)
            r_norm = self._compute_norm(r, norm_fn)

            try:
                self._on_iteration(k, x, r_norm)
            except Exception as exc:
                logger.warning("_on_iteration hook raised an exception: %s", exc)

            if r_norm < tol:
                logger.info("Newton converged after %d iterations (|R|=%.2e)", k, r_norm)
                info.update(iterations=k, residual_norm=r_norm)
                # Notify acceptance hook
                try:
                    self._on_accept(x, iterations=k, residual_norm=r_norm)
                except Exception as exc:
                    logger.warning("_on_accept hook raised an exception: %s", exc)
                return x, info

            J = self._compute_jacobian(x, residual_fn, jacobian_fn, fd_step)
            delta = self._solve_delta(J, r)

            try:
                x_new, r_norm_new, alpha_used = stepper(x, delta, r_norm)
            except Exception as exc:
                # Map step strategy failures to convergence errors at backend level
                raise ConvergenceError(
                    f"Step strategy failed to produce an update at iter {k}: {exc}"
                ) from exc

            logger.debug(
                "Newton iter %d/%d: |R|=%.2e -> %.2e (alpha=%.2e)",
                k + 1,
                max_attempts,
                r_norm,
                r_norm_new,
                alpha_used,
            )
            x = x_new

        r_final = self._compute_residual(x, residual_fn)
        r_final_norm = self._compute_norm(r_final, norm_fn)

        # Call acceptance hook if converged in the final check
        if r_final_norm < tol:
            try:
                self._on_accept(x, iterations=max_attempts, residual_norm=r_final_norm)
            except Exception as exc:
                logger.warning("_on_accept hook raised an exception during final check: %s", exc)

        try:
            self._on_failure(x, iterations=max_attempts, residual_norm=r_final_norm)
        except Exception as exc:
            logger.warning("_on_failure hook raised an exception during final check: %s", exc)

        raise ConvergenceError(
            f"Newton did not converge after {max_attempts} iterations (|R|={r_final_norm:.2e})."
        ) from None
