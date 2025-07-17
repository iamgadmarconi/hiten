from abc import ABC
from typing import Any, Tuple

import numpy as np

from hiten.algorithms.corrector._step_interface import _ArmijoStepInterface
from hiten.algorithms.corrector.base import (JacobianFn, NormFn, ResidualFn,
                                             _Corrector)
from hiten.algorithms.corrector.line import _LineSearchConfig
from hiten.utils.log_config import logger


class _NewtonCore(_ArmijoStepInterface, _Corrector, ABC):

    def __init__(self, *, line_search_config: _LineSearchConfig | bool | None = None, **kwargs) -> None:
        """Core Newton solver.

        Parameters
        ----------
        line_search_config : _LineSearchConfig, bool, or None, optional
            Armijo back-tracking line search configuration. If passed, the
            line-search will be used for the correction. If passed as a boolean,
            the line-search will be used with default parameters, otherwise not.
        **kwargs
            Remaining keyword arguments are forwarded to the mix-ins upper in
            the MRO (currently unused but preserved for future compatibility).
        """

        # Delegate line-search handling to mix-in
        super().__init__(line_search_config=line_search_config, **kwargs)

    def _on_iteration(self, k: int, x: np.ndarray, r_norm: float) -> None:
        """Hook executed after each Newton iteration.

        Subclasses can override this method to perform custom bookkeeping or
        adaptive strategies (e.g. trust-region radius updates, dynamic
        tolerances, detailed logging) without touching the core solver. The
        default implementation is a *no-op*.

        Parameters
        ----------
        k : int
            Current iteration index (starting at 0).
        x : ndarray
            Current estimate of the solution.
        r_norm : float
            Norm of the residual vector at *x*.
        """

        pass

    def _on_accept(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Hook executed once after the solver *converged*.

        This complements :py:meth:`_on_iteration` and allows subclasses to
        perform post-processing that should only happen *once* (e.g. cache
        factorisations, update trust-region radii, record convergence stats)
        without adding conditional logic to the per-iteration hook.

        Parameters
        ----------
        x : ndarray
            Converged solution vector.
        iterations : int
            Total number of iterations performed (zero-based count).
        residual_norm : float
            Norm of the residual at convergence (≤ *tol*).
        """
        pass

    def _on_failure(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        """Hook executed once after the solver *failed*.

        This complements :py:meth:`_on_iteration` and allows subclasses to
        perform post-processing that should only happen *once* (e.g. cache
        factorisations, update trust-region radii, record convergence stats)
        without adding conditional logic to the per-iteration hook.

        Parameters
        ----------
        x : ndarray
            Failed solution vector.
        iterations : int
            Total number of iterations performed (zero-based count).
        residual_norm : float
            Norm of the residual at failure (≥ *tol*).
        """
        pass

    def _compute_residual(self, x: np.ndarray, residual_fn: ResidualFn) -> np.ndarray:
        """Compute the residual vector R(x).

        Isolated in a separate method so that subclasses can override or
        accelerate this step (e.g. with numba) without touching the overall
        solver logic.
        """
        return residual_fn(x)

    def _compute_norm(self, residual: np.ndarray, norm_fn: NormFn) -> float:
        """Compute the norm of the residual vector.

        Extracted to its own method for easier customisation of the convergence
        metric.
        """
        return norm_fn(residual)

    def _compute_jacobian(
        self,
        x: np.ndarray,
        residual_fn: ResidualFn,
        jacobian_fn: JacobianFn | None,
        fd_step: float,
    ) -> np.ndarray:
        """Return the Jacobian matrix J(x).

        By default this method uses the supplied *jacobian_fn* when available
        and falls back to a second-order central finite-difference
        approximation otherwise.  Subclasses may override this method to
        provide analytical Jacobians or accelerated implementations.
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
        """Solve the linear Newton system.

        The default implementation applies light Tikhonov regularisation when
        the Jacobian is ill-conditioned or singular and automatically switches
        to least-squares solutions for rectangular systems.  Overriding this
        method enables alternative linear solvers (e.g. GPU-accelerated or
        iterative Krylov methods).
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

    # _apply_step removed; step-size control delegated to _Stepper strategy

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
        if norm_fn is None:
            norm_fn = lambda r: float(np.linalg.norm(r))

        x = x0.copy()
        info: dict[str, Any] = {}

        # Obtain the stepper callable from the strategy mix-in
        stepper = self._build_line_searcher(residual_fn, norm_fn, max_delta)

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

            x_new, r_norm_new, alpha_used = stepper(x, delta, r_norm)

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

        raise RuntimeError(
            f"Newton did not converge after {max_attempts} iterations (|R|={r_final_norm:.2e})."
        ) from None
