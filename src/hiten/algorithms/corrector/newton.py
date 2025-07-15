from typing import TYPE_CHECKING, Any, Optional, Tuple, Callable

import numpy as np

from hiten.algorithms.corrector.base import (JacobianFn, NormFn, ResidualFn,
                                             _Corrector)
from hiten.algorithms.corrector.line import armijo_line_search
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit



class _NewtonCorrector(_Corrector):
    """Classical Newton-Raphson solver with optional Armijo line search.

    The algorithm solves ``R(x) = 0`` for *x* given *residual_fn* (and
    optionally *jacobian_fn*).  If the Jacobian is not provided, it is
    approximated by forward finite differences.
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
        line_search: bool = False,
        max_delta: float | None,
        alpha_reduction: float | None,
        min_alpha: float | None,
        armijo_c: float | None,
        callback: "Callable[[int, np.ndarray, float], None] | None" = None,
        fd_step: float = 1e-8,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        if norm_fn is None:
            norm_fn = lambda r: float(np.linalg.norm(r))

        x = x0.copy()
        info: dict[str, Any] = {}

        for k in range(max_attempts):
            r = residual_fn(x)
            r_norm = norm_fn(r)

            # Optional user-provided callback for real-time diagnostics
            if callback is not None:
                try:
                    callback(k, x, r_norm)
                except Exception as exc:
                    logger.warning("Newton callback raised an exception: %s", exc)

            if r_norm < tol:
                logger.info("Newton converged after %d iterations (|R|=%.2e)", k, r_norm)
                info.update(iterations=k, residual_norm=r_norm)
                return x, info

            # Compute / approximate Jacobian
            if jacobian_fn is not None:
                J = jacobian_fn(x)
            else:
                # Finite-difference approximation (central diff, O(h^2))
                n = x.size
                J = np.zeros((r.size, n))
                for i in range(n):
                    x_pert_p = x.copy()
                    x_pert_m = x.copy()
                    h_i = fd_step * max(1.0, abs(x[i]))
                    x_pert_p[i] += h_i
                    x_pert_m[i] -= h_i
                    J[:, i] = (residual_fn(x_pert_p) - residual_fn(x_pert_m)) / (2.0 * h_i)

            try:
                cond_J = np.linalg.cond(J)
            except np.linalg.LinAlgError:
                cond_J = np.inf

            _COND_THRESH = 1e8
            lambda_reg = 0.0  # track actual regularisation strength used
            if J.shape[0] == J.shape[1]:
                if np.isnan(cond_J) or cond_J > _COND_THRESH:
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
                lambda_reg = 1e-12 if (np.isnan(cond_J) or cond_J > _COND_THRESH) else 0.0
                JTJ = J.T @ J + lambda_reg * np.eye(J.shape[1])
                JTr = J.T @ r
                logger.debug("Jacobian cond=%.2e, lambda_reg=%.1e", cond_J, lambda_reg)
                try:
                    delta = np.linalg.solve(JTJ, -JTr)
                except np.linalg.LinAlgError:
                    logger.warning("Normal equations singular; falling back to SVD lstsq")
                    delta = np.linalg.lstsq(J, -r, rcond=None)[0]

            # Apply step update: either with Armijo backtracking or directly (toggle)
            if line_search:
                # Armijo + step capping
                x_new, r_norm_new, alpha_used = armijo_line_search(
                    x0=x,
                    delta=delta,
                    residual_fn=residual_fn,
                    current_norm=r_norm,
                    norm_fn=norm_fn,
                    max_delta=max_delta,
                    alpha_reduction=alpha_reduction,
                    min_alpha=min_alpha,
                    armijo_c=armijo_c,
                )
            else:
                # Optional step capping without line search
                if (max_delta is not None) and (not np.isinf(max_delta)):
                    delta_norm = np.linalg.norm(delta, ord=np.inf)
                    if delta_norm > max_delta:
                        delta = delta * (max_delta / delta_norm)
                        logger.info(
                            "Capping Newton step (|delta|=%.2e > %.2e)",
                            delta_norm,
                            max_delta,
                        )

                x_new = x + delta
                r_norm_new = norm_fn(residual_fn(x_new))
                alpha_used = 1.0

            logger.debug(
                "Newton iter %d/%d: |R|=%.2e -> %.2e (alpha=%.2e)",
                k + 1,
                max_attempts,
                r_norm,
                r_norm_new,
                alpha_used,
            )
            x = x_new

        # One final convergence check after exhausting the loop
        r_final = residual_fn(x)
        r_final_norm = norm_fn(r_final)

        # Final callback after exiting the loop (non-converged case)
        if callback is not None:
            try:
                callback(max_attempts, x, r_final_norm)
            except Exception as exc:
                logger.warning("Newton callback raised an exception during final call: %s", exc)

        if r_final_norm < tol:
            logger.info("Newton converged after %d iterations (|R|=%.2e)", max_attempts, r_final_norm)
            info.update(iterations=max_attempts, residual_norm=r_final_norm)
            return x, info

        raise RuntimeError(
            f"Newton did not converge after {max_attempts} iterations (|R|={r_final_norm:.2e})."
        )


class _OrbitCorrector(_Corrector):
    """Periodic-orbit specific helper that delegates to :class:`_NewtonCorrector`."""

    def __init__(self, core: _NewtonCorrector | None = None):
        # Allow dependency injection of the generic Newton solver
        self._core = core or _NewtonCorrector()

    def correct(
        self,
        orbit: "PeriodicOrbit",
        *,
        tol: float,
        max_attempts: int,
        forward: int,
        max_delta: float | None,
        alpha_reduction: float,
        min_alpha: float,
        line_search: bool,
        armijo_c: Optional[float],
        finite_difference: bool,
    ) -> Tuple[np.ndarray, float]:
        """Refine *orbit* in-place using the underlying :class:`_NewtonCorrector`.

        All keyword arguments are forwarded to the generic solver.
        """

        cfg = orbit._correction_config
        residual_indices = list(cfg.residual_indices)
        control_indices = list(cfg.control_indices)

        target_vec = np.array(cfg.target)

        # Fixed components (non-control) are kept constant throughout iterations
        base_state = orbit.initial_state.copy()

        # Initial parameter vector consists only of the control components
        p0 = base_state[control_indices]

        # Closure capturing latest half-period so we can update orbit.period
        last_t_event: float | None = None

        def _to_full_state(p_vec: np.ndarray) -> np.ndarray:
            """Embed control-vector *p* into the full 6-dimensional state."""
            x = base_state.copy()
            x[control_indices] = p_vec
            return x

        def _residual_fn(p_vec: np.ndarray) -> np.ndarray:
            nonlocal last_t_event
            x_full = _to_full_state(p_vec)
            last_t_event, X_ev_local = cfg.event_func(
                dynsys=orbit.system._dynsys,
                x0=x_full,
                forward=forward,
            )
            return X_ev_local[residual_indices] - target_vec

        jacobian_fn = None

        if not finite_difference:
            # Build analytic Jacobian function (either from supplied monodromy or by integrating STM)
            def _jacobian_fn(p_vec: np.ndarray) -> np.ndarray:
                x_full = _to_full_state(p_vec)
                # Evaluate event to get half-period and event state (needed for extra_jacobian or on-the-fly STM)
                t_ev_local, X_ev_local = cfg.event_func(
                    dynsys=orbit.system._dynsys,
                    x0=x_full,
                    forward=forward,
                )

                # Compute STM over the half-period for current iterate
                _, _, Phi_flat, _ = _compute_stm(
                    orbit.libration_point._var_eq_system,
                    x_full,
                    t_ev_local,
                    steps=cfg.steps,
                    method=cfg.method,
                    order=cfg.order,
                )
                Phi = Phi_flat

                J_red = Phi[np.ix_(residual_indices, control_indices)]
                if cfg.extra_jacobian is not None:
                    J_red -= cfg.extra_jacobian(X_ev_local, Phi)
                return J_red

            jacobian_fn = _jacobian_fn

        # Infinity-norm as before
        _norm_inf: NormFn = lambda r: float(np.linalg.norm(r, ord=np.inf))

        # Call the generic Newton solver on the reduced parameter space
        p_corr, info = self._core.correct(
            x0=p0,
            residual_fn=_residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=_norm_inf,
            tol=tol,
            max_attempts=max_attempts,
            line_search=line_search,
            alpha_reduction=alpha_reduction,
            max_delta=max_delta,
            min_alpha=min_alpha,
            armijo_c=armijo_c,
        )

        # Ensure we captured the event time
        if last_t_event is None:
            last_t_event, _ = cfg.event_func(
                dynsys=orbit.system._dynsys,
                x0=_to_full_state(p_corr),
                forward=forward,
            )

        # Build the corrected full state vector and update the orbit
        x_corr = _to_full_state(p_corr)

        orbit._reset()
        orbit._initial_state = x_corr
        orbit._period = 2.0 * last_t_event

        logger.info(
            "_OrbitCorrector converged in %d iterations (|R|=%.2e)",
            info.get("iterations", -1),
            info.get("residual_norm", float('nan')),
        )

        return x_corr, last_t_event