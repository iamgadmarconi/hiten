from typing import TYPE_CHECKING, Any, Tuple

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
        max_delta: float | None = 1e-2,
        alpha_reduction: float = 0.5,
        min_alpha: float = 1e-4,
        armijo_c: float = 0.1,
        fd_step: float = 1e-8,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        if norm_fn is None:
            norm_fn = lambda r: float(np.linalg.norm(r))

        x = x0.copy()
        info: dict[str, Any] = {}

        for k in range(max_attempts + 1):
            r = residual_fn(x)
            r_norm = norm_fn(r)
            if r_norm < tol:
                logger.info("Newton converged after %d iterations (|R|=%.2e)", k, r_norm)
                info.update(iterations=k, residual_norm=r_norm)
                return x, info

            # Compute / approximate Jacobian
            if jacobian_fn is not None:
                J = jacobian_fn(x)
            else:
                # Finite-difference approximation (forward diff)
                n = x.size
                J = np.zeros((r.size, n))
                for i in range(n):
                    x_pert = x.copy()
                    h_i = fd_step * max(1.0, abs(x[i]))
                    x_pert[i] += h_i
                    J[:, i] = (residual_fn(x_pert) - r) / h_i

            # Regularise singular / ill-conditioned Jacobian
            try:
                cond_J = np.linalg.cond(J)
                if np.isnan(cond_J) or cond_J > 1e12:
                    logger.debug("Jacobian ill-conditioned (cond=%.2e); adding regularisation", cond_J)
                    J += np.eye(J.shape[0]) * 1e-8
                delta = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                logger.warning("Jacobian singular; switching to least-squares update")
                delta = np.linalg.lstsq(J, -r, rcond=None)[0]

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

            logger.debug(
                "Newton iter %d/%d: |R|=%.2e → %.2e (alpha=%.2e)",
                k + 1,
                max_attempts,
                r_norm,
                r_norm_new,
                alpha_used,
            )
            x = x_new

        raise RuntimeError(f"Newton did not converge after {max_attempts} iterations (|R|={r_norm:.2e}).")


class _OrbitCorrector(_Corrector):
    """Periodic-orbit specific helper that delegates to :class:`_NewtonCorrector`."""

    def __init__(self, core: _NewtonCorrector | None = None):
        # Allow dependency injection of the generic Newton solver
        self._core = core or _NewtonCorrector()

    def correct(
        self,
        orbit: "PeriodicOrbit",
        *,
        tol: float = 1e-10,
        max_attempts: int = 25,
        forward: int = 1,
        max_delta: float | None = None,
        alpha_reduction: float = 0.5,
        min_alpha: float = 1e-4,
        armijo_c: float = 0.02,
        finite_difference: bool = False,
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
            max_delta=max_delta,
            alpha_reduction=alpha_reduction,
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