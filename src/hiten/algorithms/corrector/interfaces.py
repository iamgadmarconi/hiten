from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.corrector.base import (JacobianFn, NormFn,
                                             _BaseCorrectionConfig, _Corrector)
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.poincare.sh.backend import _y_plane_crossing
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


@dataclass(frozen=True, slots=True)
class _OrbitCorrectionConfig(_BaseCorrectionConfig):

    residual_indices: tuple[int, ...] = ()  # Components used to build R(x)
    control_indices: tuple[int, ...] = ()   # Components allowed to change
    extra_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    target: tuple[float, ...] = (0.0,)  # Desired residual values

    event_func: Callable[..., tuple[float, np.ndarray]] = _y_plane_crossing

    method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy"
    order: int = 8
    steps: int = 500

    forward: int = 1


class _PeriodicOrbitCorrectorInterface(_Corrector):
    """Interface for periodic orbit differential correction.
    
    This class provides orbit-specific correction functionality and is designed
    to be used as a mixin with a concrete corrector implementation (e.g., _NewtonCore).
    The orbit-specific correct() method translates orbit parameters to generic
    corrector parameters and delegates numerical work to the concrete implementation
    via super().correct().
    """
    @dataclass(slots=True)
    class _EventCache:
        p_vec: np.ndarray
        t_event: float
        X_event: np.ndarray
        Phi: np.ndarray | None  # None when finite-difference Jacobian is used

    _event_cache: _EventCache | None = None  # initialised lazily
    _fd_mode: bool = False  # finite-difference mode flag set per correction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure cache is reset for every new corrector instance
        self._event_cache = None

    def _to_full_state(
        self,
        base_state: np.ndarray,
        control_indices: list[int],
        p_vec: np.ndarray,
    ) -> np.ndarray:
        """Insert the parameter vector *p_vec* back into the full 6-D state."""
        x_full = base_state.copy()
        x_full[control_indices] = p_vec
        return x_full

    def _evaluate_event(
        self,
        orbit: "PeriodicOrbit",
        x_full: np.ndarray,
        cfg,
        forward: int,
    ) -> Tuple[float, np.ndarray]:
        """Call the section event and return (t_event, X_event)."""
        return cfg.event_func(
            dynsys=orbit.system._dynsys,
            x0=x_full,
            forward=forward,
        )

    _last_t_event: Optional[float] = None

    def _residual_vec(
        self,
        p_vec: np.ndarray,
        *,
        orbit: "PeriodicOrbit",
        base_state: np.ndarray,
        control_indices: list[int],
        residual_indices: list[int],
        target_vec: np.ndarray,
        cfg,
        forward: int,
    ) -> np.ndarray:
        """Default residual: event state minus target on selected indices."""
        x_full = self._to_full_state(base_state, control_indices, p_vec)

        # Evaluate event section
        t_event, X_ev_local = self._evaluate_event(orbit, x_full, cfg, forward)

        Phi_local: np.ndarray | None = None
        if not self._fd_mode:
            # Analytical Jacobian will be requested, compute STM now
            _, _, Phi_flat, _ = _compute_stm(
                orbit.libration_point._var_eq_system,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            Phi_local = Phi_flat

        # Update cache for potential reuse by Jacobian
        self._event_cache = self._EventCache(
            p_vec=p_vec.copy(),
            t_event=t_event,
            X_event=X_ev_local,
            Phi=Phi_local,
        )

        self._last_t_event = t_event
        return X_ev_local[residual_indices] - target_vec

    def _jacobian_mat(
        self,
        p_vec: np.ndarray,
        *,
        orbit: "PeriodicOrbit",
        base_state: np.ndarray,
        control_indices: list[int],
        residual_indices: list[int],
        cfg,
        forward: int,
    ) -> np.ndarray:
        """Analytical Jacobian using the state-transition matrix."""

        cache_valid = (
            self._event_cache is not None
            and np.array_equal(self._event_cache.p_vec, p_vec)
            and self._event_cache.Phi is not None
        )

        if cache_valid:
            # Reuse cached data
            X_ev_local = self._event_cache.X_event
            Phi = self._event_cache.Phi
        else:
            # Recompute event and STM, then refresh cache
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            t_event, X_ev_local = self._evaluate_event(orbit, x_full, cfg, forward)

            _, _, Phi_flat, _ = _compute_stm(
                orbit.libration_point._var_eq_system,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            Phi = Phi_flat

            self._event_cache = self._EventCache(
                p_vec=p_vec.copy(),
                t_event=t_event,
                X_event=X_ev_local,
                Phi=Phi.copy(),
            )

        # Phi is already 2-D array
        J_red = Phi[np.ix_(residual_indices, control_indices)]

        if cfg.extra_jacobian is not None:
            J_red -= cfg.extra_jacobian(X_ev_local, Phi)

        return J_red

    def correct(
        self,
        orbit: "PeriodicOrbit",
        *,
        tol: float = 1e-10,
        max_attempts: int = 25,
        forward: int = 1,
        max_delta: float | None = 1e-2,
        finite_difference: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Differential correction driver."""

        cfg = orbit._correction_config

        residual_indices = list(cfg.residual_indices)
        control_indices = list(cfg.control_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        # Reset event bookkeeping at the start of every correction run
        self._last_t_event = None

        # Record FD mode for caching logic
        self._fd_mode = finite_difference

        base_state = orbit.initial_state.copy()
        p0 = base_state[control_indices]

        # Build residual / Jacobian callables using *partial* to capture
        # constant arguments while keeping the signature expected by
        # _NewtonCore.
        residual_fn = partial(
            self._residual_vec,
            orbit=orbit,
            base_state=base_state,
            control_indices=control_indices,
            residual_indices=residual_indices,
            target_vec=target_vec,
            cfg=cfg,
            forward=forward,
        )

        jacobian_fn: JacobianFn | None = None
        if not finite_difference:
            jacobian_fn = partial(
                self._jacobian_mat,
                orbit=orbit,
                base_state=base_state,
                control_indices=control_indices,
                residual_indices=residual_indices,
                cfg=cfg,
                forward=forward,
            )

        # Infinity norm is the standard for orbit residuals
        _norm_inf: NormFn = lambda r: float(np.linalg.norm(r, ord=np.inf))

        # Delegate numerical work to the super-class (usually _NewtonCore)
        p_corr, info = super().correct( 
            x0=p0,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=_norm_inf,
            tol=tol,
            max_attempts=max_attempts,
            max_delta=max_delta,
        )

        # Ensure we have a valid half-period
        if self._last_t_event is None:
            self._last_t_event, _ = self._evaluate_event(
                orbit,
                self._to_full_state(base_state, control_indices, p_corr),
                cfg,
                forward,
            )
    
        x_corr = self._to_full_state(base_state, control_indices, p_corr)
        orbit._reset()
        orbit._initial_state = x_corr
        orbit._period = 2.0 * self._last_t_event

        logger.info(
            "Periodic-orbit corrector converged in %d iterations (|R|=%.2e)",
            info.get("iterations", -1),
            info.get("residual_norm", float("nan")),
        )

        return x_corr, self._last_t_event


class _InvariantToriCorrectorInterface:
    pass