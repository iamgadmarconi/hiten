"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
to specific problem domains. These interfaces handle the translation between
domain objects (orbits, manifolds) and the abstract vector representations
expected by the correction algorithms.
"""

from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.types import JacobianFn, NormFn
from hiten.algorithms.dynamics.rtbp import _compute_stm

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitCorrectorInterface:
    """Stateless adapter for periodic orbit correction.
    
    Produces residual and Jacobian closures and provides helpers to translate
    between parameter vectors and full states. Contains no mutable state.
    """

    def initial_guess(self, orbit: "PeriodicOrbit", cfg) -> np.ndarray:
        control_indices = list(cfg.control_indices)
        return orbit.initial_state[control_indices].copy()

    def residual_fn(self, orbit: "PeriodicOrbit", cfg, forward: int) -> Callable[[np.ndarray], np.ndarray]:
        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        # Closure-local cache (optional reuse for sibling Jacobian if desired)
        cache: dict[str, np.ndarray | float | None] = {"p": None, "t": None, "X": None, "Phi": None}

        def _residual(p_vec: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
            Phi_local: np.ndarray | None = None
            if not getattr(cfg, "finite_difference", False):
                _, _, Phi_flat, _ = _compute_stm(
                    orbit.libration_point._var_eq_system,
                    x_full,
                    t_event,
                    steps=cfg.steps,
                    method=cfg.method,
                    order=cfg.order,
                )
                Phi_local = Phi_flat
            cache["p"] = p_vec.copy()
            cache["t"] = float(t_event)
            cache["X"] = X_event
            cache["Phi"] = Phi_local
            return X_event[residual_indices] - target_vec

        _residual._cache = cache  # type: ignore[attr-defined]
        return _residual

    def jacobian_fn(self, orbit: "PeriodicOrbit", cfg, forward: int) -> JacobianFn | None:
        if getattr(cfg, "finite_difference", False):
            return None

        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)

        def _jacobian(p_vec: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
            _, _, Phi_flat, _ = _compute_stm(
                orbit.libration_point._var_eq_system,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            J_red = Phi_flat[np.ix_(residual_indices, control_indices)]
            if cfg.extra_jacobian is not None:
                J_red -= cfg.extra_jacobian(X_event, Phi_flat)
            return J_red

        return _jacobian

    def norm_fn(self) -> NormFn:
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def build_functions(
        self,
        orbit: "PeriodicOrbit",
        cfg,
        forward: int,
        *,
        finite_difference: bool,
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], JacobianFn | None, Callable[[np.ndarray], np.ndarray]]:
        """Create residual and Jacobian closures with shared cache and a to_full_state helper.

        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.
        forward : int
            Forward integration direction.
        finite_difference : bool
            Use finite-difference Jacobian instead of analytical.
        
        Returns
        -------
        residual_fn : Callable[[np.ndarray], np.ndarray]
            Residual function.
        jacobian_fn : JacobianFn | None
            Jacobian function.
        to_full_state_fn : Callable[[np.ndarray], np.ndarray]
            Function to convert parameter vector to full state.
        """
        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        cache: dict[str, np.ndarray | float | None] = {"p": None, "t": None, "X": None, "Phi": None}

        def to_full_state(p_vec: np.ndarray) -> np.ndarray:
            x_full = base_state.copy()
            x_full[control_indices] = p_vec
            return x_full

        def residual_fn(p_vec: np.ndarray) -> np.ndarray:
            x_full = to_full_state(p_vec)
            t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
            Phi_local: np.ndarray | None = None
            if not finite_difference:
                _, _, Phi_flat, _ = _compute_stm(
                    orbit.libration_point._var_eq_system,
                    x_full,
                    t_event,
                    steps=cfg.steps,
                    method=cfg.method,
                    order=cfg.order,
                )
                Phi_local = Phi_flat
            cache["p"] = p_vec.copy()
            cache["t"] = float(t_event)
            cache["X"] = X_event
            cache["Phi"] = Phi_local
            return X_event[residual_indices] - target_vec

        if finite_difference:
            jacobian_fn = None
        else:
            def jacobian_fn(p_vec: np.ndarray) -> np.ndarray:
                # Reuse cache if same p and Phi available
                if (cache["p"] is not None) and np.array_equal(cache["p"], p_vec) and (cache["Phi"] is not None):
                    X_event = cache["X"]  # type: ignore[assignment]
                    Phi_flat = cache["Phi"]  # type: ignore[assignment]
                else:
                    x_full = to_full_state(p_vec)
                    t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
                    _, _, Phi_flat, _ = _compute_stm(
                        orbit.libration_point._var_eq_system,
                        x_full,
                        t_event,
                        steps=cfg.steps,
                        method=cfg.method,
                        order=cfg.order,
                    )
                    cache["p"] = p_vec.copy()
                    cache["t"] = float(t_event)
                    cache["X"] = X_event
                    cache["Phi"] = Phi_flat

                J_red = Phi_flat[np.ix_(residual_indices, control_indices)]  # type: ignore[index]
                if cfg.extra_jacobian is not None:
                    J_red -= cfg.extra_jacobian(X_event, Phi_flat)  # type: ignore[arg-type]
                return J_red

        return residual_fn, jacobian_fn, to_full_state

    def compute_half_period(self, orbit: "PeriodicOrbit", corrected_state: np.ndarray, cfg, forward: int) -> float:
        try:
            t_final, _ = cfg.event_func(
                dynsys=orbit.system._dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_final)
        except Exception:
            t_fallback, _ = self._evaluate_event(orbit, corrected_state, cfg, forward)
            return float(t_fallback)

    def apply_results_to_orbit(self, orbit: "PeriodicOrbit", *, corrected_state: np.ndarray, half_period: float) -> None:
        orbit._reset()
        orbit._initial_state = corrected_state
        orbit._period = 2.0 * half_period

    @staticmethod
    def _to_full_state(base_state: np.ndarray, control_indices: list[int], p_vec: np.ndarray) -> np.ndarray:
        x_full = base_state.copy()
        x_full[control_indices] = p_vec
        return x_full

    @staticmethod
    def _evaluate_event(orbit: "PeriodicOrbit", x_full: np.ndarray, cfg, forward: int) -> Tuple[float, np.ndarray]:
        return cfg.event_func(
            dynsys=orbit.system._dynsys,
            x0=x_full,
            forward=forward,
        )