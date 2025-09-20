"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
to specific problem domains. These interfaces handle the translation between
domain objects (orbits, manifolds) and the abstract vector representations
expected by the correction algorithms.
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple, Callable

import numpy as np

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.types import JacobianFn, NormFn
from hiten.algorithms.dynamics.rtbp import _compute_stm

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit



class _PeriodicOrbitCorrectorInterface(_CorrectorBackend):
    """Provide an interface for periodic orbit differential correction.
    
    Provides orbit-specific correction functionality designed to be used as a
    mixin with concrete corrector implementations. Handles parameter extraction,
    constraint formulation, and Jacobian computation for periodic orbits.
    """
    @dataclass(slots=True)
    class _EventCache:
        """Cache for expensive event and STM computations.
        
        Attributes
        ----------
        p_vec : ndarray
            Parameter vector for which cache is valid.
        t_event : float
            Time of Poincare section crossing.
        X_event : ndarray
            State at Poincare section crossing.
        Phi : ndarray or None
            State transition matrix (None for finite-difference mode).
        """
        p_vec: np.ndarray
        t_event: float
        X_event: np.ndarray
        Phi: np.ndarray | None  # None when finite-difference Jacobian is used

    _event_cache: _EventCache | None = None  # initialised lazily
    _fd_mode: bool = False  # finite-difference mode flag set per correction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_cache = None

    def _to_full_state(
        self,
        base_state: np.ndarray,
        control_indices: list[int],
        p_vec: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct full state from base state and parameter vector.
        
        Parameters
        ----------
        base_state : ndarray
            Base 6D state vector.
        control_indices : list of int
            Indices of components to update.
        p_vec : ndarray
            Parameter vector with new values.
            
        Returns
        -------
        ndarray
            Full 6D state with updated components.
        """
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
        """Evaluate Poincare section crossing.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit object containing system information.
        x_full : ndarray
            Initial state for integration.
        cfg : _OrbitCorrectionConfig
            Configuration with event function.
        forward : int
            Integration direction.
            
        Returns
        -------
        t_event : float
            Time of section crossing.
        X_event : ndarray
            State at section crossing.
        """
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
        """Compute residual vector for orbit correction.
        
        Evaluates the difference between the actual state at Poincare section
        crossing and the target values for selected components.
        
        Parameters
        ----------
        p_vec : ndarray
            Current parameter vector.
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit being corrected.
        base_state : ndarray
            Base state vector.
        control_indices : list of int
            Indices of parameters being optimized.
        residual_indices : list of int
            Indices of state components in residual.
        target_vec : ndarray
            Target values for residual components.
        cfg : _OrbitCorrectionConfig
            Correction configuration.
        forward : int
            Integration direction.
            
        Returns
        -------
        ndarray
            Residual vector (actual - target).
        """
        x_full = self._to_full_state(base_state, control_indices, p_vec)
        t_event, X_ev_local = self._evaluate_event(orbit, x_full, cfg, forward)
        Phi_local: np.ndarray | None = None
        if not self._fd_mode:
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
        """Compute analytical Jacobian using state transition matrix.
        
        Uses cached STM when available or computes new STM and updates cache.
        Extracts the relevant submatrix corresponding to residual and control
        indices.
        
        Parameters
        ----------
        p_vec : ndarray
            Current parameter vector.
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit being corrected.
        base_state : ndarray
            Base state vector.
        control_indices : list of int
            Indices of parameters being optimized.
        residual_indices : list of int
            Indices of state components in residual.
        cfg : _OrbitCorrectionConfig
            Correction configuration.
        forward : int
            Integration direction.
            
        Returns
        -------
        ndarray
            Jacobian matrix of residual with respect to parameters.
        """
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

        # Extract relevant submatrix
        J_red = Phi[np.ix_(residual_indices, control_indices)]

        if cfg.extra_jacobian is not None:
            J_red -= cfg.extra_jacobian(X_ev_local, Phi)
        return J_red

    # --- Newton hooks for rich diagnostics without changing solver core ---
    def _on_iteration(self, k: int, x: np.ndarray, r_norm: float) -> None:
        if not getattr(self, "_enable_diagnostics", False):
            return
        # diagnostics disabled

    def _on_accept(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        return

    def _on_failure(self, x: np.ndarray, *, iterations: int, residual_norm: float) -> None:
        return

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
        """Correct periodic orbit to satisfy Poincare section constraints.
        
        Main entry point for orbit correction. Extracts parameters from orbit
        configuration, builds residual and Jacobian functions, delegates to
        numerical corrector, and updates the orbit with corrected values.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        tol : float, default=1e-10
            Convergence tolerance for residual norm.
        max_attempts : int, default=25
            Maximum number of correction iterations.
        forward : int, default=1
            Integration direction (1 for forward, -1 for backward).
        max_delta : float or None, default=1e-2
            Maximum step size for numerical stability.
        finite_difference : bool, default=False
            Use finite-difference Jacobian instead of analytical.
            
        Returns
        -------
        x_corr : ndarray
            Corrected initial state.
        t_event : float
            Half-period (time to Poincare section crossing),
            nondimensional.
        """
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
        # _NewtonBackend.
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

        p_corr, info = super().correct( 
            x0=p0,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=_norm_inf,
            tol=tol,
            max_attempts=max_attempts,
            max_delta=max_delta,
        )

        x_corr = self._to_full_state(base_state, control_indices, p_corr)
        # Recompute half-period from corrected state using default window (no hints),
        # matching old pipeline semantics
        try:
            t_final, _ = cfg.event_func(
                dynsys=orbit.system._dynsys,
                x0=x_corr,
                forward=forward,
            )
            self._last_t_event = float(t_final)
        except Exception:
            # Fallback to evaluating via interface if direct call fails
            if self._last_t_event is None:
                self._last_t_event, _ = self._evaluate_event(
                    orbit,
                    x_corr,
                    cfg,
                    forward,
                )

        orbit._reset()
        orbit._initial_state = x_corr
        orbit._period = 2.0 * self._last_t_event

        # silent in normal runs

        return x_corr, self._last_t_event


class _PeriodicOrbitInterface:
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