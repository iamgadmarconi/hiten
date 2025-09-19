"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
to specific problem domains. These interfaces handle the translation between
domain objects (orbits, manifolds) and the abstract vector representations
expected by the correction algorithms.
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import time

from hiten.algorithms.corrector.base import JacobianFn, NormFn, _Corrector
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit



class _PeriodicOrbitCorrectorInterface(_Corrector):
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
        orbit : PeriodicOrbit
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
        orbit : PeriodicOrbit
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
        _t0 = time.perf_counter()
        x_full = self._to_full_state(base_state, control_indices, p_vec)

        # Evaluate event section
        _te0 = time.perf_counter()
        t_event, X_ev_local = self._evaluate_event(orbit, x_full, cfg, forward)
        _te1 = time.perf_counter()

        Phi_local: np.ndarray | None = None
        if not self._fd_mode:
            # Analytical Jacobian will be requested, compute STM now
            _ts0 = time.perf_counter()
            _, _, Phi_flat, _ = _compute_stm(
                orbit.libration_point._var_eq_system,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            _ts1 = time.perf_counter()
            Phi_local = Phi_flat

        # Update cache for potential reuse by Jacobian
        self._event_cache = self._EventCache(
            p_vec=p_vec.copy(),
            t_event=t_event,
            X_event=X_ev_local,
            Phi=Phi_local,
        )

        self._last_t_event = t_event
        print(f"[Corrector] residual: event={( _te1 - _te0)*1e3:.2f} ms, STM={( _ts1 - _ts0)*1e3:.2f} ms, total={( time.perf_counter() - _t0)*1e3:.2f} ms; t_event={t_event:.6g}")
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
        orbit : PeriodicOrbit
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

        _t0 = time.perf_counter()
        if cache_valid:
            # Reuse cached data
            X_ev_local = self._event_cache.X_event
            Phi = self._event_cache.Phi
        else:
            # Recompute event and STM, then refresh cache
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            _te0 = time.perf_counter()
            t_event, X_ev_local = self._evaluate_event(orbit, x_full, cfg, forward)
            _te1 = time.perf_counter()

            _ts0 = time.perf_counter()
            _, _, Phi_flat, _ = _compute_stm(
                orbit.libration_point._var_eq_system,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            _ts1 = time.perf_counter()
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
        print(f"[Corrector] jacobian: cache={cache_valid}, total={( time.perf_counter() - _t0)*1e3:.2f} ms")
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
        """Correct periodic orbit to satisfy Poincare section constraints.
        
        Main entry point for orbit correction. Extracts parameters from orbit
        configuration, builds residual and Jacobian functions, delegates to
        numerical corrector, and updates the orbit with corrected values.
        
        Parameters
        ----------
        orbit : PeriodicOrbit
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
            Half-period (time to Poincare section crossing).
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
    """Provide an interface for invariant tori correction (placeholder).
    
    Reserved for future implementation of invariant tori correction
    algorithms.
    """
    pass