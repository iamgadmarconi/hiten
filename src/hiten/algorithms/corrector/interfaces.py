"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
 to specific problem domains. These interfaces handle the translation between
 domain objects (orbits, manifolds) and the abstract vector representations
 expected by the correction algorithms.
"""

from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.types import (JacobianFn, NormFn,
                                              _OrbitCorrectionProblem)
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

    def norm_fn(self) -> NormFn:
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def residual_fn(self, orbit: "PeriodicOrbit", cfg, forward: int) -> Callable[[np.ndarray], np.ndarray]:
        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        def _residual(p_vec: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
            return X_event[residual_indices] - target_vec

        return _residual

    def jacobian_fn(self, orbit: "PeriodicOrbit", cfg, forward: int) -> JacobianFn | None:
        finite_difference = bool(getattr(cfg, "finite_difference", False))
        if finite_difference:
            return None

        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)

        def _jac(p_vec: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, p_vec)
            t_event, X_event = self._evaluate_event(orbit, x_full, cfg, forward)
            _, _, Phi_flat, _ = _compute_stm(
                orbit.var_dynsys,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            J_red = Phi_flat[np.ix_(residual_indices, control_indices)]  # type: ignore[index]
            if cfg.extra_jacobian is not None:
                J_red -= cfg.extra_jacobian(X_event, Phi_flat)  # type: ignore[arg-type]
            return J_red

        return _jac

    def create_problem(self, orbit: "PeriodicOrbit", cfg) -> _OrbitCorrectionProblem:
        """Compose a backend correction problem from a domain orbit and config.

        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            Immutable problem object containing initial guess, closures, and numeric params.
        """
        forward = getattr(cfg, "forward", 1)
        residual_fn = self.residual_fn(orbit, cfg, forward)
        jacobian_fn = self.jacobian_fn(orbit, cfg, forward)

        norm_fn = self.norm_fn()
        p0 = self.initial_guess(orbit, cfg)
        return _OrbitCorrectionProblem(
            initial_guess=p0,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=norm_fn,
            orbit=orbit,
            cfg=cfg,
        )

    def compute_half_period(self, orbit: "PeriodicOrbit", corrected_state: np.ndarray, cfg, forward: int) -> float:
        """Compute the half-period of a periodic orbit.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        corrected_state : np.ndarray
            Corrected state vector.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.
        forward : int
            Forward integration direction.
        """
        try:
            t_final, _ = cfg.event_func(
                dynsys=orbit.system.dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_final)

        except Exception:
            t_fallback, _ = self._evaluate_event(orbit, corrected_state, cfg, forward)
            return float(t_fallback)

    def apply_results_to_orbit(self, orbit: "PeriodicOrbit", *, corrected_state: np.ndarray, half_period: float) -> None:
        """Apply the results of the correction to the orbit.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        corrected_state : np.ndarray
            Corrected state vector.
        half_period : float
            Half-period of the orbit.
        """
        orbit._reset()
        orbit._initial_state = corrected_state
        orbit._period = 2.0 * half_period

    @staticmethod
    def _to_full_state(base_state: np.ndarray, control_indices: list[int], p_vec: np.ndarray) -> np.ndarray:
        """
        Convert a parameter vector to a full state vector.
        
        Parameters
        ----------
        base_state : np.ndarray
            Base state vector.
        control_indices : list[int]
            Indices of the control variables.
        p_vec : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Full state vector.
        """
        x_full = base_state.copy()
        x_full[control_indices] = p_vec
        return x_full

    @staticmethod
    def _evaluate_event(orbit: "PeriodicOrbit", x_full: np.ndarray, cfg, forward: int) -> Tuple[float, np.ndarray]:
        """Evaluate the event function.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        x_full : np.ndarray
            Full state vector.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.
        forward : int
            Forward integration direction.
        """
        return cfg.event_func(
            dynsys=orbit.system.dynsys,
            x0=x_full,
            forward=forward,
        )
