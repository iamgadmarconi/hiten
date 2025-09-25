"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
 to specific problem domains. These interfaces handle the translation between
 domain objects (orbits, manifolds) and the abstract vector representations
 expected by the correction algorithms.
"""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
from hiten.algorithms.corrector.types import (JacobianFn, NormFn,
                                              OrbitCorrectionResult,
                                              StepperFactory,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.types.core import BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitCorrectorInterface(
    _HitenBaseInterface[
        "PeriodicOrbit",
        _OrbitCorrectionConfig,
        _OrbitCorrectionProblem,
        OrbitCorrectionResult,
        tuple[np.ndarray, dict[str, Any]],
    ]
):
    """Adapter wiring periodic orbits to the Newton correction backend."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *, 
        orbit: "PeriodicOrbit", 
        config: _OrbitCorrectionConfig, 
        stepper_factory: StepperFactory | None = None
    ) -> _OrbitCorrectionProblem:
        forward = getattr(config, "forward", 1)
        residual_fn = self._residual_fn(orbit, config, forward)
        jacobian_fn = self._jacobian_fn(orbit, config, forward)
        norm_fn = self._norm_fn()
        initial_guess = self._initial_guess(orbit, config)
        problem = _OrbitCorrectionProblem(
            initial_guess=initial_guess,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=norm_fn,
            stepper_factory=stepper_factory,
            orbit=orbit,
            cfg=config,
        )
        return problem

    def to_backend_inputs(self, problem: _OrbitCorrectionProblem) -> BackendCall:
        return BackendCall(
            args=(problem.initial_guess,),
            kwargs={
                "residual_fn": problem.residual_fn,
                "jacobian_fn": problem.jacobian_fn,
                "norm_fn": problem.norm_fn,
                "stepper_factory": problem.stepper_factory,
                "tol": problem.cfg.tol,
                "max_attempts": problem.cfg.max_attempts,
                "max_delta": problem.cfg.max_delta,
                "fd_step": problem.cfg.fd_step,
            },
        )

    def to_domain(self,outputs: tuple[np.ndarray, dict[str, Any]], *, problem: _OrbitCorrectionProblem) -> dict[str, Any]:
        x_corr, info = outputs
        control_indices = list(problem.cfg.control_indices)
        base_state = problem.orbit.initial_state.copy()
        x_full = self._to_full_state(base_state, control_indices, x_corr)
        half_period = self._half_period(problem.orbit, x_full, problem.cfg)
        problem.orbit._reset()
        problem.orbit._initial_state = x_full
        problem.orbit._period = 2.0 * half_period
        info = dict(info)
        info["half_period"] = half_period
        info["x_full"] = x_full
        return info

    def to_results(self, outputs: tuple[np.ndarray, dict[str, Any]], *, problem: _OrbitCorrectionProblem) -> OrbitCorrectionResult:
        x_corr, info = outputs
        info = dict(info)
        iterations = int(info.get("iterations", 0))
        residual_norm = float(info.get("residual_norm", np.nan))
        half_period = float(info.get("half_period", np.nan))
        x_full = info.get("x_full")
        if x_full is None:
            control_indices = list(problem.cfg.control_indices)
            base_state = problem.orbit.initial_state.copy()
            x_full = self._to_full_state(base_state, control_indices, x_corr)
        return OrbitCorrectionResult(
            converged=True,
            x_corrected=x_full,
            residual_norm=residual_norm,
            iterations=iterations,
            half_period=half_period,
        )

    def _initial_guess(self, orbit: "PeriodicOrbit", cfg: _OrbitCorrectionConfig) -> np.ndarray:
        indices = list(cfg.control_indices)
        return orbit.initial_state[indices].copy()

    def _norm_fn(self) -> NormFn:
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def _residual_fn(self, orbit: "PeriodicOrbit", cfg: _OrbitCorrectionConfig, forward: int) -> Callable[[np.ndarray], np.ndarray]:
        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        def _fn(params: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, params)
            _, x_event = self._evaluate_event(orbit, x_full, cfg, forward)
            return x_event[residual_indices] - target_vec

        return _fn

    def _jacobian_fn(self, orbit: "PeriodicOrbit", cfg: _OrbitCorrectionConfig, forward: int) -> JacobianFn | None:
        if bool(getattr(cfg, "finite_difference", False)):
            return None

        base_state = orbit.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)

        def _fn(params: np.ndarray) -> np.ndarray:
            x_full = self._to_full_state(base_state, control_indices, params)
            t_event, x_event = self._evaluate_event(orbit, x_full, cfg, forward)
            _, _, Phi_flat, _ = _compute_stm(
                orbit.var_dynsys,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            jac = Phi_flat[np.ix_(residual_indices, control_indices)]  # type: ignore[index]
            if cfg.extra_jacobian is not None:
                jac -= cfg.extra_jacobian(x_event, Phi_flat)  # type: ignore[arg-type]
            return jac

        return _fn

    def _half_period(self, orbit: "PeriodicOrbit", corrected_state: np.ndarray, cfg: _OrbitCorrectionConfig) -> float:
        forward = getattr(cfg, "forward", 1)
        try:
            t_final, _ = cfg.event_func(
                dynsys=orbit.system.dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_final)
        except Exception:
            try:
                fallback, _ = self._evaluate_event(orbit, corrected_state, cfg, forward)
                return float(fallback)
            except Exception as exc:
                raise ValueError("Failed to evaluate orbit event for corrected state") from exc

    def _to_full_state(self, base_state: np.ndarray, control_indices: list[int], params: np.ndarray) -> np.ndarray:
        x_full = base_state.copy()
        x_full[control_indices] = params
        return x_full

    def _evaluate_event(
        self,
        orbit: "PeriodicOrbit",
        full_state: np.ndarray,
        cfg: _OrbitCorrectionConfig,
        forward: int,
    ) -> tuple[float, np.ndarray]:
        return cfg.event_func(
            dynsys=orbit.system.dynsys,
            x0=full_state,
            forward=forward,
        )
