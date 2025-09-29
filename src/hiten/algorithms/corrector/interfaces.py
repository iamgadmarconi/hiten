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
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitCorrectorInterface(
    _HitenBaseInterface[
        _OrbitCorrectionConfig,
        _OrbitCorrectionProblem,
        OrbitCorrectionResult,
        tuple[np.ndarray, int, float],
    ]
):
    """Adapter wiring periodic orbits to the Newton correction backend."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *, 
        domain_obj: "PeriodicOrbit", 
        config: _OrbitCorrectionConfig, 
        stepper_factory: StepperFactory | None = None
    ) -> _OrbitCorrectionProblem:
        """Create a correction problem.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object to correct.
        config : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The configuration for the correction problem.
        stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` or None
            The stepper factory for the correction problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        """
        forward = getattr(config, "forward", 1)
        residual_fn = self._residual_fn(domain_obj, config, forward)
        jacobian_fn = self._jacobian_fn(domain_obj, config, forward)
        norm_fn = self._norm_fn()
        initial_guess = self._initial_guess(domain_obj, config)
        problem = _OrbitCorrectionProblem(
            initial_guess=initial_guess,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=norm_fn,
            max_attempts=config.max_attempts,
            tol=config.tol,
            max_delta=config.max_delta,
            line_search_config=config.line_search_config,
            finite_difference=config.finite_difference,
            fd_step=config.fd_step,
            method=config.method,
            order=config.order,
            steps=config.steps,
            forward=config.forward,
            stepper_factory=stepper_factory,
            domain_obj=domain_obj,
            residual_indices=config.residual_indices,
            control_indices=config.control_indices,
            extra_jacobian=config.extra_jacobian,
            target=config.target,
            event_func=config.event_func,
        )
        return problem

    def to_backend_inputs(self, problem: _OrbitCorrectionProblem) -> _BackendCall:
        """Convert a correction problem to backend inputs.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend inputs.
        """
        return _BackendCall(
            args=(problem.initial_guess,),
            kwargs={
                "residual_fn": problem.residual_fn,
                "jacobian_fn": problem.jacobian_fn,
                "norm_fn": problem.norm_fn,
                "stepper_factory": problem.stepper_factory,
                "tol": problem.tol,
                "max_attempts": problem.max_attempts,
                "max_delta": problem.max_delta,
                "fd_step": problem.fd_step,
            },
        )

    def to_domain(self, outputs: tuple[np.ndarray, int, float], *, problem: _OrbitCorrectionProblem) -> dict[str, Any]:
        """Convert backend outputs to domain results.
        
        Parameters
        ----------
        outputs : tuple of :class:`~numpy.ndarray`, int, float
            The backend outputs.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        
        Returns
        -------
        dict of str, Any
            The domain results.
        """
        x_corr, iterations, residual_norm = outputs
        control_indices = list(problem.control_indices)
        base_state = problem.domain_obj.initial_state.copy()
        x_full = self._to_full_state(base_state, control_indices, x_corr)
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        
        problem.domain_obj.dynamics.reset()
        problem.domain_obj.dynamics._initial_state = x_full
        problem.domain_obj.dynamics.period = 2.0 * half_period
        
        return {
            "iterations": iterations,
            "residual_norm": residual_norm,
            "half_period": half_period,
            "x_full": x_full
        }

    def to_results(self, outputs: tuple[np.ndarray, int, float], *, problem: _OrbitCorrectionProblem, domain_payload: dict[str, Any] = None) -> OrbitCorrectionResult:
        """Convert backend outputs to domain results.
        
        Parameters
        ----------
        outputs : tuple of :class:`~numpy.ndarray`, int, float
            The backend outputs.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        domain_payload : dict of str, Any
            The domain payload.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.OrbitCorrectionResult`
            The domain results.
        """
        x_corr, iterations, residual_norm = outputs
        control_indices = list(problem.control_indices)
        base_state = problem.domain_obj.initial_state.copy()
        x_full = self._to_full_state(base_state, control_indices, x_corr)
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        
        return OrbitCorrectionResult(
            converged=True,
            x_corrected=x_full,
            residual_norm=float(residual_norm),
            iterations=int(iterations),
            half_period=half_period,
        )

    def _initial_guess(self, domain_obj: "PeriodicOrbit", cfg: _OrbitCorrectionConfig) -> np.ndarray:
        """Get the initial guess.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The configuration.
        
        Returns
        -------
        :class:`~numpy.ndarray`
            The initial guess.
        """
        indices = list(cfg.control_indices)
        return domain_obj.initial_state[indices].copy()

    def _norm_fn(self) -> NormFn:
        """Get the norm function.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.NormFn`
            The norm function.
        """
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def _residual_fn(self, domain_obj: "PeriodicOrbit", cfg: _OrbitCorrectionConfig, forward: int) -> Callable[[np.ndarray], np.ndarray]:
        """Get the residual function.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The configuration.
        forward : int
            The forward direction.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.ResidualFn`
            The residual function.
        """
        base_state = domain_obj.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)
        target_vec = np.asarray(cfg.target, dtype=float)

        def _fn(params: np.ndarray) -> np.ndarray:
            """Get the residual function.
            
            Parameters
            ----------
            params : np.ndarray
                The parameters.
            
            Returns
            -------
            np.ndarray
                The residual.
            """
            x_full = self._to_full_state(base_state, control_indices, params)
            _, x_event = self._evaluate_event(domain_obj, x_full, cfg, forward)
            return x_event[residual_indices] - target_vec

        return _fn

    def _jacobian_fn(self, domain_obj: "PeriodicOrbit", cfg: _OrbitCorrectionConfig, forward: int) -> JacobianFn | None:
        """Get the Jacobian function.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The configuration.
        forward : int
            The forward direction.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.JacobianFn` | None
            The Jacobian function.
        """
        if bool(getattr(cfg, "finite_difference", False)):
            return None

        base_state = domain_obj.initial_state.copy()
        control_indices = list(cfg.control_indices)
        residual_indices = list(cfg.residual_indices)

        def _fn(params: np.ndarray) -> np.ndarray:
            """Get the Jacobian function.
            
            Parameters
            ----------
            params : np.ndarray
                The parameters.
            
            Returns
            -------
            np.ndarray
                The Jacobian.
            """
            x_full = self._to_full_state(base_state, control_indices, params)
            # Create a temporary problem object for _evaluate_event
            temp_problem = _OrbitCorrectionProblem(
                initial_guess=np.array([]),
                residual_fn=lambda x: x,
                jacobian_fn=None,
                norm_fn=None,
                max_attempts=0,
                tol=0.0,
                max_delta=0.0,
                line_search_config=None,
                finite_difference=False,
                fd_step=0.0,
                method=cfg.method,
                order=cfg.order,
                steps=cfg.steps,
                forward=forward,
                stepper_factory=None,
                domain_obj=domain_obj,
                residual_indices=cfg.residual_indices,
                control_indices=cfg.control_indices,
                extra_jacobian=cfg.extra_jacobian,
                target=cfg.target,
                event_func=cfg.event_func,
            )
            t_event, x_event = self._evaluate_event(domain_obj, x_full, temp_problem, forward)
            _, _, Phi_flat, _ = _compute_stm(
                domain_obj.dynamics.var_dynsys,
                x_full,
                t_event,
                steps=cfg.steps,
                method=cfg.method,
                order=cfg.order,
            )
            jac = Phi_flat[np.ix_(residual_indices, control_indices)]
            if cfg.extra_jacobian is not None:
                jac -= cfg.extra_jacobian(x_event, Phi_flat)
            return jac

        return _fn

    def _half_period(self, domain_obj: "PeriodicOrbit", corrected_state: np.ndarray, problem: _OrbitCorrectionProblem) -> float:
        """Get the half period.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        corrected_state : np.ndarray
            The corrected state.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        
        Returns
        -------
        float
            The half period.
        """
        forward = problem.forward
        try:
            t_final, _ = problem.event_func(
                dynsys=domain_obj.dynamics.dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_final)
        except Exception:
            try:
                fallback, _ = self._evaluate_event(domain_obj, corrected_state, problem, forward)
                return float(fallback)
            except Exception as exc:
                raise ValueError("Failed to evaluate domain_obj event for corrected state") from exc

    def _to_full_state(self, base_state: np.ndarray, control_indices: list[int], params: np.ndarray) -> np.ndarray:
        """Get the full state.
        
        Parameters
        ----------
        base_state : np.ndarray
            The base state.
        control_indices : list[int]
            The control indices.
        params : np.ndarray
            The parameters.
        
        Returns
        -------
        np.ndarray
            The full state.
        """
        x_full = base_state.copy()
        x_full[control_indices] = params
        return x_full

    def _evaluate_event(
        self,
        domain_obj: "PeriodicOrbit",
        full_state: np.ndarray,
        problem: _OrbitCorrectionProblem,
        forward: int,
    ) -> tuple[float, np.ndarray]:
        """Get the event function.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        full_state : np.ndarray
            The full state.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        forward : int
            The forward direction.
        
        Returns
        -------
        tuple[float, np.ndarray]
            The event function.
        """
        return problem.event_func(
            dynsys=domain_obj.dynamics.dynsys,
            x0=full_state,
            forward=forward,
        )
