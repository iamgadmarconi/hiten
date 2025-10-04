"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
to specific problem domains. These interfaces handle the translation between
domain objects (orbits, manifolds) and the abstract vector representations
expected by the correction algorithms.
"""

from typing import TYPE_CHECKING, Sequence

import numpy as np

from hiten.algorithms.corrector.config import (
    MultipleShootingOrbitCorrectionConfig, OrbitCorrectionConfig)
from hiten.algorithms.corrector.operators import (
    _MultipleShootingOrbitOperators, _SingleShootingOrbitOperators)
from hiten.algorithms.corrector.options import (
    MultipleShootingCorrectionOptions, OrbitCorrectionOptions)
from hiten.algorithms.corrector.types import (CorrectorInput,
                                              CorrectorOutput,
                                              MultipleShootingDomainPayload,
                                              MultipleShootingResult, NormFn,
                                              OrbitCorrectionDomainPayload,
                                              OrbitCorrectionResult,
                                              StepperFactory,
                                              _MultipleShootingProblem,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _OrbitCorrectionInterfaceBase:
    """Shared helpers for orbit correction interfaces.

    Contains common utilities used by both single-shooting and
    multiple-shooting interfaces (norm policy and half-period
    computation from an event with safe fallback).
    """

    def _norm_fn(self) -> NormFn:
        """Infinity norm emphasizing max constraint violation."""
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def _half_period(
        self,
        domain_obj: "PeriodicOrbit",
        corrected_state: np.ndarray,
        problem,
    ) -> float:
        """Compute half-period using the problem's event function.

        Falls back to `patch_times[-1]` when available (multiple shooting)
        if event evaluation fails.
        """
        forward = getattr(problem, "forward", 1)
        try:
            t_final, _ = problem.event_func(
                dynsys=domain_obj.dynamics.dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_final)
        except Exception as exc:
            if hasattr(problem, "patch_times"):
                logger.warning(
                    f"Failed to compute half-period via event detection: {exc}. "
                    f"Falling back to patch_times estimate."
                )
                return float(problem.patch_times[-1])
            raise ValueError("Failed to evaluate event for corrected state") from exc

    @staticmethod
    def _reconstruct_full_state(
        template: np.ndarray,
        control_indices: Sequence[int],
        params: np.ndarray,
    ) -> np.ndarray:
        """Inject control parameters into a template state."""
        x_full = template.copy()
        x_full[list(control_indices)] = params
        return x_full


class _OrbitCorrectionInterface(
    _OrbitCorrectionInterfaceBase,
    _HitenBaseInterface[
        OrbitCorrectionConfig,
        _OrbitCorrectionProblem,
        OrbitCorrectionResult,
        CorrectorOutput,
    ]
):
    """Adapter wiring periodic orbits to the Newton correction backend."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self, 
        *, 
        domain_obj: "PeriodicOrbit", 
        config: OrbitCorrectionConfig,
        options: OrbitCorrectionOptions,
        stepper_factory: StepperFactory | None = None
    ) -> _OrbitCorrectionProblem:
        """Create a correction problem.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object to correct.
        config : :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            Compile-time configuration (algorithm structure).
        options : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`, optional
            Runtime options (tuning parameters). If None, defaults are used.
        stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` or None
            The stepper factory for the correction problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The correction problem.
        """
        # Build operators
        ops = _SingleShootingOrbitOperators(
            domain_obj=domain_obj,
            control_indices=config.control_indices,
            residual_indices=config.residual_indices,
            target=config.target,
            extra_jacobian=config.extra_jacobian,
            event_func=config.event_func,
            forward=options.forward,
            method=config.integration.method,
            order=options.base.integration.order,
            steps=options.base.integration.steps,
        )
        
        # Build residual/Jacobian from operators
        residual_fn = ops.build_residual_fn()
        jacobian_fn = None if config.numerical.finite_difference else ops.build_jacobian_fn()
        norm_fn = self._norm_fn()
        initial_guess = self._initial_guess(domain_obj, config)
        
        problem = _OrbitCorrectionProblem(
            initial_guess=initial_guess,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=norm_fn,
            max_attempts=options.base.convergence.max_attempts,
            tol=options.base.convergence.tol,
            max_delta=options.base.convergence.max_delta,
            finite_difference=config.numerical.finite_difference,
            fd_step=options.base.numerical.fd_step,
            method=config.integration.method,
            order=options.base.integration.order,
            steps=options.base.integration.steps,
            forward=options.forward,
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
        request = CorrectorInput(
            initial_guess=problem.initial_guess,
            residual_fn=problem.residual_fn,
            jacobian_fn=problem.jacobian_fn,
            norm_fn=problem.norm_fn,
            max_attempts=problem.max_attempts,
            tol=problem.tol,
            max_delta=problem.max_delta,
            fd_step=problem.fd_step,
        )
        return _BackendCall(request=request, kwargs={"stepper_factory": problem.stepper_factory})

    def to_domain(self, outputs: CorrectorOutput, *, problem: _OrbitCorrectionProblem) -> OrbitCorrectionDomainPayload:
        """Convert backend outputs to domain payload."""
        x_corr = outputs.x_corrected
        iterations = outputs.iterations
        residual_norm = outputs.residual_norm
        control_indices = list(problem.control_indices)
        base_state = problem.domain_obj.initial_state
        x_full = self._reconstruct_full_state(base_state, control_indices, x_corr)
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        return OrbitCorrectionDomainPayload._from_mapping(
            {
                "iterations": int(iterations),
                "residual_norm": float(residual_norm),
                "half_period": float(half_period),
                "x_full": np.asarray(x_full, dtype=float),
            }
        )

    def to_results(
        self,
        outputs: CorrectorOutput,
        *,
        problem: _OrbitCorrectionProblem,
        domain_payload: OrbitCorrectionDomainPayload | None = None,
    ) -> OrbitCorrectionResult:
        """Package backend outputs into an :class:`OrbitCorrectionResult`."""

        payload = domain_payload or self.to_domain(outputs, problem=problem)
        return OrbitCorrectionResult(
            converged=True,
            x_corrected=payload.x_full,
            residual_norm=float(payload.residual_norm),
            iterations=int(payload.iterations),
            half_period=payload.half_period,
        )

    def _initial_guess(self, domain_obj: "PeriodicOrbit", cfg: OrbitCorrectionConfig) -> np.ndarray:
        """Get the initial guess.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The configuration.
        
        Returns
        -------
        :class:`~numpy.ndarray`
            The initial guess.
        """
        indices = list(cfg.control_indices)
        return domain_obj.initial_state[indices].copy()


class _MultipleShootingOrbitCorrectionInterface(
    _OrbitCorrectionInterfaceBase,
    _HitenBaseInterface[
        MultipleShootingOrbitCorrectionConfig,
        _MultipleShootingProblem,
        MultipleShootingResult,
        CorrectorOutput,
    ]
):
    def __init__(self) -> None:
        super().__init__()
