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
    _MultipleShootingOrbitOperatorsImpl, _SingleShootingOrbitOperators)
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
        outputs: tuple[np.ndarray, int, float],
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
    """Adapter wiring periodic orbits to multiple shooting backend.

    This interface handles the transformation from single periodic orbit
    correction problems to multiple shooting problems with continuity
    constraints. It manages patch initialization, residual/Jacobian
    computation, and result conversion.

    The interface follows the adapter pattern to separate domain-specific
    logic (orbits, dynamics) from the generic Newton-Raphson solver.

    Notes
    -----
    The parameter vector structure is:

    .. math::

        \\mathbf{x} = [\\mathbf{x}_0^T, \\mathbf{x}_1^T, \\ldots, \\mathbf{x}_{N-1}^T]^T

    where each :math:`\\mathbf{x}_i` contains the control indices at patch i.

    The residual structure is:

    .. math::

        \\mathbf{R}(\\mathbf{x}) = \\begin{bmatrix}
            \\mathbf{x}_1^- - \\mathbf{x}_1 \\\\
            \\vdots \\\\
            \\mathbf{x}_{N-1}^- - \\mathbf{x}_{N-1} \\\\
            \\mathbf{x}_N^- - \\mathbf{x}_{\\text{target}}
        \\end{bmatrix}

    Examples
    --------
    >>> interface = _MultipleShootingOrbitCorrectionInterface()
    >>> config = MultipleShootingOrbitCorrectionConfig(n_patches=5)
    >>> problem = interface.create_problem(
    ...     domain_obj=orbit,
    ...     config=config,
    ...     stepper_factory=None
    ... )
    >>> # Backend solves the problem
    >>> result = interface.to_results(outputs, problem=problem)
    """

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: "PeriodicOrbit",
        config: MultipleShootingOrbitCorrectionConfig,
        options: MultipleShootingCorrectionOptions,
        stepper_factory: StepperFactory | None = None,
    ) -> _MultipleShootingProblem:
        """Create a multiple shooting correction problem.

        Initializes patches by sampling the trajectory and constructs
        residual and Jacobian functions with continuity constraints.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The periodic orbit to correct.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Compile-time configuration (algorithm structure).
        options : :class:`~hiten.algorithms.corrector.options.MultipleShootingCorrectionOptions`, optional
            Runtime options (tuning parameters). If None, defaults are used.
        stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` or None
            Optional step control factory.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The correction problem ready for backend solving.

        Notes
        -----
        This method performs patch initialization by:
        1. Estimating the period (if not provided)
        2. Creating time grid for patches
        3. Propagating initial trajectory
        4. Sampling states at patch times
        5. Extracting control indices
        """
        forward = options.forward

        # Initialize patches and times
        initial_guess, patch_times, patch_templates = self._initialize_patches(
            domain_obj, config, options
        )

        # Determine continuity indices
        continuity_indices = (
            config.continuity_indices
            if config.continuity_indices
            else config.control_indices
        )

        # Determine boundary indices
        boundary_indices = (
            config.boundary_only_indices if config.boundary_only_indices else config.residual_indices
        )

        # Build operators
        ops = _MultipleShootingOrbitOperatorsImpl(
            domain_obj=domain_obj,
            control_indices=config.control_indices,
            continuity_indices=continuity_indices,
            boundary_indices=boundary_indices,
            target=config.target,
            patch_times=patch_times,
            patch_templates=patch_templates,
            extra_jacobian=config.extra_jacobian,
            event_func=config.event_func,
            forward=forward,
            method=config.integration.method,
            order=options.base.integration.order,
            steps=options.base.integration.steps,
        )

        # Build residual/Jacobian from operators
        residual_fn = ops.build_residual_fn()
        jacobian_fn = None if config.numerical.finite_difference else ops.build_jacobian_fn()
        norm_fn = self._norm_fn()

        problem = _MultipleShootingProblem(
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
            n_patches=options.n_patches,
            patch_times=patch_times,
            patch_templates=patch_templates,
            patch_strategy=config.patch_strategy,
            residual_indices=config.residual_indices,
            control_indices=config.control_indices,
            continuity_indices=continuity_indices,
            boundary_indices=boundary_indices,
            target=config.target,
            event_func=config.event_func,
        )
        return problem

    def to_backend_inputs(self, problem: _MultipleShootingProblem) -> _BackendCall:
        """Convert correction problem to backend call format.

        Parameters
        ----------
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The correction problem.

        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            Backend call with args and kwargs.
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
            metadata={
                "n_patches": problem.n_patches,
                "patch_times": problem.patch_times,
                "continuity_indices": problem.continuity_indices,
                "boundary_indices": problem.boundary_indices,
            },
        )
        return _BackendCall(request=request, kwargs={"stepper_factory": problem.stepper_factory})

    def to_domain(
        self, outputs: CorrectorOutput, *, problem: _MultipleShootingProblem
    ) -> MultipleShootingDomainPayload:
        """Convert backend outputs to domain payload."""
        x_corr = outputs.x_corrected
        iterations = outputs.iterations
        residual_norm = outputs.residual_norm
        
        # Reshape corrected parameters into patch states
        patch_states = self._extract_patch_states(x_corr, problem)
        
        # Get the initial patch (full state)
        x_full = patch_states[0]
        
        # Compute half period
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        
        # Compute continuity errors for diagnostics
        continuity_errors = self._compute_continuity_errors(patch_states, problem)
        return MultipleShootingDomainPayload._from_mapping(
            {
                "iterations": int(iterations),
                "residual_norm": float(residual_norm),
                "half_period": float(half_period),
                "x_full": np.asarray(x_full, dtype=float),
                "patch_states": tuple(np.asarray(ps, dtype=float) for ps in patch_states),
                "patch_times": np.asarray(problem.patch_times, dtype=float),
                "continuity_errors": np.asarray(continuity_errors, dtype=float),
            }
        )

    def to_results(
        self,
        outputs: tuple[np.ndarray, int, float],
        *,
        problem: _MultipleShootingProblem,
        domain_payload: MultipleShootingDomainPayload | None = None,
    ) -> MultipleShootingResult:
        """Package backend outputs into a :class:`MultipleShootingResult`."""

        payload = domain_payload or self.to_domain(outputs, problem=problem)
        problem.domain_obj.services.correction.apply_correction(payload)
        return MultipleShootingResult(
            converged=True,
            x_corrected=payload.x_full,
            residual_norm=float(payload.residual_norm),
            iterations=int(payload.iterations),
            n_patches=problem.n_patches,
            patch_states=tuple(payload.patch_states),
            patch_times=np.asarray(payload.patch_times, dtype=float),
            continuity_errors=np.asarray(payload.continuity_errors, dtype=float),
            half_period=payload.half_period,
        )

    def _initialize_patches(
        self,
        domain_obj: "PeriodicOrbit",
        config: MultipleShootingOrbitCorrectionConfig,
        options,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Initialize patch states and times by sampling trajectory.

        Strategy:
        1. Estimate total period (or use provided period)
        2. Create patch time grid based on strategy
        3. Propagate initial guess trajectory
        4. Sample states at patch times
        5. Extract control indices from sampled states

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit to initialize patches from.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Compile-time configuration (algorithm structure).
        options : :class:`~hiten.algorithms.corrector.options.MultipleShootingCorrectionOptions`
            Runtime options (including n_patches).

        Returns
        -------
        patch_params : np.ndarray
            Flattened array of patch initial states [x_0, x_1, ..., x_n-1].
            Shape: (n_patches * n_control,)
        patch_times : np.ndarray
            Time values at patch boundaries [t_0, t_1, ..., t_n].
            Shape: (n_patches + 1,)
        patch_templates : list[np.ndarray]
            List of full-state templates evaluated along the trajectory at
            each patch boundary. Used to reconstruct full states during
            correction iterations.
        """
        n_patches = options.n_patches
        control_indices = list(config.control_indices)
        n_control = len(control_indices)

        # Determine patch times based on strategy
        if config.patch_strategy == "manual":
            patch_times = np.array(config.manual_patch_times)
            logger.debug(f"Using manual patch times: {patch_times}")
        elif config.patch_strategy == "uniform":
            # Estimate period for uniform spacing
            period_estimate = self._estimate_period(domain_obj, config)
            patch_times = np.linspace(0, period_estimate, n_patches + 1)
            logger.debug(
                f"Using uniform patches over T={period_estimate:.4f} "
                f"(dt={patch_times[1] - patch_times[0]:.4f})"
            )
        else:
            raise NotImplementedError(
                f"Patch strategy '{config.patch_strategy}' not yet implemented"
            )

        # Propagate trajectory and sample at patch times
        trajectory = self._propagate_for_sampling(
            domain_obj, patch_times, config, options
        )

        # Extract patch initial states (not including final point) and store
        # the full state templates for each patch boundary.
        patch_states: list[np.ndarray] = []
        patch_templates: list[np.ndarray] = []
        for i in range(n_patches):
            state_i = trajectory[i]
            patch_states.append(state_i[control_indices])
            patch_templates.append(state_i.copy())

        # Flatten to parameter vector and keep templates aligned with patch indices
        patch_params = np.concatenate(patch_states) if patch_states else np.array([])

        logger.debug(
            f"Initialized {n_patches} patches: param vector size = {patch_params.size}"
        )

        return patch_params, patch_times, patch_templates

    def _estimate_period(
        self, domain_obj: "PeriodicOrbit", config: MultipleShootingOrbitCorrectionConfig
    ) -> float:
        """Estimate orbit period for patch initialization.

        Strategy:
        1. Use provided period if available
        2. Otherwise, propagate until event detection
        3. Fall back to default time span

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Configuration.

        Returns
        -------
        float
            Estimated half-period for patch distribution.
        """
        # Try to use existing period
        if hasattr(domain_obj.dynamics, "period") and domain_obj.dynamics.period is not None:
            half_period = domain_obj.dynamics.period / 2.0
            logger.debug(f"Using existing half-period: {half_period:.4f}")
            return half_period

        # Try event detection
        try:
            if config.event_func is not None:
                t_event, _ = config.event_func(
                    dynsys=domain_obj.dynamics.dynsys,
                    x0=domain_obj.initial_state,
                    forward=config.forward,
                )
                logger.debug(f"Estimated half-period from event: {t_event:.4f}")
                return float(t_event)
        except Exception as e:
            logger.debug(f"Event-based period estimation failed: {e}")

        # Default fallback
        default_period = 2.0
        logger.warning(
            f"Could not estimate period; using default {default_period} "
            "(may need adjustment)"
        )
        return default_period

    def _propagate_for_sampling(
        self,
        domain_obj: "PeriodicOrbit",
        patch_times: np.ndarray,
        config: MultipleShootingOrbitCorrectionConfig,
        options,
    ) -> list[np.ndarray]:
        """Propagate trajectory and sample at patch times.

        Uses dense output for accurate state interpolation at patch times.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        patch_times : np.ndarray
            Time values to sample at.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Compile-time configuration.
        options : :class:`~hiten.algorithms.corrector.options.MultipleShootingCorrectionOptions`
            Runtime options (integration params).

        Returns
        -------
        list[np.ndarray]
            State vectors at each patch time.
        """

        dynsys = domain_obj.dynamics.dynsys
        x0 = domain_obj.initial_state

        # Propagate with dense output for interpolation
        sol = _propagate_dynsys(
            dynsys=dynsys,
            state0=x0,
            t0=patch_times[0],
            tf=patch_times[-1],
            forward=1,
            steps=options.base.integration.steps * len(patch_times),  # More steps for accuracy
            method=config.integration.method,
            order=options.base.integration.order,
        )

        # Interpolate solution at requested patch times to avoid sampling errors
        patch_times_arr = np.asarray(patch_times, dtype=float)
        states_interp = sol.interpolate(patch_times_arr)
        trajectory = [states_interp[i].copy() for i in range(len(patch_times_arr))]

        return trajectory

    def _half_period(
        self,
        domain_obj: "PeriodicOrbit",
        corrected_state: np.ndarray,
        problem: _MultipleShootingProblem,
    ) -> float:
        """Compute half period of corrected orbit.

        For multiple shooting, we need to propagate from the corrected initial
        state to the actual boundary event, since the patch times are based on
        the initial guess trajectory and may not reflect the corrected orbit.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        corrected_state : np.ndarray
            Corrected initial state.
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem containing event function.

        Returns
        -------
        float
            Half-period (total time from t=0 to final boundary).

        Notes
        -----
        Unlike the initial implementation that just returned patch_times[-1],
        we now propagate from the corrected state to find the actual event time.
        This ensures the period matches the corrected trajectory, not the initial guess.
        """
        forward = problem.forward
        try:
            t_event, _ = problem.event_func(
                dynsys=domain_obj.dynamics.dynsys,
                x0=corrected_state,
                forward=forward,
            )
            return float(t_event)
        except Exception as exc:
            logger.warning(
                f"Failed to compute half-period via event detection: {exc}. "
                f"Falling back to patch_times estimate."
            )
            return float(problem.patch_times[-1])

    # _norm_fn inherited from _OrbitCorrectionInterfaceBase

    def _extract_patch_states(
        self, x_corr: np.ndarray, problem: _MultipleShootingProblem
    ) -> list[np.ndarray]:
        """Extract full patch states from corrected parameter vector.

        Parameters
        ----------
        x_corr : np.ndarray
            Corrected parameter vector (all patches stacked).
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem.

        Returns
        -------
        list[np.ndarray]
            Full state vectors at each patch point.
        """
        control_indices = list(problem.control_indices)
        n_patches = problem.n_patches
        n_control = len(control_indices)
        templates = problem.patch_templates

        patches = x_corr.reshape(n_patches, n_control)

        patch_states = []
        for i in range(n_patches):
            # Use template for uncontrolled components
            x_full = self._reconstruct_full_state(
                templates[i], control_indices, patches[i]
            )
            patch_states.append(x_full)

        return patch_states

    def _compute_continuity_errors(
        self, patch_states: list[np.ndarray], problem: _MultipleShootingProblem
    ) -> np.ndarray:
        """Compute continuity errors at patch junctions.

        Parameters
        ----------
        patch_states : list[np.ndarray]
            Full states at each patch.
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem.

        Returns
        -------
        np.ndarray
            Continuity errors at each junction.
        """
        continuity_indices = list(problem.continuity_indices)
        n_patches = problem.n_patches
        errors = []

        for i in range(n_patches - 1):
            x_i = patch_states[i]
            dt = problem.patch_times[i + 1] - problem.patch_times[i]

            # Propagate from patch i to patch i+1
            sol = _propagate_dynsys(
                dynsys=problem.domain_obj.dynamics.dynsys,
                state0=x_i,
                t0=0,
                tf=dt,
                method=problem.method,
                order=problem.order,
                steps=problem.steps,
                forward=1,
            )
            x_next_minus = sol.states[-1, :]

            # Compare with patch i+1
            x_next = patch_states[i + 1]
            error = x_next_minus[continuity_indices] - x_next[continuity_indices]
            errors.append(error)

        return np.concatenate(errors) if errors else np.array([])
