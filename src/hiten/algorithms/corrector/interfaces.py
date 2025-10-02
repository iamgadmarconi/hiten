"""Provide domain-specific interfaces for correction algorithms.

This module provides interface classes that adapt generic correction algorithms
 to specific problem domains. These interfaces handle the translation between
 domain objects (orbits, manifolds) and the abstract vector representations
 expected by the correction algorithms.
"""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from hiten.algorithms.corrector.config import (
    MultipleShootingOrbitCorrectionConfig, OrbitCorrectionConfig)
from hiten.algorithms.corrector.options import (MultipleShootingCorrectionOptions,
                                                OrbitCorrectionOptions)
from hiten.algorithms.corrector.types import (JacobianFn,
                                              MultipleShootingResult, NormFn,
                                              OrbitCorrectionResult,
                                              StepperFactory,
                                              _MultipleShootingProblem,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _OrbitCorrectionInterface(
    _HitenBaseInterface[
        OrbitCorrectionConfig,
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
        forward = options.forward
        residual_fn = self._residual_fn(domain_obj, config, forward)
        jacobian_fn = self._jacobian_fn(domain_obj, config, forward, options)
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

    def _norm_fn(self) -> NormFn:
        """Get the norm function.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.NormFn`
            The norm function.
        """
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def _residual_fn(self, domain_obj: "PeriodicOrbit", cfg: OrbitCorrectionConfig, forward: int) -> Callable[[np.ndarray], np.ndarray]:
        """Get the residual function.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
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

    def _jacobian_fn(self, domain_obj: "PeriodicOrbit", cfg: OrbitCorrectionConfig, forward: int, options: OrbitCorrectionOptions) -> JacobianFn | None:
        """Get the Jacobian function.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        cfg : :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The configuration.
        forward : int
            The forward direction.
        options : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`, optional
            Runtime options for integration parameters.
        
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
        
        # Extract runtime parameters from options
        order = options.base.integration.order
        steps = options.base.integration.steps
        method = cfg.integration.method

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
                finite_difference=False,
                fd_step=0.0,
                method=method,
                order=order,
                steps=steps,
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
                steps=steps,
                method=method,
                order=order,
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


class _MultipleShootingOrbitCorrectionInterface(
    _HitenBaseInterface[
        MultipleShootingOrbitCorrectionConfig,
        _MultipleShootingProblem,
        MultipleShootingResult,
        tuple[np.ndarray, int, float],
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
        initial_guess, patch_times = self._initialize_patches(domain_obj, config, options)

        # Build residual and Jacobian functions with patch_times closure
        residual_fn = self._residual_fn(domain_obj, config, forward, patch_times, options)
        jacobian_fn = self._jacobian_fn(domain_obj, config, forward, patch_times, options)
        norm_fn = self._norm_fn()

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

    def to_domain(
        self, outputs: tuple[np.ndarray, int, float], *, problem: _MultipleShootingProblem
    ) -> dict[str, Any]:
        """Convert backend outputs to domain payload.

        Parameters
        ----------
        outputs : tuple of numpy.ndarray, int, float
            Backend outputs: (x_corrected, iterations, residual_norm).
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem that was solved.

        Returns
        -------
        dict of str, Any
            Domain-specific payload with corrected orbit information.
        """
        x_corr, iterations, residual_norm = outputs
        
        # Reshape corrected parameters into patch states
        patch_states = self._extract_patch_states(x_corr, problem)
        
        # Get the initial patch (full state)
        x_full = patch_states[0]
        
        # Compute half period
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        
        # Compute continuity errors for diagnostics
        continuity_errors = self._compute_continuity_errors(patch_states, problem)
        
        # Update domain object
        problem.domain_obj.dynamics.reset()
        problem.domain_obj.dynamics._initial_state = x_full
        problem.domain_obj.dynamics.period = 2.0 * half_period
        
        return {
            "iterations": iterations,
            "residual_norm": residual_norm,
            "half_period": half_period,
            "x_full": x_full,
            "patch_states": patch_states,
            "patch_times": problem.patch_times,
            "continuity_errors": continuity_errors,
        }

    def to_results(
        self,
        outputs: tuple[np.ndarray, int, float],
        *,
        problem: _MultipleShootingProblem,
        domain_payload: dict[str, Any] | None = None,
    ) -> MultipleShootingResult:
        """Convert backend outputs to result object.

        Parameters
        ----------
        outputs : tuple of :class:`~numpy.ndarray`, int, float
            Backend outputs: (x_corrected, iterations, residual_norm).
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem that was solved.
        domain_payload : dict of str, Any, optional
            Pre-computed domain payload from to_domain().

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.MultipleShootingResult`
            Complete result with convergence info and patch diagnostics.
        """
        x_corr, iterations, residual_norm = outputs
        
        # Reshape corrected parameters into patch states
        patch_states = self._extract_patch_states(x_corr, problem)
        
        # Get the initial patch (full state)
        x_full = patch_states[0]
        
        # Compute half period
        half_period = self._half_period(problem.domain_obj, x_full, problem)
        
        # Compute continuity errors for diagnostics
        continuity_errors = self._compute_continuity_errors(patch_states, problem)
        
        return MultipleShootingResult(
            converged=True,
            x_corrected=x_full,
            residual_norm=float(residual_norm),
            iterations=int(iterations),
            n_patches=problem.n_patches,
            patch_states=patch_states,
            patch_times=problem.patch_times,
            continuity_errors=continuity_errors,
            half_period=half_period,
        )

    def _initialize_patches(
        self, domain_obj: "PeriodicOrbit", config: MultipleShootingOrbitCorrectionConfig, options
    ) -> tuple[np.ndarray, np.ndarray]:
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
        trajectory = self._propagate_for_sampling(domain_obj, patch_times, config, options)

        # Extract patch initial states (not including final point)
        patch_states = []
        for i in range(n_patches):
            state_i = trajectory[i]  # Full state at patch i
            patch_states.append(state_i[control_indices])

        # Flatten to parameter vector
        patch_params = np.concatenate(patch_states)

        logger.debug(
            f"Initialized {n_patches} patches: param vector size = {patch_params.size}"
        )

        return patch_params, patch_times

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

        # Extract states at patch times
        trajectory = [sol.states[i, :] for i in range(len(patch_times))]

        return trajectory

    def _residual_fn(
        self,
        domain_obj: "PeriodicOrbit",
        config: MultipleShootingOrbitCorrectionConfig,
        forward: int,
        patch_times: np.ndarray,
        options: MultipleShootingCorrectionOptions,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build residual function with continuity constraints.

        Residual structure for n_patches=3:
        [x_1^- - x_0]  ← continuity at patch 0->1
        [x_2^- - x_1]  ← continuity at patch 1->2
        [x_3^- - target]  ← boundary at patch 2->3

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Configuration.
        forward : int
            Integration direction.
        patch_times : np.ndarray
            Patch boundary times.
        options : :class:`~hiten.algorithms.corrector.options.MultipleShootingCorrectionOptions`
            Runtime options.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Residual function R(x).
        """
        base_state = domain_obj.initial_state.copy()
        control_indices = list(config.control_indices)
        continuity_indices = list(
            config.continuity_indices if config.continuity_indices else config.control_indices
        )
        boundary_indices = list(
            config.boundary_only_indices if config.boundary_only_indices else config.residual_indices
        )
        target = np.array(config.target)
        n_patches = options.n_patches
        n_control = len(control_indices)
        method = config.integration.method
        order = options.base.integration.order
        steps = options.base.integration.steps

        def _fn(params: np.ndarray) -> np.ndarray:
            """Evaluate residual for parameter vector.

            Parameters
            ----------
            params : np.ndarray
                Flattened patch states, shape (n_patches * n_control,)

            Returns
            -------
            residual : np.ndarray
                Concatenated [continuity_residuals, boundary_residuals]
            """
            # Reshape into individual patch states
            patches = params.reshape(n_patches, n_control)

            residuals = []

            # Continuity constraints for internal patches
            for i in range(n_patches - 1):
                # Current patch initial state (full state)
                x_i = self._to_full_state(base_state, control_indices, patches[i])

                # Propagate to next patch time
                dt = patch_times[i + 1] - patch_times[i]
                x_next_minus = self._propagate_patch(domain_obj, x_i, dt, method, order, steps)

                # Next patch initial state (full state)
                x_next = self._to_full_state(base_state, control_indices, patches[i + 1])

                # Continuity error (only on continuity indices)
                continuity_error = (
                    x_next_minus[continuity_indices] - x_next[continuity_indices]
                )
                residuals.append(continuity_error)

            # Final patch: propagate and check boundary conditions
            x_final = self._to_full_state(base_state, control_indices, patches[-1])

            # Event-based boundary
            _, x_boundary = config.event_func(
                dynsys=domain_obj.dynamics.dynsys, x0=x_final, forward=forward
            )

            boundary_error = x_boundary[boundary_indices] - target
            residuals.append(boundary_error)

            return np.concatenate(residuals)

        return _fn

    def _jacobian_fn(
        self,
        domain_obj: "PeriodicOrbit",
        config: MultipleShootingOrbitCorrectionConfig,
        forward: int,
        patch_times: np.ndarray,
        options: MultipleShootingCorrectionOptions,
    ) -> JacobianFn | None:
        """Build block-structured Jacobian.

        Structure for n_patches=3, n_control=3:

        [Phi_01  -I   0  ]  <- dR_0/dx_0, dR_0/dx_1
        [ 0   Phi_12  -I ]  <- dR_1/dx_1, dR_1/dx_2
        [dBC dBC  Phi_23]  <- dR_2/dx_2, dR_2/dx_3

        Where Phi_ij = STM from patch i to patch j.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        config : :class:`~hiten.algorithms.corrector.config.MultipleShootingOrbitCorrectionConfig`
            Configuration.
        forward : int
            Integration direction.
        patch_times : np.ndarray
            Patch boundary times.
        options : :class:`~hiten.algorithms.corrector.options.MultipleShootingCorrectionOptions`, optional
            Runtime options for integration parameters.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.JacobianFn` or None
            Jacobian function or None for finite differences.
        """
        if config.numerical.finite_difference:
            return None  # Use FD approximation

        base_state = domain_obj.initial_state.copy()
        control_indices = list(config.control_indices)
        continuity_indices = list(
            config.continuity_indices if config.continuity_indices else config.control_indices
        )
        boundary_indices = list(
            config.boundary_only_indices if config.boundary_only_indices else config.residual_indices
        )
        n_patches = options.n_patches
        n_control = len(control_indices)
        n_continuity = len(continuity_indices)
        n_boundary = len(boundary_indices)
        
        # Extract runtime parameters from options
        order = options.base.integration.order
        steps = options.base.integration.steps
        method = config.integration.method

        # Determine if we should use sparse assembly
        use_sparse_assembly = getattr(config, 'use_sparse_jacobian', False)

        def _fn(params: np.ndarray):
            """Compute block-structured Jacobian matrix (dense or sparse)."""
            patches = params.reshape(n_patches, n_control)

            # Compute STM for each patch
            stm_blocks = []
            Phi_full_final = None  # Store full STM for final patch (for extra_jacobian)
            x_boundary = None      # Store boundary state (for extra_jacobian)
            
            for i in range(n_patches):
                x_i = self._to_full_state(base_state, control_indices, patches[i])

                if i < n_patches - 1:
                    dt = patch_times[i + 1] - patch_times[i]
                else:
                    t_event, x_boundary = config.event_func(
                        dynsys=domain_obj.dynamics.dynsys, x0=x_i, forward=forward
                    )
                    dt = t_event
                # Compute STM for this patch
                _, _, Phi_full, _ = _compute_stm(
                    domain_obj.dynamics.var_dynsys,
                    x_i,
                    dt,
                    steps=steps,
                    method=method,
                    order=order,
                )

                # Store full STM for final patch (needed for extra_jacobian)
                if i == n_patches - 1:
                    Phi_full_final = Phi_full

                # Extract relevant block
                if i < n_patches - 1:
                    # Continuity rows: extract continuity_indices x control_indices
                    Phi_block = Phi_full[np.ix_(continuity_indices, control_indices)]
                else:
                    # Boundary rows: extract boundary_indices x control_indices
                    Phi_block = Phi_full[np.ix_(boundary_indices, control_indices)]

                stm_blocks.append(Phi_block)

            # Compute extra Jacobian if needed (before assembly)
            extra_jac = None
            if config.extra_jacobian is not None and Phi_full_final is not None and x_boundary is not None:
                extra_jac = config.extra_jacobian(x_boundary, Phi_full_final)

            # Choose assembly method based on configuration
            if use_sparse_assembly:
                # Use sparse assembly (direct sparse construction)
                try:
                    jac_sparse = self._assemble_sparse_jacobian(
                        stm_blocks=stm_blocks,
                        n_patches=n_patches,
                        n_control=n_control,
                        n_continuity=n_continuity,
                        n_boundary=n_boundary,
                        continuity_indices=continuity_indices,
                        control_indices=control_indices,
                        extra_jac=extra_jac,
                    )
                    
                    if jac_sparse is not None:
                        # Validate the sparse matrix
                        expected_rows = n_continuity * (n_patches - 1) + n_boundary
                        expected_cols = n_patches * n_control
                        
                        if jac_sparse.shape != (expected_rows, expected_cols):
                            raise ValueError(
                                f"Sparse Jacobian has wrong shape: {jac_sparse.shape}, "
                                f"expected ({expected_rows}, {expected_cols})"
                            )
                        
                        return jac_sparse
                    else:
                        # Sparse assembly returned None (scipy not available)
                        logger.warning(
                            "Sparse assembly unavailable (scipy not installed), "
                            "falling back to dense"
                        )
                except Exception as e:
                    # Sparse assembly raised an exception
                    raise RuntimeError(f"Sparse Jacobian assembly failed: {e}")
                # Fall through to dense assembly

            # Dense assembly (original method)
            n_total_residuals = n_continuity * (n_patches - 1) + n_boundary
            n_total_params = n_patches * n_control
            jac = np.zeros((n_total_residuals, n_total_params))

            row_offset = 0

            # Continuity blocks
            for i in range(n_patches - 1):
                # Phi_i block (positive contribution from patch i)
                col_start_i = i * n_control
                jac[
                    row_offset : row_offset + n_continuity,
                    col_start_i : col_start_i + n_control,
                ] = stm_blocks[i]

                # -I block (negative contribution from patch i+1)
                col_start_ip1 = (i + 1) * n_control
                # Identity block should match dimensions properly
                if n_continuity == n_control:
                    identity_block = -np.eye(n_continuity)
                else:
                    # Partial continuity: need submatrix of identity
                    identity_block = np.zeros((n_continuity, n_control))
                    for j, idx in enumerate(continuity_indices):
                        if idx in control_indices:
                            k = control_indices.index(idx)
                            identity_block[j, k] = -1.0

                jac[
                    row_offset : row_offset + n_continuity,
                    col_start_ip1 : col_start_ip1 + n_control,
                ] = identity_block

                row_offset += n_continuity

            # Boundary block (final patch affects boundary)
            col_start_final = (n_patches - 1) * n_control
            jac[
                row_offset : row_offset + n_boundary,
                col_start_final : col_start_final + n_control,
            ] = stm_blocks[-1]

            # Apply extra Jacobian if computed
            if extra_jac is not None:
                jac[
                    row_offset : row_offset + n_boundary,
                    col_start_final : col_start_final + n_control,
                ] -= extra_jac
                
                logger.debug(
                    f"Applied extra_jacobian to final patch boundary rows "
                    f"(shape: {extra_jac.shape})"
                )

            return jac

        return _fn

    def _assemble_sparse_jacobian(
        self,
        stm_blocks: list[np.ndarray],
        n_patches: int,
        n_control: int,
        n_continuity: int,
        n_boundary: int,
        continuity_indices: list[int],
        control_indices: list[int],
        extra_jac: np.ndarray | None = None,
    ):
        """Assemble block tridiagonal Jacobian directly in sparse format.

        This method builds the multiple shooting Jacobian using scipy.sparse
        constructors, avoiding the allocation of a large dense matrix. This is
        significantly more memory-efficient and faster for large numbers of patches.

        Parameters
        ----------
        stm_blocks : list[np.ndarray]
            List of STM blocks, one per patch. Each is either:
            - For patches 0..n-2: shape (n_continuity, n_control)
            - For patch n-1: shape (n_boundary, n_control)
        n_patches : int
            Number of shooting patches.
        n_control : int
            Number of control variables per patch.
        n_continuity : int
            Number of continuity constraints per junction.
        n_boundary : int
            Number of boundary constraints.
        continuity_indices : list[int]
            Indices of states enforced for continuity.
        control_indices : list[int]
            Indices of states used as control variables.
        extra_jac : np.ndarray or None, optional
            Extra Jacobian term to subtract from final boundary block (e.g.,
            for period correction in periodic orbits).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Jacobian matrix in CSR format with shape:
            (n_continuity * (n_patches - 1) + n_boundary, n_patches * n_control)

        Notes
        -----
        The block tridiagonal structure has the form:

        .. code-block:: text

            [Phi_0   -I     0      0   ]
            [ 0     Phi_1  -I      0   ]
            [ 0      0    Phi_2   -I   ]
            [ 0      0     0    Phi_3  ]

        Where:
        - Phi_i are the STM blocks (dense, small matrices)
        - -I are identity blocks (sparse, diagonal)
        - 0 are true structural zeros (not stored)

        This structure is ~80-90% sparse and can be exploited for O(n)
        solving time instead of O(n³).

        See Also
        --------
        scipy.sparse.bmat : Block matrix constructor used here
        scipy.sparse.eye : Sparse identity matrix constructor
        """
        try:
            from scipy.sparse import bmat, csr_matrix, eye
        except ImportError:
            logger.warning(
                "scipy not available for sparse assembly; "
                "falling back to dense Jacobian"
            )
            return None

        # Build block rows
        block_rows = []

        # Continuity rows (patches 0 to n-2)
        for i in range(n_patches - 1):
            row_blocks = [None] * n_patches

            # Phi_i block (STM from patch i)
            row_blocks[i] = csr_matrix(stm_blocks[i])

            # -I block (identity for patch i+1)
            if n_continuity == n_control:
                row_blocks[i + 1] = -eye(n_continuity, n_control, format="csr")
            else:
                # Partial continuity: selective identity
                identity_block = np.zeros((n_continuity, n_control))
                for j, idx in enumerate(continuity_indices):
                    if idx in control_indices:
                        k = control_indices.index(idx)
                        identity_block[j, k] = -1.0
                row_blocks[i + 1] = csr_matrix(identity_block)

            block_rows.append(row_blocks)

        # Boundary row (final patch)
        boundary_row = [None] * n_patches
        final_block = stm_blocks[-1].copy()  # Shape: (n_boundary, n_control)

        # Apply extra Jacobian if provided
        if extra_jac is not None:
            final_block = final_block - extra_jac

        boundary_row[-1] = csr_matrix(final_block)
        block_rows.append(boundary_row)

        # Assemble using scipy's block matrix constructor
        try:
            J_sparse = bmat(block_rows, format="csr")
        except Exception as e:
            logger.error(
                f"Failed to assemble sparse block matrix: {e}. "
                f"Block structure: {len(block_rows)} rows, "
                f"each with {len(block_rows[0]) if block_rows else 0} columns"
            )
            raise

        # Validate result
        expected_rows = n_continuity * (n_patches - 1) + n_boundary
        expected_cols = n_patches * n_control
        
        if J_sparse.shape != (expected_rows, expected_cols):
            raise ValueError(
                f"Sparse Jacobian dimension mismatch: got {J_sparse.shape}, "
                f"expected ({expected_rows}, {expected_cols})"
            )

        return J_sparse

    def _propagate_patch(
        self,
        domain_obj: "PeriodicOrbit",
        x0: np.ndarray,
        dt: float,
        method: str,
        order: int,
        steps: int,
    ) -> np.ndarray:
        """Propagate a single patch segment.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        x0 : np.ndarray
            Initial state (full state).
        dt : float
            Time span to propagate.
        method : str
            Integration method.
        order : int
            Integration order.
        steps : int
            Number of integration steps.

        Returns
        -------
        np.ndarray
            Final state after propagation.
        """

        dynsys = domain_obj.dynamics.dynsys

        # Fixed-time integration
        sol = _propagate_dynsys(
            dynsys=dynsys,
            state0=x0,
            t0=0,
            tf=dt,
            method=method,
            order=order,
            steps=steps,
            forward=1,
        )

        return sol.states[-1, :]  # Final state

    def _half_period(
        self,
        domain_obj: "PeriodicOrbit",
        corrected_state: np.ndarray,
        problem: _MultipleShootingProblem,
    ) -> float:
        """Compute half period of corrected orbit.

        For multiple shooting, the half-period is already determined by the
        sum of patch times (patch_times[-1]), which represents the total time
        from initial state to the boundary event.

        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        corrected_state : np.ndarray
            Corrected initial state (not used, kept for interface compatibility).
        problem : :class:`~hiten.algorithms.corrector.types._MultipleShootingProblem`
            The problem containing patch times.

        Returns
        -------
        float
            Half-period (total time from t=0 to final boundary).

        Notes
        -----
        Unlike single shooting where we need to propagate to find the period,
        in multiple shooting the period is inherently determined by the sum
        of all patch time intervals, which is stored in patch_times[-1].
        
        The final patch is constrained to reach the boundary event, so the
        total time patch_times[-1] is automatically the correct half-period.
        """
        # The patch times already encode the correct half-period
        # patch_times[-1] = t_0 + dt_0 + dt_1 + ... + dt_{n-1}
        # where dt_{n-1} is the time for the final patch to reach the boundary
        return float(problem.patch_times[-1])

    def _norm_fn(self) -> NormFn:
        """Get the norm function.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.NormFn`
            Infinity norm (emphasizes largest constraint violation).
        """
        return lambda r: float(np.linalg.norm(r, ord=np.inf))

    def _to_full_state(
        self, base_state: np.ndarray, control_indices: list[int], params: np.ndarray
    ) -> np.ndarray:
        """Reconstruct full state from control parameters.

        Parameters
        ----------
        base_state : np.ndarray
            Base state template.
        control_indices : list[int]
            Indices of control variables.
        params : np.ndarray
            Control parameter values.

        Returns
        -------
        np.ndarray
            Full state vector.
        """
        x_full = base_state.copy()
        x_full[control_indices] = params
        return x_full

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
        base_state = problem.domain_obj.initial_state.copy()
        control_indices = list(problem.control_indices)
        n_patches = problem.n_patches
        n_control = len(control_indices)

        patches = x_corr.reshape(n_patches, n_control)

        patch_states = []
        for i in range(n_patches):
            x_full = self._to_full_state(base_state, control_indices, patches[i])
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
            x_next_minus = self._propagate_patch(
                problem.domain_obj, x_i, dt, problem.method, problem.order, problem.steps
            )

            # Compare with patch i+1
            x_next = patch_states[i + 1]
            error = x_next_minus[continuity_indices] - x_next[continuity_indices]
            errors.append(error)

        return np.concatenate(errors) if errors else np.array([])
