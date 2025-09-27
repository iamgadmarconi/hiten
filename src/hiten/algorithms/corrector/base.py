"""Generic facade for correction algorithms.

This module provides a single, generic facade for correction algorithms that
works with any domain through the interface pattern. The facade orchestrates
the complete pipeline: facade → engine → interface → backend.
"""

from typing import Any, Callable, Generic, Literal, Optional

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.config import _LineSearchConfig
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import \
    _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.stepping import (make_armijo_stepper,
                                                 make_plain_stepper)
from hiten.algorithms.corrector.stepping.norm import (_default_norm,
                                                      _infinity_norm)
from hiten.algorithms.corrector.types import CorrectionResult, StepperFactory
from hiten.algorithms.poincare.singlehit.backend import _y_plane_crossing
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBaseFacade)


class Corrector(_HitenBaseFacade, Generic[DomainT, ConfigT, ResultT]):
    """Generic facade for correction algorithms.
    
    This facade provides a clean, high-level interface for correcting
    any domain object (orbits, manifolds, etc.) using the configured
    correction algorithm. It orchestrates the complete correction
    pipeline and handles configuration, error management, and result processing.
    
    The facade is domain-agnostic and works with any domain through
    the interface pattern. Domain-specific logic is handled by the
    interface, not the facade.
    
    Parameters
    ----------
    config : ConfigT
        Configuration object for the correction algorithm.
    engine : Any, optional
        Engine instance to use for correction. If None, must be set later
        or use with_default_engine() factory method.
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> from hiten.algorithms.corrector import CorrectorFacade
    >>> from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
    >>> from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
    >>> from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
    >>> from hiten.algorithms.corrector.backends.newton import _NewtonBackend
    >>> 
    >>> # Create components
    >>> config = _OrbitCorrectionConfig()
    >>> backend = _NewtonBackend()
    >>> interface = _PeriodicOrbitCorrectorInterface()
    >>> engine = _OrbitCorrectionEngine(backend=backend, interface=interface)
    >>> 
    >>> # Create facade
    >>> corrector = CorrectorFacade(config, engine)
    >>> result = corrector.correct(orbit)
    
    Using the pipeline factory:
    
    >>> from hiten.algorithms.types.core import _PipelineFactory
    >>> 
    >>> pipeline = (_PipelineFactory()
    ...     .with_backend(_NewtonBackend())
    ...     .with_interface(_PeriodicOrbitCorrectorInterface())
    ...     .with_engine(_OrbitCorrectionEngine)
    ...     .with_config(_OrbitCorrectionConfig())
    ...     .build_facade(CorrectorFacade))
    >>> result = pipeline.correct(orbit)
    
    Notes
    -----
    The facade handles the complete correction workflow:
    1. Validates the domain object and configuration
    2. Creates a correction problem from the domain object
    3. Delegates to the engine for computation
    4. Processes and returns the results
    
    Domain-specific logic is handled by the interface, which:
    - Translates domain objects to backend-compatible inputs
    - Converts backend outputs to domain results
    - Handles domain-specific validation and processing
    """

    def __init__(self, config: ConfigT, interface, engine: _CorrectionEngine = None) -> None:
        super().__init__(config, interface, engine)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None) -> "Corrector[DomainT, ConfigT, ResultT]":
        backend = _NewtonBackend(stepper_factory=make_armijo_stepper(_LineSearchConfig(norm_fn=_infinity_norm)))
        intf = interface or _PeriodicOrbitCorrectorInterface()
        engine = _CorrectionEngine(backend=backend, interface=intf)
        return cls(config, intf, engine)

    def correct(
        self, 
        domain_obj: DomainT,
        override: bool = False,
        *,
        max_attempts: Optional[int] = None,
        tol: Optional[float] = None,
        max_delta: Optional[float] = None,
        line_search_config: Optional[_LineSearchConfig] = None,
        finite_difference: Optional[bool] = None,
        fd_step: Optional[float] = None,  
        method: Optional[Literal["fixed", "adaptive", "symplectic"]] = None,
        order: Optional[int] = None,
        steps: Optional[int] = None,
        forward: Optional[int] = None,
        residual_indices: Optional[tuple[int, ...]] = None,
        control_indices: Optional[tuple[int, ...]] = None,
        extra_jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        target: Optional[tuple[float, ...]] = None,
        event_func: Optional[Callable[..., tuple[float, np.ndarray]]] = None,
        stepper: Optional[StepperFactory] = None,
        ) -> ResultT:
        """Correct the domain object using the configured engine.
        
        This method corrects any domain object using the configured
        correction algorithm. It delegates to the interface to handle
        domain-specific logic.
        
        Parameters
        ----------
        domain_obj : DomainT
            The domain object to correct (e.g., PeriodicOrbit, Manifold).
        **kwargs
            Additional correction parameters that vary by facade implementation.
            Common parameters may include:
            - Correction tolerances and limits
            - Algorithm-specific options
            - Output formatting preferences
            
        Returns
        -------
        ResultT
            Domain-specific correction result containing:
            - Corrected parameters
            - Convergence information
            - Algorithm diagnostics
        """

        kwargs = {
            "max_attempts": max_attempts,
            "tol": tol,
            "max_delta": max_delta,
            "line_search_config": line_search_config,
            "finite_difference": finite_difference,
            "fd_step": fd_step,
            "method": method,
            "order": order,
            "steps": steps,
            "forward": forward,
            "residual_indices": residual_indices,
            "control_indices": control_indices,
            "extra_jacobian": extra_jacobian,
            "target": target,
            "event_func": event_func,
        }

        problem = self._create_problem(domain_obj=domain_obj, override=override, stepper_factory=stepper, **kwargs)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.
        
        Parameters
        ----------
        config : ConfigT
            The configuration object to validate.
            
        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        super()._validate_config(config)
        
        if hasattr(config, 'tol') and config.tol <= 0:
            raise ValueError("Tolerance must be positive")
        if hasattr(config, 'max_attempts') and config.max_attempts <= 0:
            raise ValueError("Max attempts must be positive")
        if hasattr(config, 'max_delta') and config.max_delta is not None and config.max_delta <= 0:
            raise ValueError("Max delta must be positive")

    def get_convergence_summary(self) -> dict[str, Any]:
        """Get a summary of convergence statistics for batch results.
        
        Parameters
        ----------
        results : list[ResultT]
            List of correction results.
            
        Returns
        -------
        dict[str, Any]
            Summary statistics including:
            - total_objects: Total number of objects
            - converged: Number of converged objects
            - failed: Number of failed objects
            - success_rate: Percentage of successful corrections
            - avg_iterations: Average iterations for converged objects
            - avg_residual_norm: Average residual norm for converged objects
        """
        total = len(self._results)
        converged = sum(1 for r in self._results if r.converged)
        failed = total - converged
        
        if converged > 0:
            converged_results = [r for r in self._results if r.converged]
            avg_iterations = sum(r.iterations for r in converged_results) / converged
            avg_residual_norm = sum(r.residual_norm for r in converged_results) / converged
        else:
            avg_iterations = 0.0
            avg_residual_norm = float('inf')
        
        return {
            "total_objects": total,
            "converged": converged,
            "failed": failed,
            "success_rate": (converged / total) * 100.0 if total > 0 else 0.0,
            "avg_iterations": avg_iterations,
            "avg_residual_norm": avg_residual_norm
        }
