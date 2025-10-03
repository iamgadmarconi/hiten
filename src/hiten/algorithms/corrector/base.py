"""Generic facade for correction algorithms.

This module provides a single, generic facade for correction algorithms that
works with any domain through the interface pattern. The facade orchestrates
the complete pipeline: facade -> engine -> interface -> backend.
"""

from typing import TYPE_CHECKING, Any, Generic, Optional

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import \
    _OrbitCorrectionInterface
from hiten.algorithms.corrector.stepping import make_armijo_stepper
from hiten.algorithms.corrector.types import StepperFactory
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)

if TYPE_CHECKING:
    from hiten.algorithms.corrector.options import OrbitCorrectionOptions


class CorrectorPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
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
    config : :class:`~hiten.algorithms.types.core.ConfigT`
        Configuration object for the correction algorithm.
    interface : :class:`~hiten.algorithms.types.core.InterfaceT`
        Interface instance for the correction algorithm.
    engine : :class:`~hiten.algorithms.corrector.engine.base._CorrectionEngine`, optional
        Engine instance to use for correction. If None, must be set later
        or use with_default_engine() factory method.
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> from hiten.algorithms.corrector import CorrectorPipeline
    >>> from hiten.algorithms.corrector.config import OrbitCorrectionConfig
    >>> from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
    >>> from hiten.algorithms.corrector.interfaces import _OrbitCorrectionInterface
    >>> from hiten.algorithms.corrector.backends.newton import _NewtonBackend
    >>> 
    >>> # Create components
    >>> config = OrbitCorrectionConfig()
    >>> backend = _NewtonBackend()
    >>> interface = _OrbitCorrectionInterface()
    >>> engine = _OrbitCorrectionEngine(backend=backend, interface=interface)
    >>> 
    >>> # Create facade
    >>> corrector = CorrectorPipeline(config, interface, engine)
    >>> result = corrector.correct(orbit)
    """

    def __init__(self, config: ConfigT, engine: _CorrectionEngine, interface: InterfaceT = None, backend: _CorrectorBackend = None) -> None:
        super().__init__(config, engine, interface, backend)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None, backend: Optional[_CorrectorBackend] = None) -> "CorrectorPipeline[DomainT, InterfaceT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).

        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            Configuration object for the correction algorithm.
        interface : :class:`~hiten.algorithms.types.core.InterfaceT`, optional
            Interface instance for the correction algorithm. If None, uses the default _OrbitCorrectionInterface.
        backend : :class:`~hiten.algorithms.corrector.backends.base._CorrectorBackend`, optional
            Backend instance for the correction algorithm. If None, uses the default _NewtonBackend.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.base.CorrectorPipeline`
            A correction facade instance with a default engine injected.
        """
        backend = backend or _NewtonBackend(stepper_factory=make_armijo_stepper())
        intf = interface or _OrbitCorrectionInterface()
        engine = _CorrectionEngine(backend=backend, interface=intf)
        return cls(config, engine, intf, backend)

    def correct(
        self, 
        domain_obj: DomainT,
        options: "OrbitCorrectionOptions",
        *,
        stepper: Optional[StepperFactory] = None,
        ) -> ResultT:
        """Correct the domain object using the configured engine.
        
        This method corrects any domain object using the configured
        correction algorithm. It delegates to the interface to handle
        domain-specific logic.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.core.DomainT`
            The domain object to correct (e.g., PeriodicOrbit, Manifold).
        options : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            Runtime options for the correction pipeline.
        stepper : StepperFactory, optional
            Custom stepper factory for line search.
            
        Returns
        -------
        :class:`~hiten.algorithms.types.core.ResultT`
            Domain-specific correction result containing:
            - Corrected parameters
            - Convergence information
            - Algorithm diagnostics
        """
        problem = self._create_problem(
            domain_obj=domain_obj,
            options=options,
            stepper_factory=stepper,
        )
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

    def get_convergence_summary(self) -> dict[str, Any]:
        """Get a summary of convergence statistics for batch results.
        
        Parameters
        ----------
        results : list[:class:`~hiten.algorithms.types.core.ResultT`]
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

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
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
