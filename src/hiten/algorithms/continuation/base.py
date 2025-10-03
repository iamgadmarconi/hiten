"""User-facing facades for continuation workflows.

These facades assemble the engine, backend, and interface using DI and
provide a simple API to run continuation with domain-friendly inputs.
"""

from typing import TYPE_CHECKING, Generic, Optional

import numpy as np

from hiten.algorithms.continuation.options import OrbitContinuationOptions
from hiten.algorithms.continuation.types import ContinuationResult
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)

if TYPE_CHECKING:
    from hiten.algorithms.continuation.backends.base import \
        _ContinuationBackend
    from hiten.algorithms.continuation.engine.base import _ContinuationEngine


class ContinuationPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """Facade for natural-parameter continuation varying selected state components.

    Users supply an engine (DI). Use `ContinuationPipeline.with_default_engine()` to
    construct a default engine wired with the generic predict-correct backend
    and the periodic-orbit interface.

    Parameters
    ----------
    config : :class:`~hiten.algorithms.types.core.ConfigT`
        Configuration object for the continuation algorithm.
    interface : :class:`~hiten.algorithms.types.core.InterfaceT`
        Interface object for the continuation algorithm.
    engine : :class:`~hiten.algorithms.continuation.engine.base._ContinuationEngine`
        Engine object for the continuation algorithm.
    """

    def __init__(self, config: ConfigT, engine: "_ContinuationEngine", interface: InterfaceT = None, backend: "_ContinuationBackend" = None) -> None:
        super().__init__(config, engine, interface, backend)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None, backend: Optional["_ContinuationBackend"] = None) -> "ContinuationPipeline[DomainT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).

        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            Configuration object for the continuation algorithm.
        interface : :class:`~hiten.algorithms.types.core.InterfaceT`, optional
            Interface object for the continuation algorithm. If None, uses the default _OrbitContinuationInterface.
        backend : :class:`~hiten.algorithms.continuation.backends.base._ContinuationBackend`, optional
            Backend instance for the continuation algorithm. If None, uses the default _PredictorCorrectorContinuationBackend.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.base.ContinuationPipeline`
            A continuation facade instance with a default engine injected.
        """
        from hiten.algorithms.continuation.backends.pc import \
            _PredictorCorrectorContinuationBackend
        from hiten.algorithms.continuation.engine.engine import \
            _OrbitContinuationEngine
        from hiten.algorithms.continuation.interfaces import \
            _OrbitContinuationInterface
        from hiten.algorithms.continuation.stepping import (
            make_natural_stepper, make_secant_stepper)
        from hiten.algorithms.continuation.stepping.support import (
            _NullStepSupport, _VectorSpaceSecantSupport)

        if backend is None:
            if config.stepper == "secant":
                backend = _PredictorCorrectorContinuationBackend(
                    stepper_factory=make_secant_stepper(),
                    support_factory=_VectorSpaceSecantSupport,
                )
            else:
                backend = _PredictorCorrectorContinuationBackend(
                    stepper_factory=make_natural_stepper(),
                    support_factory=_NullStepSupport,
                )

        intf = interface or _OrbitContinuationInterface()
        engine = _OrbitContinuationEngine(backend=backend, interface=intf)
        return cls(config, engine, intf, backend)

    def generate(
        self,
        domain_obj: DomainT,
        options: OrbitContinuationOptions,
    ) -> ContinuationResult:
        """Generate a continuation result.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.core.DomainT`
            The domain object to continue.
        options : :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            Runtime continuation options.

        Returns
        -------
        ContinuationResult
            The continuation result.
        """
        problem = self._create_problem(domain_obj=domain_obj, options=options)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

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
        
        if hasattr(config, 'step_min') and config.step_min <= 0:
            raise ValueError("Tolerance must be positive")
        if hasattr(config, 'step_max') and config.step_max <= 0:
            raise ValueError("Step max must be positive")
        if hasattr(config, 'max_members') and config.max_members <= 0:
            raise ValueError("Max delta must be positive")