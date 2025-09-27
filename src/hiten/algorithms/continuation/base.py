"""User-facing facades for continuation workflows.

These facades assemble the engine, backend, and interface using DI and
provide a simple API to run continuation with domain-friendly inputs.
"""

from typing import (TYPE_CHECKING, Callable, Generic, Literal, Optional,
                    Sequence)

import numpy as np

from hiten.algorithms.continuation.types import ContinuationResult
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBaseFacade)
from hiten.algorithms.types.states import SynodicState

if TYPE_CHECKING:
    from hiten.algorithms.continuation.engine.engine import \
        _OrbitContinuationEngine
    from hiten.system.orbits.base import PeriodicOrbit


class StateParameter(_HitenBaseFacade, Generic[DomainT, ConfigT, ResultT]):
    """Facade for natural-parameter continuation varying selected state components.

    Users supply an engine (DI). Use `StateParameter.with_default_engine()` to
    construct a default engine wired with the generic predict-correct backend
    and the periodic-orbit interface.
    """

    def __init__(self, config: ConfigT, interface, engine: "_OrbitContinuationEngine" = None) -> None:
        super().__init__(config, interface, engine)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None) -> "StateParameter[DomainT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).
        """
        from hiten.algorithms.continuation.backends.pc import \
            _PCContinuationBackend
        from hiten.algorithms.continuation.engine.engine import \
            _OrbitContinuationEngine
        from hiten.algorithms.continuation.interfaces import \
            _PeriodicOrbitContinuationInterface

        backend = _PCContinuationBackend()
        intf = interface or _PeriodicOrbitContinuationInterface()
        engine = _OrbitContinuationEngine(backend=backend, interface=intf)
        return cls(config, interface, engine)

    def generate(
        self,
        domain_obj: DomainT,
        override: bool = False,
        *,
        state: Optional[SynodicState | Sequence[SynodicState] | int | Sequence[int]] = None,
        target: Optional[Sequence[float] | np.ndarray] = None,
        step: Optional[float | Sequence[float] | np.ndarray] = None,
        max_members: Optional[int] = None,
        max_retries_per_step: Optional[int] = None,
        step_min: Optional[float] = None,
        step_max: Optional[float] = None,
        extra_params: Optional[dict] = None,
        shrink_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        getter: Optional[Callable[["PeriodicOrbit"], float]] = None,
        stepper: Optional[Literal["natural", "secant"]] = None,
    ) -> ContinuationResult:
        target_arr = np.asarray(target, dtype=float)
        step_arr = np.asarray(step, dtype=float)

        kwargs = {
            "target": target_arr,
            "step": step_arr,
            "max_members": int(max_members),
            "max_retries_per_step": int(max_retries_per_step),
            "step_min": float(step_min),
            "step_max": float(step_max),
            "state": state,
            "getter": getter,
            "extra_params": extra_params or {},
            "shrink_policy": shrink_policy,
            "stepper": str(stepper),
        }

        problem = self._create_problem(domain_obj=domain_obj, override=override, **kwargs)
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
        
        if hasattr(config, 'step_min') and config.step_min <= 0:
            raise ValueError("Tolerance must be positive")
        if hasattr(config, 'step_max') and config.step_max <= 0:
            raise ValueError("Step max must be positive")
        if hasattr(config, 'max_members') and config.max_members <= 0:
            raise ValueError("Max delta must be positive")