"""User-facing facades for continuation workflows.

These facades assemble the engine, backend, and interface using DI and
provide a simple API to run continuation with domain-friendly inputs.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hiten.algorithms.continuation.backends.pc import _PCContinuationBackend
from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.algorithms.continuation.engine.engine import \
    _OrbitContinuationEngine
from hiten.algorithms.continuation.interfaces import \
    _PeriodicOrbitContinuationInterface
from hiten.algorithms.continuation.types import ContinuationResult
from hiten.algorithms.utils.types import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


@dataclass
class StateParameter:
    """Facade for natural-parameter continuation varying selected state components.

    Users supply an engine (DI). Use `StateParameter.with_default_engine()` to
    construct a default engine wired with the generic predict-correct backend
    and the periodic-orbit interface.
    """

    engine: _OrbitContinuationEngine

    @classmethod
    def with_default_engine(cls) -> "StateParameter":
        """Create a facade instance with a default engine (factory).

        The default engine uses `_PCContinuationBackend` and
        `_PeriodicOrbitContinuationInterface`.
        """
        backend = _PCContinuationBackend()
        interface = _PeriodicOrbitContinuationInterface()
        engine = _OrbitContinuationEngine(backend=backend, interface=interface)
        return cls(engine=engine)

    def solve(
        self,
        seed: PeriodicOrbit,
        *,
        state: SynodicState | Sequence[SynodicState] | int | Sequence[int] | None,
        target: Sequence[float] | np.ndarray,
        step: float | Sequence[float] | np.ndarray,
        max_members: int = 256,
        max_retries_per_step: int = 10,
        step_min: float = 1e-10,
        step_max: float = 1.0,
        extra_params: dict | None = None,
        shrink_policy=None,
        stepper: str = "natural",
    ) -> ContinuationResult:
        # Normalize inputs for config construction
        target_arr = np.asarray(target, dtype=float)
        step_arr = np.asarray(step, dtype=float)

        cfg = _OrbitContinuationConfig(
            target=target_arr,
            step=step_arr,
            max_members=int(max_members),
            max_retries_per_step=int(max_retries_per_step),
            step_min=float(step_min),
            step_max=float(step_max),
            state=state if state is None else state,
            amplitude=False,
            getter=None,
            extra_params=extra_params or {},
            shrink_policy=shrink_policy,
            stepper=str(stepper),
        )

        engine = self.engine
        return engine.solve(seed, cfg)


