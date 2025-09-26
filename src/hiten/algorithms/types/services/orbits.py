"""Adapter helpers orchestrating periodic-orbit numerics and persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from hiten.algorithms.corrector.base import Corrector
from hiten.algorithms.corrector.config import (_LineSearchConfig,
                                               _OrbitCorrectionConfig)
from hiten.algorithms.corrector.stepping import (make_armijo_stepper,
                                                 make_plain_stepper)
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_monodromy, _compute_stm
from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.types.services.base import (_PersistenceServiceBase,
                                                  _DynamicsServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicStateVector,
                                           Trajectory)
from hiten.utils.io.orbits import (load_periodic_orbit,
                                   load_periodic_orbit_inplace,
                                   save_periodic_orbit)

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _OrbitPersistenceService(_PersistenceServiceBase):
    """Thin wrapper around orbit persistence helpers."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda orbit, path, **kw: save_periodic_orbit(orbit, Path(path), **kw),
            load_fn=lambda path, **kw: load_periodic_orbit(Path(path), **kw),
            load_inplace_fn=lambda orbit, path, **kw: load_periodic_orbit_inplace(orbit, path, **kw),
        )


class _OrbitCorrectionService(_DynamicsServiceBase):
    """Drive Newton-based differential correction for periodic orbits."""

    def __init__(self, domain_obj: Any = None) -> None:
        super().__init__(domain_obj)

    def correct(self, *, overrides: Dict[str, Any] | None = None) -> Tuple[np.ndarray, float]:
        overrides = overrides or {}

        cfg_base: "_OrbitCorrectionConfig" = self._domain_obj._correction_config
        cfg = replace(cfg_base, **{k: v for k, v in overrides.items() if hasattr(cfg_base, k)})

        line_search = overrides.get("line_search_config", cfg.line_search_config)
        if line_search is True:
            stepper_factory = make_armijo_stepper(_LineSearchConfig())
        elif line_search is False or line_search is None:
            stepper_factory = make_plain_stepper()
        else:
            stepper_factory = make_armijo_stepper(line_search)

        corrector = Corrector.with_default_engine(config=cfg)

        results = corrector.correct(self._domain_obj)

        return results.corrected_state, 2 * results.half_period


class _OrbitDynamicsService(_DynamicsServiceBase):
    """Integrate periodic orbits using the system dynamics."""

    def __init__(self, orbit: "PeriodicOrbit", *, initial_state: np.ndarray | None = None) -> None:
        super().__init__(orbit)
        self._system = self._domain_obj.system
        self._initial_state = initial_state

    def propagate(self, *, steps: int, method: str, order: int) -> Tuple[np.ndarray, float]:

        if self._initial_state is None:
            raise ValueError("Initial state must be provided")

        key = self.make_key("propagate", steps, method, order)
        
        def _factory() -> Tuple[np.ndarray, float]:
            sol = _propagate_dynsys(
                dynsys=self._system.dynsys,
                state0=self._domain_obj.initial_state,
                t0=0.0,
                tf=self._domain_obj.period,
                forward=1,
                steps=steps,
                method=method,
                order=order,
            )

            traj = Trajectory.from_solution(
                sol,
                state_vector_cls=SynodicStateVector,
                frame=ReferenceFrame.ROTATING,
            )
            return sol, traj.period
        
        return self.get_or_create(key, _factory)

    def monodromy(self):
        key = self.make_key("monodromy")

        def _factory() -> np.ndarray:
            return _compute_monodromy(self._system.var_dynsys, self._domain_obj.initial_state, self._domain_obj.period)

        return self.get_or_create(key, _factory)

    def compute_stability(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = self.make_key("stability")

        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Phi = _compute_stm(self._system.var_dynsys, self._domain_obj.initial_state, self._domain_obj.period)[2]
            backend = _LinalgBackend()
            indices, eigvals, eigvecs = backend.stability_indices(Phi)
            return indices, eigvals, eigvecs

        return self.get_or_create(key, _factory)
    
    def reset(self) -> None:
        """Reset all cached data."""
        super().reset()


@dataclass
class _OrbitServices(_ServiceBundleBase):
    domain_obj: "PeriodicOrbit"
    correction: _OrbitCorrectionService
    dynamics: _OrbitDynamicsService
    persistence: _OrbitPersistenceService

    @classmethod
    def default(cls, orbit: "PeriodicOrbit") -> "_OrbitServices":
        return cls(
            domain_obj=orbit,
            correction=_OrbitCorrectionService(orbit),
            dynamics=_OrbitDynamicsService(orbit),
            persistence=_OrbitPersistenceService()
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _OrbitDynamicsService) -> "_OrbitServices":
        return cls(
            domain_obj=dynamics._domain_obj,
            correction=_OrbitCorrectionService(dynamics._domain_obj),
            dynamics=dynamics,
            persistence=_OrbitPersistenceService()
        )
