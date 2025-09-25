"""Adapter helpers orchestrating periodic-orbit numerics and persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.config import (_LineSearchConfig,
                                               _OrbitCorrectionConfig)
from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
from hiten.algorithms.corrector.interfaces import \
    _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.stepping import (make_armijo_stepper,
                                                 make_plain_stepper)
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import (_compute_monodromy, _compute_stm)
from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.integrators.base import _Solution
from hiten.algorithms.types.adapters.base import (_CachedDynamicsAdapter,
                                                  _PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicStateVector,
                                           Trajectory)
from hiten.utils.io.orbits import (load_periodic_orbit,
                                   load_periodic_orbit_inplace,
                                   save_periodic_orbit)

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.orbits.base import PeriodicOrbit

@dataclass
class _CorrectionResult:
    corrected_state: Any
    period: float


@dataclass
class _PropagationResult:
    solution: "_Solution"
    trajectory: Trajectory


@dataclass
class _StabilityResult:
    indices: Any
    eigenvalues: Any
    eigenvectors: Any


class _OrbitPersistenceAdapter(_PersistenceAdapterMixin):
    """Thin wrapper around orbit persistence helpers."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda orbit, path, **kw: save_periodic_orbit(orbit, Path(path), **kw),
            load_fn=lambda path, **kw: load_periodic_orbit(Path(path), **kw),
            load_inplace_fn=lambda orbit, path, **kw: load_periodic_orbit_inplace(orbit, path, **kw),
        )


class _OrbitCorrectionAdapter:
    """Drive Newton-based differential correction for periodic orbits."""

    def correct(
        self,
        orbit: "PeriodicOrbit",
        *,
        overrides: Dict[str, Any] | None = None,
    ) -> _CorrectionResult:
        overrides = overrides or {}

        cfg_base: "_OrbitCorrectionConfig" = orbit._correction_config
        cfg = replace(cfg_base, **{k: v for k, v in overrides.items() if hasattr(cfg_base, k)})

        line_search = overrides.get("line_search_config", cfg.line_search_config)
        if line_search is True:
            stepper_factory = make_armijo_stepper(_LineSearchConfig())
        elif line_search is False or line_search is None:
            stepper_factory = make_plain_stepper()
        else:
            stepper_factory = make_armijo_stepper(line_search)

        backend = _NewtonBackend(stepper_factory=stepper_factory)
        interface = _PeriodicOrbitCorrectorInterface()
        engine = _OrbitCorrectionEngine(backend=backend, interface=interface)

        problem = interface.create_problem(orbit=orbit, config=cfg, stepper_factory=stepper_factory)
        result = engine.solve(problem)
        half_period = getattr(result, "half_period", None)
        if half_period is not None and not np.isnan(half_period):
            period = 2 * half_period
        else:
            # Fallback to orbit's current period or a default
            period = orbit.period if orbit.period is not None and not np.isnan(orbit.period) else np.pi
        return _CorrectionResult(corrected_state=result.x_corrected, period=period)


class _OrbitDynamicsAdapter(_CachedDynamicsAdapter):
    """Integrate periodic orbits using the system dynamics."""

    def __init__(self, system: "System") -> None:
        self._system = system

    def propagate(
        self,
        orbit: "PeriodicOrbit",
        *,
        steps: int,
        method: str,
        order: int,
    ) -> _PropagationResult:
        sol = _propagate_dynsys(
            dynsys=self._system.dynsys,
            state0=orbit.initial_state,
            t0=0.0,
            tf=orbit.period,
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
        return _PropagationResult(solution=sol, trajectory=traj)

    def monodromy(self, orbit: "PeriodicOrbit"):
        return _compute_monodromy(self._system.var_dynsys, orbit.initial_state, orbit.period)

    def compute_stability(self, orbit: "PeriodicOrbit") -> _StabilityResult:
        Phi = _compute_stm(self._system.var_dynsys, orbit.initial_state, orbit.period)[2]
        backend = _LinalgBackend()
        indices, eigvals, eigvecs = backend.stability_indices(Phi)
        return _StabilityResult(indices=indices, eigenvalues=eigvals, eigenvectors=eigvecs)


@dataclass
class _OrbitServices(_ServiceBundleBase):
    correction: _OrbitCorrectionAdapter
    dynamics: _OrbitDynamicsAdapter
    persistence: _OrbitPersistenceAdapter

    @classmethod
    def for_system(cls, system: "System") -> "_OrbitServices":
        correction = getattr(system, "_orbit_correction_adapter", None)
        if correction is None:
            correction = _OrbitCorrectionAdapter()
            setattr(system, "_orbit_correction_adapter", correction)
        return cls(
            correction=correction,
            dynamics=_OrbitDynamicsAdapter(system),
            persistence=_OrbitPersistenceAdapter(),
        )
