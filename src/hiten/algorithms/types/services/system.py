"""Adapters coordinating dynamics, libration, and persistence services for systems.

This module supplies the concrete adapter implementations used by the
`hiten.system` package to bridge user-facing classes with the algorithms
layer. Each adapter concentrates the knowledge required to instantiate
backends, interfaces, and engines while exposing a slim API back to the
system module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.dynamics.rtbp import (jacobian_dynsys, rtbp_dynsys,
                                            variational_dynsys)
from hiten.algorithms.integrators.base import _Solution
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.utils.coordinates import _get_mass_parameter
from hiten.utils.io.system import load_system, load_system_inplace, save_system

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.body import Body



class _SystemPersistenceService(_PersistenceServiceBase):
    """Thin adapter around system IO helpers for testability and indirection."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda system, path, **kw: save_system(system, Path(path), **kw),
            load_fn=lambda path, **kw: load_system(Path(path), **kw),
            load_inplace_fn=lambda target, path, **kw: load_system_inplace(target, Path(path), **kw),
        )


class _SystemsDynamicsService(_DynamicsServiceBase):
    """Lazily construct and cache dynamical system backends for a CR3BP system."""

    def __init__(self, domain_obj: "System") -> None:
        super().__init__(domain_obj)
        self._primary = domain_obj.primary
        self._secondary = domain_obj.secondary
        self._distance = domain_obj.distance
        self._mu = _get_mass_parameter(domain_obj.primary.mass, domain_obj.secondary.mass)

    @property
    def primary(self) -> "Body":
        return self._primary

    @property
    def secondary(self) -> "Body":
        return self._secondary

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def dynsys(self) -> _DynamicalSystemProtocol:
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "dynsys")
        return self.get_or_create(
            key,
            lambda: rtbp_dynsys(self._mu, name=self._make_dynsys_name("rtbp")),
        )

    @property
    def variational(self) -> _DynamicalSystemProtocol:
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "variational")
        return self.get_or_create(
            key,
            lambda: variational_dynsys(
                self._mu,
                name=self._make_dynsys_name("variational"),
            ),
        )

    @property
    def jacobian(self) -> _DynamicalSystemProtocol:
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "jacobian")
        return self.get_or_create(
            key,
            lambda: jacobian_dynsys(
                self._mu,
                name=self._make_dynsys_name("jacobian"),
            ),
        )
    
    def _make_dynsys_name(self, suffix: str) -> str:
        return f"{self._primary.name}_{self._secondary.name}_{suffix}"

    def reset(self) -> None:
        super().reset()

    def propagate(
        self,
        state0: Sequence[float],
        *,
        tf: float,
        steps: int,
        method: str,
        order: int,
        forward: int,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> _Solution:
        """Delegate propagation to the shared CR3BP integrator."""

        kwargs = extra_kwargs or {}
        return _propagate_dynsys(
            dynsys=self.dynsys,
            state0=state0,
            t0=0.0,
            tf=tf,
            forward=forward,
            steps=steps,
            method=method,
            order=order,
            **kwargs,
        )


@dataclass
class _SystemServices(_ServiceBundleBase):
    domain_obj: "System"
    dynamics: _SystemsDynamicsService
    persistence: _SystemPersistenceService

    @classmethod
    def default(cls, system: "System") -> "_SystemServices":
        dynamics = _SystemsDynamicsService(system)
        persistence = _SystemPersistenceService()
        return cls(domain_obj=system, dynamics=dynamics, persistence=persistence)

    @classmethod
    def with_shared_dynamics(cls, dynamics: _SystemsDynamicsService) -> "_SystemServices":
        return cls(domain_obj=dynamics._domain_obj, dynamics=dynamics, persistence=_SystemPersistenceService())


