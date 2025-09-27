"""Adapters for orbit family persistence and services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path    
from typing import TYPE_CHECKING

from hiten.algorithms.types.services.base import (_DynamicsServiceBase, _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.utils.io.family import load_family, load_family_inplace, save_family

if TYPE_CHECKING:
    from hiten.system.family import OrbitFamily


class _OrbitFamilyPersistenceService(_PersistenceServiceBase):
    """Handle HDF5 serialisation for orbit families."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda family, path, **kw: save_family(family, Path(path), **kw),
            load_fn=lambda path, **kw: load_family(Path(path), **kw),
            load_inplace_fn=lambda family, path, **kw: load_family_inplace(family, Path(path), **kw),
        )


class _OrbitFamilyDynamicsService(_DynamicsServiceBase):

    def __init__(self, family: "OrbitFamily") -> None:
        super().__init__(family)
@dataclass
class _OrbitFamilyServices(_ServiceBundleBase):

    domain_obj: "OrbitFamily"
    persistence: _OrbitFamilyPersistenceService
    dynamics: _OrbitFamilyDynamicsService

    @classmethod
    def default(cls, family: "OrbitFamily") -> "_OrbitFamilyServices":
        return cls(
            domain_obj=family,
            persistence=_OrbitFamilyPersistenceService(),
            dynamics=_OrbitFamilyDynamicsService(family)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _OrbitFamilyDynamicsService) -> "_OrbitFamilyServices":
        return cls(
            domain_obj=dynamics._domain_obj,
            persistence=_OrbitFamilyPersistenceService(),
            dynamics=dynamics
        )
