"""Adapters supporting persistence for `hiten.system.body` objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hiten.algorithms.types.services.base import (_PersistenceServiceBase,
                                                  _ServiceBundleBase,
                                                  _DynamicsServiceBase)
from hiten.utils.io.body import load_body, load_body_inplace, save_body

if TYPE_CHECKING:
    from hiten.system.body import Body

class _BodyPersistenceService(_PersistenceServiceBase):
    """Encapsulate IO helpers for bodies to simplify testing and substitution."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda body, path, **kw: save_body(body, Path(path), **kw),
            load_fn=lambda path, **kw: load_body(Path(path), **kw),
            load_inplace_fn=lambda body, path, **kw: load_body_inplace(body, Path(path), **kw),
        )

class _BodyDynamicsService(_DynamicsServiceBase):
    """Encapsulate dynamics helpers for bodies to simplify testing and substitution."""

    def __init__(self, body: "Body") -> None:
        super().__init__(body)

    @property
    def name(self) -> str:
        return self.domain_obj.name
    
    @property
    def mass(self) -> float:
        return self.domain_obj.mass
    
    @property
    def radius(self) -> float:
        return self.domain_obj.radius
    
    @property
    def color(self) -> str:
        return self.domain_obj.color
    
    @property
    def parent(self) -> "Body":
        return self.domain_obj.parent


class _BodyServices(_ServiceBundleBase):

    def __init__(self, domain_obj: "Body", persistence: _BodyPersistenceService, dynamics: _BodyDynamicsService) -> None:
        super().__init__(domain_obj)
        self._persistence = persistence
        self._dynamics = dynamics

    @classmethod
    def default(cls, domain_obj: "Body") -> _BodyServices:
        return cls(domain_obj=domain_obj, persistence=_BodyPersistenceService(), dynamics=_BodyDynamicsService(domain_obj))

    @classmethod
    def with_shared_dynamics(cls, dynamics: _BodyDynamicsService) -> _BodyServices:
        return cls(domain_obj=dynamics.domain_obj, persistence=_BodyPersistenceService(), dynamics=dynamics)