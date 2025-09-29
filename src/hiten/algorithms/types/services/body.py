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
    """Encapsulate IO helpers for bodies to simplify testing and substitution.
    
    Parameters
    ----------
    save_fn : Callable[..., Any]
        The function to save the object.
    load_fn : Callable[..., Any]
        The function to load the object.
    load_inplace_fn : Optional[Callable[..., Any]] = None
        The function to load the object in place.
    """

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda body, path, **kw: save_body(body, Path(path), **kw),
            load_fn=lambda path, **kw: load_body(Path(path), **kw),
            load_inplace_fn=lambda body, path, **kw: load_body_inplace(body, Path(path), **kw),
        )

class _BodyDynamicsService(_DynamicsServiceBase):
    """Encapsulate dynamics helpers for bodies to simplify testing and substitution.
    
    Parameters
    ----------
    domain_obj : Body
        The domain object.
    """

    def __init__(self, body: "Body") -> None:
        super().__init__(body)

    @property
    def name(self) -> str:
        """Get the name of the body."""
        return self.domain_obj._name

    @property
    def mass(self) -> float:
        """Get the mass of the body."""
        return self.domain_obj._mass

    @property
    def radius(self) -> float:
        """Get the radius of the body."""
        return self.domain_obj._radius

    @property
    def color(self) -> str:
        """Get the color of the body."""
        return self.domain_obj._color

    @property
    def parent(self) -> "Body":
        """Get the parent of the body."""
        return self.domain_obj._parent


class _BodyServices(_ServiceBundleBase):
    """Encapsulate services for bodies.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.body.Body`
        The domain object.
    persistence : :class:`~hiten.algorithms.types.services.base._BodyPersistenceService`
        The persistence service.
    dynamics : :class:`~hiten.algorithms.types.services.base._BodyDynamicsService`
        The dynamics service.
    """

    def __init__(self, domain_obj: "Body", persistence: _BodyPersistenceService, dynamics: _BodyDynamicsService) -> None:
        super().__init__(domain_obj)
        self.persistence = persistence
        self.dynamics = dynamics

    @classmethod
    def default(cls, domain_obj: "Body") -> "_BodyServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.body.Body`
            The domain object.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.base._BodyServices`
            The service bundle.
        """
        return cls(domain_obj=domain_obj, persistence=_BodyPersistenceService(), dynamics=_BodyDynamicsService(domain_obj))

    @classmethod
    def with_shared_dynamics(cls, dynamics: _BodyDynamicsService) -> "_BodyServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.base._BodyDynamicsService`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.base._BodyServices`
            The service bundle.
        """
        return cls(domain_obj=dynamics.domain_obj, persistence=_BodyPersistenceService(), dynamics=dynamics)