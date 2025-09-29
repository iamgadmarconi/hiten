"""Adapters for orbit family persistence and services."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.utils.io.family import load_family, load_family_inplace, save_family

if TYPE_CHECKING:
    from hiten.system.family import OrbitFamily


class _OrbitFamilyPersistenceService(_PersistenceServiceBase):
    """Handle serialization for orbit families.
    
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
            save_fn=lambda family, path, **kw: save_family(family, Path(path), **kw),
            load_fn=lambda path, **kw: load_family(Path(path), **kw),
            load_inplace_fn=lambda family, path, **kw: load_family_inplace(family, Path(path), **kw),
        )


class _OrbitFamilyDynamicsService(_DynamicsServiceBase):
    """Encapsulate services for orbit family.
    
    Parameters
    ----------
    family : :class:`~hiten.system.family.OrbitFamily`
        The domain object.
    """
    def __init__(self, family: "OrbitFamily") -> None:
        super().__init__(family)


class _OrbitFamilyServices(_ServiceBundleBase):
    """Encapsulate services for orbit family.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.family.OrbitFamily`
        The domain object.
    persistence : :class:`~hiten.algorithms.types.services.family._OrbitFamilyPersistenceService`
        The persistence service.
    dynamics : :class:`~hiten.algorithms.types.services.family._OrbitFamilyDynamicsService`
        The dynamics service.
    """
    def __init__(self, domain_obj: "OrbitFamily", persistence: _OrbitFamilyPersistenceService, dynamics: _OrbitFamilyDynamicsService) -> None:
        super().__init__(domain_obj)
        self.persistence = persistence
        self.dynamics = dynamics

    @classmethod
    def default(cls, domain_obj: "OrbitFamily") -> "_OrbitFamilyServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.family.OrbitFamily`
            The domain object.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.family._OrbitFamilyServices`
            The service bundle.
        """
        return cls(
            domain_obj=domain_obj,
            persistence=_OrbitFamilyPersistenceService(),
            dynamics=_OrbitFamilyDynamicsService(domain_obj)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _OrbitFamilyDynamicsService) -> "_OrbitFamilyServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.family._OrbitFamilyDynamicsService`
            The dynamics service.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.services.family._OrbitFamilyServices`
            The service bundle.
        """
        return cls(
            domain_obj=dynamics.domain_obj,
            persistence=_OrbitFamilyPersistenceService(),
            dynamics=dynamics
        )
