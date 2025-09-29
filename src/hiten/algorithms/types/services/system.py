"""Adapters coordinating dynamics, libration, and persistence services for systems.

This module supplies the concrete adapter implementations used by the
:mod:`~hiten.system` package to bridge user-facing classes with the algorithms
layer. Each adapter concentrates the knowledge required to instantiate
backends, interfaces, and engines while exposing a slim API back to the
system module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import (jacobian_dynsys, rtbp_dynsys,
                                            variational_dynsys)
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicStateVector,
                                           Trajectory)
from hiten.algorithms.utils.coordinates import _get_mass_parameter
from hiten.utils.io.system import load_system, load_system_inplace, save_system

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.body import Body
    from hiten.system.libration.base import LibrationPoint



class _SystemPersistenceService(_PersistenceServiceBase):
    """Thin adapter around system IO helpers for testability and indirection.
    
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
            save_fn=lambda system, path, **kw: save_system(system, Path(path), **kw),
            load_fn=lambda path, **kw: load_system(Path(path), **kw),
            load_inplace_fn=lambda target, path, **kw: load_system_inplace(target, Path(path), **kw),
        )


class _SystemsDynamicsService(_DynamicsServiceBase):
    """Lazily construct and cache dynamical system backends for a CR3BP system.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.base.System`
        The domain object.

    Attributes
    ----------
    primary : :class:`~hiten.system.body.Body`
        The primary body.
    secondary : :class:`~hiten.system.body.Body`
        The secondary body.
    distance : float
        The distance between the primary and secondary bodies.
    """

    def __init__(self, domain_obj: "System") -> None:
        super().__init__(domain_obj)
        self._primary = self.domain_obj._primary
        self._secondary = self.domain_obj._secondary
        self._distance = self.domain_obj._distance

    @property
    def mu(self) -> float:
        """Mass parameter mu."""
        cache_key = self.make_key(id(self._primary), id(self._secondary), "mu")

        def _factory() -> float:
            return _get_mass_parameter(self._primary._mass, self._secondary._mass)

        return self.get_or_create(cache_key, _factory)

    @property
    def libration_points(self) -> Dict[int, "LibrationPoint"]:
        """Libration points."""
        return self.domain_obj._libration_points

    @property
    def primary(self) -> "Body":
        """Primary body."""
        return self._primary

    @property
    def secondary(self) -> "Body":
        """Secondary body."""
        return self._secondary

    @property
    def distance(self) -> float:
        """Distance between the primary and secondary bodies."""
        return self._distance

    @property
    def dynsys(self) -> _DynamicalSystem:
        """Dynamical system."""
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "dynsys")

        def _factory() -> _DynamicalSystem:
            return rtbp_dynsys(self.mu, name=self._make_dynsys_name("rtbp"))

        return self.get_or_create(key, _factory)

    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """Variational dynamical system."""
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "variational")

        def _factory() -> _DynamicalSystem:
            return variational_dynsys(self.mu, name=self._make_dynsys_name("variational"))

        return self.get_or_create(key, _factory)

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """Jacobi dynamical system."""
        key = self.make_key(id(self._primary), id(self._secondary), self._distance, "jacobian")

        def _factory() -> _DynamicalSystem:
            return jacobian_dynsys(self.mu, name=self._make_dynsys_name("jacobian"))

        return self.get_or_create(key, _factory)
    
    def get_point(self, index: int) -> "LibrationPoint":
        """Get a libration point."""
        if index not in self.domain_obj._libration_points:
            self.domain_obj._libration_points[index] = self._build_libration_point(index)
        return self.domain_obj._libration_points[index]

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
    ) -> Trajectory:
        """Delegate propagation to the shared CR3BP integrator.
        
        Parameters
        ----------
        state0 : Sequence[float]
            The initial state.
        tf : float
            The final time.
        steps : int
            The number of steps.
        method : str
            The method to use for propagation.
        order : int
            The order of the method to use for propagation.
        forward : int
            The forward direction.
        extra_kwargs : Optional[dict[str, Any]] = None
            The extra keyword arguments.

        Returns
        -------
        :class:`~hiten.algorithms.types.states.Trajectory`
            The propagated trajectory.
        """
        cache_key = self.make_key("propagate", state0, tf, steps, method, order, forward, extra_kwargs)

        def _factory() -> Trajectory:
            kwargs = extra_kwargs or {}
            sol = _propagate_dynsys(
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
            traj = Trajectory.from_solution(
                solution=sol,
                state_vector_cls=SynodicStateVector,
                frame=ReferenceFrame.ROTATING,
            )
            return traj

        return self.get_or_create(cache_key, _factory)

    def _build_libration_point(self, index: int) -> "LibrationPoint":
        """Instantiate and wire a libration point with shared services.
        
        Parameters
        ----------
        index : int
            The index of the libration point.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point.
        """
        from hiten.system.libration.collinear import L1Point, L2Point, L3Point
        from hiten.system.libration.triangular import L4Point, L5Point

        mapping: Dict[int, type["LibrationPoint"]] = {
            1: L1Point,
            2: L2Point,
            3: L3Point,
            4: L4Point,
            5: L5Point,
        }
        try:
            point_cls = mapping[index]
        except KeyError as exc:
            raise ValueError("Libration point index must be in {1,2,3,4,5}.") from exc

        return point_cls(self.domain_obj)

    def _make_dynsys_name(self, suffix: str) -> str:
        return f"{self._primary.name}_{self._secondary.name}_{suffix}"


class _SystemServices(_ServiceBundleBase):
    """Bundle all system services together.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.base.System`
        The domain object.

    Attributes
    ----------
    dynamics : :class:`~hiten.algorithms.types.services.system._SystemsDynamicsService`
        The dynamics service.
    persistence : :class:`~hiten.algorithms.types.services.system._SystemPersistenceService`
        The persistence service.
    """

    def __init__(self, domain_obj: "System", dynamics: _SystemsDynamicsService, persistence: _SystemPersistenceService) -> None:
        super().__init__(domain_obj)
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, system: "System") -> "_SystemServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        system : :class:`~hiten.system.base.System`
            The system.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.system._SystemServices`
            The service bundle.
        """
        dynamics = _SystemsDynamicsService(system)
        persistence = _SystemPersistenceService()
        return cls(domain_obj=system, dynamics=dynamics, persistence=persistence)

    @classmethod
    def with_shared_dynamics(cls, dynamics: _SystemsDynamicsService) -> "_SystemServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.system._SystemsDynamicsService`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.system._SystemServices`
            The service bundle.
        """
        return cls(domain_obj=dynamics.domain_obj, dynamics=dynamics, persistence=_SystemPersistenceService())


