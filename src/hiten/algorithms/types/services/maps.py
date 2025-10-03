"""Adapters supporting Poincare map numerics and persistence."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np

from hiten.algorithms.dynamics.hamiltonian import _HamiltonianSystem
from hiten.algorithms.poincare.centermanifold.base import \
    CenterManifoldMapPipeline
from hiten.algorithms.poincare.centermanifold.config import \
    CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldDomainPayload, CenterManifoldMapResults)
from hiten.algorithms.poincare.core.types import _Section
from hiten.algorithms.poincare.synodic.base import SynodicMapPipeline
from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
from hiten.algorithms.poincare.synodic.types import (SynodicMapDomainPayload,
                                                     SynodicMapResults)
from hiten.algorithms.types.configs import IntegrationConfig, RefineConfig
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import Trajectory
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io.map import (load_poincare_map, load_poincare_map_inplace,
                                save_poincare_map)

if TYPE_CHECKING:
    from hiten.algorithms.poincare.centermanifold.options import \
        CenterManifoldMapOptions
    from hiten.algorithms.poincare.core.types import _Section
    from hiten.algorithms.poincare.synodic.options import SynodicMapOptions
    from hiten.system.center import CenterManifold
    from hiten.system.manifold import Manifold
    from hiten.system.maps.center import CenterManifoldMap
    from hiten.system.maps.synodic import SynodicMap
    from hiten.system.orbits.base import PeriodicOrbit


class _MapPersistenceService(_PersistenceServiceBase):
    """Handle persistence for map objects.
    
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
            save_fn=lambda map, path, **kw: save_poincare_map(map, Path(path), **kw),
            load_fn=lambda path, **kw: load_poincare_map(Path(path), **kw),
            load_inplace_fn=lambda map, path, **kw: load_poincare_map_inplace(map, Path(path), **kw),
        )


class _MapDynamicsServiceBase(_DynamicsServiceBase):
    """Base class for map dynamics services with caching.
    
    Parameters
    ----------
    domain_obj : Any
        The domain object.

    Attributes
    ----------
    section_coord : str
        The section coordinate.
    sections : dict[str, :class:`~hiten.algorithms.poincare.core.base._Section`]
        The sections.
    section : :class:`~hiten.algorithms.poincare.core.base._Section`
        The section.
    generator : str
        The key for the section.
    """

    def __init__(self, domain_obj) -> None:
        super().__init__(domain_obj)
        self._sections: dict[str, "_Section"] = {}
        self._section: Optional["_Section"] = None
        self._section_coord = None
        self._generator = None
        self._map_config = None
        self._map: Optional["_Section"] = None

    @property
    def generator(self) -> str:
        """The pipeline to generate the map."""
        if self._generator is None:
            self._generator = self._build_generator()
        return self._generator

    @property
    def map_config(self):
        """Get the map configuration.
        
        Returns the stored config, lazily initializing with defaults if needed.
        """
        if self._map_config is None:
            self._map_config = self._default_map_config()
        return self._map_config

    @abstractmethod
    def _default_map_config(self):
        """Create the default map configuration.
        
        Concrete implementations should override this to provide map-specific defaults.
        """
        pass

    @map_config.setter
    def map_config(self, value):
        """Set the map configuration.
        
        Stores the config and invalidates the generator cache.
        """
        self._map_config = value
        self._generator = None  # Invalidate cache to trigger recreation

    @abstractmethod
    def _build_generator(self):
        """Build the generator."""
        raise NotImplementedError

    @property
    def section_coord(self) -> str:
        """The most recently computed section coordinate."""
        if self._section_coord is None:
            raise ValueError("No section coordinate has been computed yet")
        return self._section_coord

    def get_section(self, section_coord: Optional[str] = None) -> "_Section":
        """Get a computed section by coordinate.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed section data.

        Raises
        ------
        KeyError
            If the section has not been computed.

        Notes
        -----
        This method returns the full section data including points,
        states, labels, and times. Use this method when you need
        access to the complete section information.
        """
        if section_coord is None:
            section_coord = self.section_coord
        if section_coord not in self._sections:
            raise KeyError(
                f"Section '{section_coord}' has not been computed. "
                f"Available: {list(self._sections.keys())}"
            )
        return self._sections[section_coord]

    def list_sections(self) -> list[str]:
        """List all computed section coordinates.

        Returns
        -------
        list[str]
            List of section coordinate identifiers that have been computed.

        Notes
        -----
        This method returns the keys of the internal section cache,
        indicating which sections are available for access.
        """
        return list(self._sections.keys())

    def has_section(self, section_coord: str) -> bool:
        """Check if a section has been computed.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier to check.

        Returns
        -------
        bool
            True if the section has been computed, False otherwise.

        Notes
        -----
        This method provides a safe way to check section availability
        before attempting to access it.
        """
        return section_coord in self._sections

    def clear(self):
        """Clear all cached sections.

        Notes
        -----
        This method clears the internal caches for sections,
        forcing recomputation on the next access. Use this method to
        free memory or force fresh computation with updated parameters.
        """
        self._sections.clear()
        self._section = None
        self._section_coord = None

    def _axis_index(self, section: "_Section", axis: str) -> int:
        """Return the column index corresponding to an axis label.

        Parameters
        ----------
        section : :class:`~hiten.algorithms.poincare.core.base._Section`
            The section containing the axis labels.
        axis : str
            The axis label to find.

        Returns
        -------
        int
            The column index of the axis in the section points.

        Raises
        ------
        ValueError
            If the axis label is not found in the section labels.

        Notes
        -----
        The default implementation assumes a 1-1 mapping between the
        section.labels tuple and columns of section.points. Concrete
        subclasses can override this method if their mapping differs
        or if axis-based projection is not supported.
        """
        try:
            return section.labels.index(axis)
        except ValueError as exc:
            raise ValueError(
                f"Axis '{axis}' not available; valid labels are {section.labels}"
            ) from exc

    def get_points(
        self,
        *,
        section_coord: str | None = None,
        axes: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Return cached points for a section with optional axis projection.

        Parameters
        ----------
        section_coord : str, optional
            Which stored section to retrieve. If None, uses the default
            section coordinate from the configuration.
        axes : tuple[str, str], optional
            Optional tuple of two axis labels (e.g., ("q3", "p2")) requesting
            a different 2D projection of the stored state. If None, returns
            the raw stored projection.

        Returns
        -------
        ndarray, shape (n, 2)
            Array of 2D points in the section plane, either the raw points
            or a projection onto the specified axes.

        Notes
        -----
        This method provides access to the computed section points with
        optional axis projection. If the section hasn't been computed,
        it triggers computation automatically. The axis projection allows
        viewing the section data from different coordinate perspectives.
        """
        key = section_coord or self.section_coord

        sec = self._get_or_compute_section(key)

        if axes is None:
            return sec.points

        idx1 = self._axis_index(sec, axes[0])
        idx2 = self._axis_index(sec, axes[1])

        return sec.points[:, (idx1, idx2)]

    def _get_or_compute_section(self, key: str) -> "_Section":
        """Return the cached section for center manifold maps, computing if necessary."""
        if key not in self._sections:
            self.compute(section_coord=key)
        return self._sections[key]

    @abstractmethod
    def compute(self, *, section_coord: str = "q3", overrides: dict[str, Any] | None = None, **kwargs) -> np.ndarray:
        """Compute or retrieve the return map for the specified section."""
        raise NotImplementedError


class _CenterManifoldMapDynamicsService(_MapDynamicsServiceBase):
    """Dynamics service for center manifold maps with caching.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.maps.center.CenterManifoldMap`
        The domain object.

    Attributes
    ----------
    center_manifold : :class:`~hiten.system.center.CenterManifold`
        The center manifold.
    energy : float
        The energy.
    generator : :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMapPipeline`
        The generator.
    section_coord : str
        The section coordinate.
    """

    def __init__(self, domain_obj: "CenterManifoldMap") -> None:
        self._energy = domain_obj._energy
        self._center_manifold = domain_obj._center_manifold
        self._map_options = None

        super().__init__(domain_obj)
        self._generator = None
        self._section_coord = None

    @property
    def center_manifold(self) -> "CenterManifold":
        """The center manifold."""
        return self._center_manifold

    @property
    def energy(self) -> float:
        """The energy."""
        return self._energy

    @property
    def hamsys(self) -> _HamiltonianSystem:
        """The Hamiltonian system."""
        return self.center_manifold.dynamics.hamsys

    def compute(self, *, section_coord: str = "q3", options: "CenterManifoldMapOptions" = None):
        """Compute or retrieve the return map for the specified section.
        
        Parameters
        ----------
        section_coord : str, default="q3"
            The section coordinate.
        options : :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`, optional
            Runtime options for the map computation. If None, uses self.map_options.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.types.CenterManifoldMapResults`
            The results of the Poincare map.
        """
        # Use self.map_options if options not provided
        if options is None:
            options = self.map_options
        
        cache_key = self.make_key("generate", section_coord, tuple(sorted(options.to_dict().items())))

        def _factory() -> CenterManifoldDomainPayload:
            self.generator.update_config(section_coord=section_coord)
            result = self.generator.generate(self.domain_obj, options)
            payload = CenterManifoldDomainPayload._from_mapping(
                {
                    "points": result.points,
                    "states": result.states,
                    "times": result.times,
                    "labels": result.labels,
                    "section_coord": section_coord,
                }
            )
            self.apply_center_manifold_map(payload, section_coord=section_coord)
            return payload

        payload = self.get_or_create(cache_key, _factory)
        return CenterManifoldMapResults(
            payload.points,
            payload.states,
            payload.labels,
            payload.times,
        )

    def get_points_with_4d_states(
        self,
        *,
        section_coord: str | None = None,
        axes: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Return 2-D projection of the Poincare map points with 4D state access.

        This method extends the base implementation to allow projections
        mixing plane coordinates with the missing coordinate by using the
        stored 4-D center manifold states.

        Parameters
        ----------
        section_coord : str
            The section coordinate.
        axes : tuple[str, str]
            The axes to project onto.
        """
        if axes is None:
            return self.get_points(section_coord=section_coord)

        key = section_coord or self.section_coord

        # Ensure section is computed and cached
        if key not in self._sections:
            self.compute(section_coord=key)
        sec = self._sections[key]

        # Mapping for full 4-D CM state stored in `sec.states`
        state_map = {"q2": 0, "p2": 1, "q3": 2, "p3": 3}

        cols = []
        for ax in axes:
            if ax in sec.labels:
                idx = sec.labels.index(ax)
                cols.append(sec.points[:, idx])
            elif ax in state_map:
                cols.append(sec.states[:, state_map[ax]])
            else:
                raise ValueError(
                    f"Axis '{ax}' not recognised; allowed are q2, p2, q3, p3"
                )

        return np.column_stack(cols)

    def to_synodic(self, poincare_point: np.ndarray, section_coord: Optional[str], tol: float) -> np.ndarray:
        """Convert the Poincare point to synodic coordinates.
        
        Parameters
        ----------
        poincare_point : np.ndarray
            The Poincaré point.
        section_coord : str
            The section coordinate.
        tol : float
            The tolerance.

        Returns
        -------
        np.ndarray
            The synodic coordinates.
        """
        if section_coord is None:
            section_coord = self.section_coord
        return self.center_manifold.dynamics.cm_point_to_synodic(cm_point=poincare_point, energy=self.energy, section_coord=section_coord, tol=tol)

    def _to_real_4d_cm(
        self,
        poincare_point: np.ndarray,
        section_coord: str,
    ) -> np.ndarray:
        """Convert the Poincare point to real 4D center manifold coordinates.
        
        Parameters
        ----------
        poincare_point : np.ndarray
            The Poincare point.
        section_coord : str
            The section coordinate.
        
        Returns
        -------
        np.ndarray
            The real 4D center manifold coordinates.
        """
        # Get plane coordinates from the generator's interface
        plane_coords = self.generator._get_interface().plane_labels(section_coord)

        known_vars: Dict[str, float] = {section_coord: 0.0}
        known_vars[plane_coords[0]] = float(poincare_point[0])
        known_vars[plane_coords[1]] = float(poincare_point[1])

        solved_val = self.generator._get_interface().lift_plane_point(
            (float(poincare_point[0]), float(poincare_point[1])),
            section_coord=section_coord,
            h0=float(self.energy),
            H_blocks=self.hamsys.poly_H(),
            clmo_table=self.hamsys.clmo_table,
        )

        if solved_val is None:
            raise RuntimeError("Failed to reconstruct full CM coordinates - root finding did not converge.")

        real_4d_cm = np.array([
            solved_val[0],  # q2
            solved_val[1],  # p2
            solved_val[2],  # q3
            solved_val[3],  # p3
        ], dtype=np.float64)

        return real_4d_cm

    def _propagate_from_point(
        self,
        cm_point,
        energy,
        *,
        steps=1000,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order=6,
    ):
        """Propagate a trajectory from a center manifold point.

        Parameters
        ----------
        cm_point : ndarray, shape (2,)
            Point on the center manifold section.
        energy : float
            Energy level for the trajectory (nondimensional units).
        steps : int, default=1000
            Number of integration steps.
        method : {'fixed', 'adaptive', 'symplectic'}, default='adaptive'
            Integration method.
        order : int, default=6
            Integration order for Runge-Kutta methods.

        Returns
        -------
        :class:`~hiten.system.orbits.base.GenericOrbit`
            Propagated orbit object.
        """
        ic = self.domain_obj.to_synodic(cm_point, section_coord=self.section_coord)
        orbit = GenericOrbit(self.domain_obj.center_manifold.point, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    def _build_generator(self) -> CenterManifoldMapPipeline:
        """Build the generator."""
        return CenterManifoldMapPipeline.with_default_engine(config=self.map_config)

    def _default_map_config(self) -> CenterManifoldMapConfig:
        """Create the default map configuration for center manifold maps.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
            The default map configuration.
        """
        return CenterManifoldMapConfig(
            seed_strategy="axis_aligned",
            seed_axis=None,
            section_coord="q3",
            integration=IntegrationConfig(method="fixed")
        )

    @property
    def map_options(self) -> "CenterManifoldMapOptions":
        """Get the map options for center manifold maps.
        
        Returns the stored options, lazily initializing with defaults if needed.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            The map options.
        """
        if self._map_options is None:
            self._map_options = self._default_map_options()
        return self._map_options
    
    def _default_map_options(self) -> "CenterManifoldMapOptions":
        """Create the default map options for center manifold maps.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            The default map options.
        """
        from hiten.algorithms.poincare.centermanifold.options import \
            CenterManifoldMapOptions
        from hiten.algorithms.poincare.core.options import (IterationOptions,
                                                            SeedingOptions)
        from hiten.algorithms.types.options import (IntegrationOptions,
                                                    WorkerOptions)
        
        return CenterManifoldMapOptions(
            integration=IntegrationOptions(
                dt=0.01,
                order=4,
                c_omega_heuristic=20,
                max_steps=2000,
            ),
            iteration=IterationOptions(n_iter=40),
            seeding=SeedingOptions(n_seeds=20),
            workers=WorkerOptions(n_workers=8),
        )
    
    @map_options.setter
    def map_options(self, value: "CenterManifoldMapOptions") -> None:
        """Set runtime options for map computation.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            New map options.
        """
        self._map_options = value

    def apply_center_manifold_map(self, payload: CenterManifoldDomainPayload, *, section_coord: str | None = None) -> CenterManifoldDomainPayload:
        result = CenterManifoldMapResults(
            payload.points,
            payload.states,
            payload.labels,
            payload.times,
        )
        section_id = section_coord or payload.section_coord or self.section_coord
        self._sections[section_id] = result
        self._section_coord = section_id
        self.domain_obj._last_map = result
        return payload


class _SynodicMapDynamicsService(_MapDynamicsServiceBase):
    """Dynamics service for synodic maps with detection-based computation.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.maps.synodic.SynodicMap`
        The domain object.

    Attributes
    ----------
    trajectories : List[:class:`~hiten.algorithms.types.states.Trajectory`]
        The trajectories.
    source : Literal[:class:`~hiten.system.orbits.base.PeriodicOrbit`, :class:`~hiten.system.manifold.Manifold`]
        The source.
    map_options : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
        Runtime options for map computation.
    """

    def __init__(self, domain_obj: "SynodicMap") -> None:
        self._trajectories = domain_obj._trajectories
        self._source = domain_obj._source
        self._map_options = None
        super().__init__(domain_obj)

    @property
    def trajectories(self) -> List[Trajectory]:
        """The trajectories."""
        return self._trajectories

    @property
    def source(self) -> Literal[PeriodicOrbit, Manifold]:
        """The source."""
        return self._source

    def compute(self, *, section_axis: str, section_offset: float, plane_coords: tuple[str, str], direction: Literal[1, -1, None], options: "SynodicMapOptions" = None):
        """Compute the synodic map.
        
        Parameters
        ----------
        section_axis : str
            The section axis.
        section_offset : float
            The section offset.
        plane_coords : tuple[str, str]
            The plane coordinates.
        direction : Literal[1, -1, None]
            The direction.
        options : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`, optional
            Runtime options for the map computation. If None, uses self.map_options.
        """
        # Use self.map_options if options not provided
        if options is None:
            options = self.map_options
        
        cache_key = self.make_key("generate", section_axis, section_offset, tuple(plane_coords), direction, tuple(sorted(options.to_dict().items())))

        def _factory() -> SynodicMapDomainPayload:
            updates = {"section_axis": section_axis,
                        "section_offset": section_offset,
                        "plane_coords": plane_coords,
                        "direction": direction}

            self.generator.update_config(**updates)
            result = self.generator.generate(self.source, options)
            payload = SynodicMapDomainPayload._from_mapping(
                {
                    "points": result.points,
                    "states": result.states,
                    "times": result.times,
                    "trajectory_indices": result.trajectory_indices,
                    "labels": result.labels,
                }
            )
            self.apply_synodic_map(
                payload,
                section_axis=section_axis,
                section_offset=section_offset,
                plane_coords=plane_coords,
                direction=direction,
            )
            return payload

        return self.get_or_create(cache_key, _factory)

    def _build_generator(self) -> SynodicMapPipeline:
        """Build the generator."""
        return SynodicMapPipeline.with_default_engine(config=self.map_config)

    def _default_map_config(self) -> SynodicMapConfig:
        """Create the default map configuration for synodic maps.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
            The default map configuration.
        """
        return SynodicMapConfig(
            section_axis="x",
            section_offset=0.0,
            section_normal=None,
            plane_coords=("y", "vy"),
            direction=None,
            interp_kind=RefineConfig(interp_kind="cubic"),
        )
    
    @property
    def map_options(self) -> "SynodicMapOptions":
        """Get the map options for synodic maps.
        
        Returns the stored options, lazily initializing with defaults if needed.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            The map options.
        """
        if self._map_options is None:
            self._map_options = self._default_map_options()
        return self._map_options
    
    def _default_map_options(self) -> "SynodicMapOptions":
        """Create the default map options for synodic maps.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            The default map options.
        """
        from hiten.algorithms.poincare.synodic.options import SynodicMapOptions
        from hiten.algorithms.types.options import RefineOptions, WorkerOptions
        
        return SynodicMapOptions(
            refine=RefineOptions(
                segment_refine=50,
                tol_on_surface=1e-6,
                dedup_time_tol=1e-9,
                dedup_point_tol=1e-6,
                max_hits_per_traj=None,
                newton_max_iter=10,
            ),
            workers=WorkerOptions(n_workers=8),
        )
    
    @map_options.setter
    def map_options(self, value: "SynodicMapOptions") -> None:
        """Set runtime options for map computation.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            New map options.
        """
        self._map_options = value

    def apply_synodic_map(self, payload: SynodicMapDomainPayload, *, section_axis: str | None = None, section_offset: float | None = None, plane_coords: tuple[str, str] | None = None, direction: Literal[1, -1, None] | None = None) -> SynodicMapDomainPayload:
        """Apply a synodic map update to the map service.

        Parameters
        ----------
        payload : SynodicMapDomainPayload
            Payload describing the computed synodic map.
        section_axis : str, optional
            Axis defining the section plane.
        section_offset : float, optional
            Offset along the section axis.
        plane_coords : tuple[str, str], optional
            Coordinate labels for the section plane projection.
        direction : {1, -1, None}, optional
            Crossing direction filter.
        """
        section_axis = section_axis or self.map_config.section_axis
        plane_coords = plane_coords or self.map_config.plane_coords
        direction = direction if direction is not None else self.map_config.direction
        section_id = self._make_section_key(section_axis, section_offset or self.map_config.section_offset, plane_coords, direction)
        results = SynodicMapResults(
            payload.points, payload.states, payload.labels, payload.times, payload.trajectory_indices
        )
        self._sections[section_id] = results
        self._section_coord = section_id
        return payload

    @staticmethod
    def _make_section_key(section_axis: str, section_offset: float | None, plane_coords: tuple[str, str], direction: Literal[1, -1, None]) -> str:
        return f"{section_axis}:{section_offset}:{plane_coords[0]}:{plane_coords[1]}:{direction}"


class _MapServices(_ServiceBundleBase):
    """Bundle all map services together.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.maps.base.Map`
        The domain object.

    Attributes
    ----------
    dynamics : :class:`~hiten.algorithms.types.services.maps._MapDynamicsServiceBase`
        The dynamics service.
    persistence : :class:`~hiten.algorithms.types.services.maps._MapPersistenceService`
        The persistence service.
    """
    
    def __init__(self, domain_obj, persistence: _MapPersistenceService, dynamics: _MapDynamicsServiceBase) -> None:
        super().__init__(domain_obj)
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, domain_obj) -> "_MapServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.maps.base.Map`
            The domain object.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.services.maps._MapServices`
            The service bundle.
        """
        dynamics = cls._check_map_type(domain_obj)
        return cls(
            domain_obj,
            _MapPersistenceService(),
            dynamics(domain_obj)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _MapDynamicsServiceBase) -> "_MapServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.maps._MapDynamicsServiceBase`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.maps._MapServices`
            The service bundle.
        """
        return cls(
            dynamics.domain_obj,
            _MapPersistenceService(),
            dynamics
        )

    @staticmethod
    def _check_map_type(domain_obj) -> type:
        """Check the type of the map and return the corresponding dynamics service.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.maps.base.Map`
            The map.

        Returns
        -------
        type
            The dynamics service.
        """
        from hiten.system.maps.center import CenterManifoldMap
        from hiten.system.maps.synodic import SynodicMap

        mapping = {
            CenterManifoldMap: _CenterManifoldMapDynamicsService,
            SynodicMap: _SynodicMapDynamicsService,
        }

        return mapping[type(domain_obj)]