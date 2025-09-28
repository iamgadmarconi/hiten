"""Adapters supporting Poincare map numerics and persistence."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.centermanifold.base import \
    _CenterManifoldMapFacade
from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.types import \
    CenterManifoldMapResults
from hiten.algorithms.poincare.core.types import _Section
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.system.center import CenterManifold
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io.map import load_poincare_map, save_poincare_map

if TYPE_CHECKING:
    from hiten.algorithms.poincare.core.types import _Section
    from hiten.system.orbits.base import PeriodicOrbit



class _MapPersistenceService(_PersistenceServiceBase):
    """Handle persistence for map objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda map, path, **kw: save_poincare_map(map, Path(path), **kw),
            load_fn=lambda path, **kw: load_poincare_map(Path(path), **kw),
        )


class _MapDynamicsServiceBase(_DynamicsServiceBase):
    """Base class for map dynamics services with caching."""

    def __init__(self, domain_obj) -> None:
        super().__init__(domain_obj)
        self._sections: dict[str, "_Section"] = {}
        self._section: Optional["_Section"] = None
        self._section_coord = None

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
    """Dynamics service for center manifold maps with caching."""

    def __init__(self, domain_obj: "CenterManifold") -> None:
        super().__init__(domain_obj)
        self._generator = None
        self._section_coord = None

    @property
    def generator(self) -> _CenterManifoldMapFacade:
        if self._generator is None:
            self._generator = _CenterManifoldMapFacade.with_default_engine(config=self.map_config)
        return self._generator

    @property
    def domain_obj(self) -> "CenterManifold":
        return super().domain_obj

    def compute(self, *, section_coord: str = "q3", overrides: dict[str, Any] | None = None, **kwargs) -> np.ndarray:
        if overrides is None:
            overrides = {}
        else:
            overrides = overrides.copy()
        overrides.update(kwargs)
        
        overrides_tuple = tuple(sorted(overrides.items())) if overrides else ()
        cache_key = self.make_key("generate", section_coord, overrides_tuple)

        def _factory() -> CenterManifoldMapResults:
            override = bool(overrides)
            updates = {"section_coord": section_coord}
            self.generator.update_config(**updates)
            results = self.generator.generate(self.domain_obj, override=override, **overrides)
            return results

        return self.get_or_create(cache_key, _factory)

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
        """
        if axes is None:
            return self.get_points(section_coord=section_coord)

        key = section_coord or self.section_coord

        # Ensure section is computed and cached
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

        # Stack the two 1-D arrays column-wise into shape (n, 2)
        return np.column_stack(cols)

    def to_synodic(self, pt: np.ndarray, *, section_coord: str | None = None) -> np.ndarray:
        """Convert a plane point to initial conditions for integration.

        Parameters
        ----------
        pt : ndarray, shape (2,)
            Point on the Poincare section plane.
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default
            section coordinate from configuration.

        Returns
        -------
        ndarray, shape (6,)
            Initial conditions [q1, q2, q3, p1, p2, p3] for integration.
        """
        key = section_coord or self.section_coord
        return self.domain_obj.to_synodic(pt, self._energy, section_coord=key)

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
        ic = self.domain_obj.to_synodic(cm_point, energy, section_coord=self.config.section_coord)
        orbit = GenericOrbit(self.domain_obj.point, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    @property
    def map_config(self) -> _CenterManifoldMapConfig:
        return _CenterManifoldMapConfig(
            n_seeds=20,
            n_iter=40,
            dt=0.01,
            method="fixed",
            order=4,
            c_omega_heuristic=20,
            max_steps=2000,
            n_workers=8,
            seed_strategy="axis_aligned",
            seed_axis=None,
            section_coord="q3"
        )

    @map_config.setter
    def map_config(self, value: _CenterManifoldMapConfig):
        self.generator._set_config(value)


class _SynodicMapDynamicsService(_MapDynamicsServiceBase):
    """Dynamics service for synodic maps with detection-based computation."""

    def __init__(self, domain_obj) -> None:
        super().__init__(domain_obj)
        self._engine = None

    def _get_or_compute_section(self, key: str) -> "_Section":
        """Return the cached section for synodic maps."""
        if key not in self._sections:
            raise KeyError(
                f"Section '{key}' has not been computed. "
                f"Available: {list(self._sections.keys())}"
            )
        return self._sections[key]

    def from_trajectories(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: Literal[1, -1, None] = None,
        recompute: bool = False,
    ) -> "_Section":
        """Compute synodic Poincare section from custom trajectories."""
        if self._engine is None:
            self._engine = self._build_engine()

        interface = self._engine._interface
        problem = interface.create_problem(
            config=self.domain_obj.config,
            plane_coords=self.domain_obj._section_iface.plane_coords,
            normal=self.domain_obj._section_iface.normal,
            offset=self.domain_obj._section_iface.offset,
            direction=direction,
            trajectories=trajectories,
        )
        sec = self._engine.solve(problem)

        key = self._section_key()
        self._sections[key] = sec
        self._section = sec
        return sec

    def from_orbit(self, orbit: "PeriodicOrbit", *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> "_Section":
        """Compute synodic Poincare section from a periodic orbit."""
        if orbit.trajectory is None:
            raise ValueError("Orbit must be propagated before extracting trajectories")
        traj = [(orbit.trajectory.times, orbit.trajectory.states)]
        return self.from_trajectories(traj, direction=direction, recompute=recompute)

    def from_manifold(self, manifold, *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> "_Section":
        """Compute synodic Poincare section from a manifold structure."""
        if manifold.trajectories is None:
            raise ValueError("Manifold must be computed before extracting trajectories")
        
        trajs = []
        for traj in manifold.trajectories:
            if traj is not None:
                trajs.append((traj.times, traj.states))

        if not trajs:
            raise ValueError("Manifold result contains no valid trajectories")
        return self.from_trajectories(trajs, direction=direction, recompute=recompute)

    def _section_key(self) -> str:
        """Generate a stable cache key for the section."""
        n = self.domain_obj._section_iface.normal
        n_key = ",".join(f"{float(v):.12g}" for v in n.tolist())
        c = float(self.domain_obj._section_iface.offset)
        i, j = self.domain_obj._section_iface.plane_coords
        return f"synodic[{i},{j}]_c={c:.12g}_n=({n_key})"

    def _build_engine(self):
        """Build the detection engine for synodic Poincare sections."""
        from hiten.algorithms.poincare.synodic.backend import \
            _SynodicDetectionBackend
        from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
        from hiten.algorithms.poincare.synodic.interfaces import (
            _SynodicEngineConfig, _SynodicInterface)
        from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy

        adapter = _SynodicEngineConfig.from_config(self.domain_obj.config, self.domain_obj._section_iface)
        backend = _SynodicDetectionBackend()
        strategy = _NoOpStrategy(adapter.section_interface, adapter)
        interface = _SynodicInterface()
        return _SynodicEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=adapter,
            interface=interface,
        )


class _MapServices(_ServiceBundleBase):
    """Bundle all map services together."""
    
    def __init__(self, domain_obj, persistence: _MapPersistenceService, dynamics: _MapDynamicsServiceBase) -> None:
        super().__init__(domain_obj)
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, domain_obj) -> "_MapServices":
        dynamics = cls._check_map_type(domain_obj)
        return cls(
            domain_obj,
            _MapPersistenceService(),
            dynamics(domain_obj)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _MapDynamicsServiceBase) -> "_MapServices":
        return cls(
            dynamics.domain_obj,
            _MapPersistenceService(),
            dynamics
        )

    @staticmethod
    def _check_map_type(domain_obj) -> type:
        from hiten.system.maps.center import CenterManifoldMap
        from hiten.system.maps.synodic import SynodicMap

        mapping = {
            CenterManifoldMap: _CenterManifoldMapDynamicsService,
            SynodicMap: _SynodicMapDynamicsService,
        }

        return mapping[type(domain_obj)]