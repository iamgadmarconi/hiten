"""User-facing interface for synodic Poincare sections.

This module provides the main `SynodicMap` class that serves as the
user-facing interface for synodic Poincare section detection on
precomputed trajectories. It implements a facade pattern that mirrors
the API of other return-map modules while providing specialized
functionality for synodic sections.

The main class :class:`~hiten.algorithms.poincare.synodic.base.SynodicMap` extends the abstract base class
to provide detection capabilities on precomputed trajectory data,
including support for orbits, manifolds, and custom trajectories.

"""

from typing import Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.core.base import _DetectionMapBase, _Section
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
from hiten.algorithms.poincare.synodic.interfaces import (
    _SynodicEngineConfig, _SynodicInterface, _SynodicSectionInterface)
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.plots import plot_poincare_map


class SynodicMap(_DetectionMapBase):
    """User-facing interface for synodic Poincare section detection.

    This class provides a facade that mirrors the API of other return-map
    modules while specializing in synodic Poincare section detection on
    precomputed trajectories. It does not propagate trajectories; callers
    supply them explicitly through various input methods.

    Parameters
    ----------
    map_cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`, optional
        Configuration object containing detection parameters, section geometry,
        and refinement settings. If None, uses default configuration.

    Attributes
    ----------
    config : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The map configuration object.
    _section_iface : :class:`~hiten.algorithms.poincare.synodic.interfaces._SynodicSectionInterface`
        The section interface derived from the map configuration.
    _engine : :class:`~hiten.algorithms.poincare.synodic.engine._SynodicEngine`
        The engine that coordinates detection and refinement.
    _sections : dict[str, :class:`~hiten.algorithms.poincare.core.base._Section`]
        Cache of computed sections keyed by section parameters.
    _section : :class:`~hiten.algorithms.poincare.core.base._Section` or None
        The most recently computed section.

    Notes
    -----
    This class implements a facade pattern that provides a consistent
    interface for synodic Poincare section detection while hiding the
    complexity of the underlying detection and refinement algorithms.

    The class supports multiple input methods:
    - Custom trajectories via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_trajectories`
    - Periodic orbits via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_orbit`
    - Manifold structures via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_manifold`

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self, map_cfg: Optional[_SynodicMapConfig] = None, *, _engine: "_SynodicEngine | None" = None) -> None:
        cfg = map_cfg or _SynodicMapConfig()
        super().__init__(cfg)
        self._section_iface = self._build_section_interface(cfg)
        self._engine: "_SynodicEngine | None" = _engine

    @classmethod
    def with_default_engine(
        cls,
        map_cfg: Optional[_SynodicMapConfig] = None,
    ) -> "SynodicMap":
        """Construct a facade with a default-wired engine injected (DI-friendly)."""
        inst = cls(map_cfg)
        inst._engine = inst._build_engine()
        return inst

    def _build_section_interface(self, cfg: _SynodicMapConfig) -> _SynodicSectionInterface:
        """Build section configuration from map configuration.

        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
            Map configuration containing section geometry parameters.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.config._SynodicSectionInterface`
            Section configuration object with normalized geometry.

        Notes
        -----
        This method translates user-facing geometry fields from the map
        configuration into a cached section configuration. It handles
        both explicit normal vectors and axis-based specifications.

        The method supports two geometry specification modes:
        1. Explicit normal vector via `section_normal`
        2. Axis-based specification via `section_axis` (string or index)

        The resulting section configuration is optimized for efficient
        use in the detection backend.
        """
        # Translate user-facing geometry fields into a cached section config
        if cfg.section_normal is not None:
            normal = np.asarray(cfg.section_normal, dtype=float)
        else:
            normal = _SynodicSectionInterface.axis_normal(cfg.section_axis)
        return _SynodicSectionInterface.from_normal(normal=normal, offset=cfg.section_offset, plane_coords=cfg.plane_coords)

    def _build_engine(self) -> _SynodicEngine:
        """Build the detection engine for synodic Poincare sections.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.engine._SynodicEngine`
            Configured detection engine with backend and strategy.

        Notes
        -----
        This method creates the detection engine by:
        1. Creating a configuration adapter for the engine
        2. Building the detection backend with section and map configurations
        3. Setting up the no-op seeding strategy (since we use precomputed trajectories)
        4. Assembling the engine with all components

        The engine coordinates the detection and refinement process
        for synodic Poincare sections.
        """
        adapter = _SynodicEngineConfig.from_config(self.config, self._section_iface)
        backend = _SynodicDetectionBackend(section_cfg=adapter.section_interface, map_cfg=adapter.config)
        strategy = _NoOpStrategy(adapter.section_interface, adapter)
        interface = _SynodicInterface()
        return _SynodicEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=adapter,
            interface=interface,
        )

    def from_trajectories(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: Literal[1, -1, None] = None,
        recompute: bool = False,
    ) -> _Section:
        """Compute synodic Poincare section from custom trajectories.

        Parameters
        ----------
        trajectories : sequence of tuple[ndarray, ndarray]
            Sequence of (times, states) tuples for each trajectory.
            Each tuple contains:
            - times: ndarray, shape (n,) - Time points (nondimensional units)
            - states: ndarray, shape (n, 6) - State vectors at each time point
        direction : {1, -1, None}, optional
            Crossing direction filter. If None, uses the default
            direction from the section configuration.
        recompute : bool, default False
            Whether to recompute the section even if it's already cached.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed synodic Poincare section containing:
            - points: 2D projected coordinates
            - states: Full 6D state vectors
            - times: Crossing times
            - labels: Coordinate labels for the projection

        Notes
        -----
        This method computes a synodic Poincare section from a collection
        of precomputed trajectories. It uses the configured detection
        backend to find crossings and applies the appropriate refinement
        method (linear or cubic) based on the configuration.

        The method automatically handles:
        - Crossing detection and refinement
        - Deduplication of nearby crossings
        - Ordering of crossings by time
        - Caching of results for future use

        All time units are in nondimensional units.
        """
        if self._engine is None:
            self._engine = self._build_engine()

        interface = self._engine._interface
        problem = interface.create_problem(
            direction=direction,
            n_workers=int(self.config.n_workers or 0),
            trajectories=trajectories,
        )
        sec = self._engine.solve(problem)

        key = self._section_key()
        self._sections[key] = sec
        self._section = sec
        return sec

    def from_orbit(self, orbit: PeriodicOrbit, *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> _Section:
        """Compute synodic Poincare section from a periodic orbit.

        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The periodic orbit to analyze. Must be propagated before calling
            this method (i.e., `orbit.times` and `orbit.trajectory` must not be None).
        direction : {1, -1, None}, optional
            Crossing direction filter. If None, uses the default
            direction from the section configuration.
        recompute : bool, default False
            Whether to recompute the section even if it's already cached.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed synodic Poincare section from the periodic orbit.

        Raises
        ------
        ValueError
            If the orbit has not been propagated (times or trajectory is None).

        Notes
        -----
        This method extracts the trajectory data from a periodic orbit
        and computes the corresponding synodic Poincare section. The
        orbit must be propagated before calling this method.

        The method automatically converts the orbit's time and state
        data to the appropriate format for the detection backend.

        All time units are in nondimensional units.
        """
        if orbit.times is None or orbit.trajectory is None:
            raise ValueError("Orbit must be propagated before extracting trajectories")
        traj = [(np.asarray(orbit.times, dtype=np.float64), np.asarray(orbit.trajectory, dtype=np.float64))]
        return self.from_trajectories(traj, direction=direction, recompute=recompute)

    def from_manifold(self, manifold, *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> _Section:
        """Compute synodic Poincare section from a manifold structure.

        Parameters
        ----------
        manifold : manifold object
            The manifold object containing trajectory data. Must have
            a `manifold_result` attribute with `times_list` and `states_list`.
        direction : {1, -1, None}, optional
            Crossing direction filter. If None, uses the default
            direction from the section configuration.
        recompute : bool, default False
            Whether to recompute the section even if it's already cached.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed synodic Poincare section from the manifold.

        Raises
        ------
        ValueError
            If the manifold result contains no valid trajectories.

        Notes
        -----
        This method extracts trajectory data from a manifold structure
        and computes the corresponding synodic Poincare section. The
        manifold must contain valid trajectory data in the expected format.

        The method automatically validates the trajectory data and
        converts it to the appropriate format for the detection backend.
        It filters out invalid trajectories and ensures all trajectories
        have the correct dimensions (6D state vectors).

        All time units are in nondimensional units.
        """
        manifold_result = manifold.manifold_result
        trajs = []
        for times, states in zip(getattr(manifold_result, "times_list", []), getattr(manifold_result, "states_list", [])):
            if times is None or states is None:
                continue
            t_arr = np.asarray(times, dtype=np.float64)
            x_arr = np.asarray(states, dtype=np.float64)
            if t_arr.ndim == 1 and x_arr.ndim == 2 and len(t_arr) == len(x_arr) and x_arr.shape[1] == 6:
                trajs.append((t_arr, x_arr))

        if not trajs:
            raise ValueError("Manifold result contains no valid trajectories")
        return self.from_trajectories(trajs, direction=direction, recompute=recompute)

    def _section_key(self) -> str:
        """Generate a stable cache key for the section.

        Returns
        -------
        str
            A canonical string key that uniquely identifies the section
            based on its geometry parameters.

        Notes
        -----
        This method generates a stable cache key that uniquely identifies
        the section based on its geometry parameters. The key includes:
        - Plane coordinate labels
        - Section offset value
        - Normal vector components

        The key is designed to be stable across different runs and
        provides a canonical identifier for caching purposes.
        """
        n = self._section_iface.normal
        n_key = ",".join(f"{float(v):.12g}" for v in n.tolist())
        c = float(self._section_iface.offset)
        i, j = self._section_iface.plane_coords
        return f"synodic[{i},{j}]_c={c:.12g}_n=({n_key})"

    def plot(
        self,
        *,
        axes: Sequence[str] | None = None,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        **kwargs,
    ):
        """Render a 2D Poincare map for the last computed synodic section.

        Parameters
        ----------
        axes : sequence of str, optional
            Coordinate axes to plot. If None, uses the default
            plane coordinates from the section configuration.
        dark_mode : bool, default True
            Whether to use dark mode for the plot.
        save : bool, default False
            Whether to save the plot to a file.
        filepath : str, default "poincare_map.svg"
            File path for saving the plot (only used if save=True).
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.

        Raises
        ------
        ValueError
            If no synodic section has been computed yet.

        Notes
        -----
        This method renders a 2D Poincare map for the most recently
        computed synodic section. It requires that `from_orbit`,
        `from_manifold`, or `from_trajectories` has been called to
        populate the cached section.

        The method supports custom axis selection and automatically
        handles the projection of the section data to 2D coordinates
        for visualization.

        The plot shows the Poincare section points in the specified
        coordinate system, providing a visual representation of the
        section's structure.
        """
        if self._section is None:
            raise ValueError("No synodic section cached. Compute from orbit or manifold first.")

        section = self._section

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            cols = []
            for ax in axes:
                if ax in section.labels:
                    idx = section.labels.index(ax)
                    cols.append(section.points[:, idx])
                else:
                    idx = self._section_iface.coordinate_index(ax)
                    cols.append(section.states[:, idx])
            pts = np.column_stack(cols)
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )
