"""
Center manifold Poincare map interface for the CR3BP.

This module provides the main user-facing interface for computing and
analyzing Poincare maps restricted to center manifolds of collinear
libration points in the Circular Restricted Three-Body Problem (CR3BP).

The :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap` 
class extends the base return map functionality with center manifold-specific seeding 
strategies and visualization capabilities.
"""

from typing import Literal, Optional, Sequence
from dataclasses import replace

import numpy as np

from hiten.algorithms.poincare.centermanifold.config import (
    _CenterManifoldMapConfig, _get_section_config)
from hiten.algorithms.poincare.centermanifold.engine import \
    _CenterManifoldEngine
from hiten.algorithms.poincare.centermanifold.interfaces import \
    _CenterManifoldInterface
from hiten.algorithms.poincare.centermanifold.strategies import _make_strategy
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldMapResults)
from hiten.algorithms.poincare.core.base import _ReturnMapBase
from hiten.system.center import CenterManifold
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io.map import (load_poincare_map, load_poincare_map_inplace,
                                save_poincare_map)
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive


class CenterManifoldMap(_ReturnMapBase):
    """Poincare return map restricted to the center manifold of a collinear libration point.

    This class provides the main interface for computing and analyzing Poincare
    maps on center manifolds in the CR3BP. It supports various seeding strategies
    and provides visualization capabilities for understanding the local dynamics.

    Parameters
    ----------
    cm : :class:`~hiten.system.center.CenterManifold`
        Center manifold object providing the underlying dynamical system.
    energy : float
        Energy level for the center manifold (nondimensional units).
    config : :class:`~hiten.algorithms.poincare.centermanifold.config._CenterManifoldMapConfig`, optional
        Configuration object specifying computation parameters. If None,
        default configuration is used.

    Notes
    -----
    State vectors are ordered as [q1, q2, q3, p1, p2, p3] where q1=0 for
    center manifold trajectories. All coordinates are in nondimensional units
    with the primary-secondary separation as the length unit.

    Examples
    --------
    >>> from hiten.system.center import CenterManifold
    >>> from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
    >>> 
    >>> # Create center manifold for L1 point
    >>> cm = CenterManifold("L1")
    >>> 
    >>> # Create Poincare map at specific energy
    >>> energy = -1.5
    >>> poincare_map = CenterManifoldMap(cm, energy)
    >>> 
    >>> # Compute the map
    >>> poincare_map.compute()
    >>> 
    >>> # Plot the results
    >>> poincare_map.plot()
    """

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[_CenterManifoldMapConfig] = None,
        *,
        _engine: _CenterManifoldEngine | None = None,
    ) -> None:
        self.cm: CenterManifold = cm
        self._energy: float = float(energy)

        # If caller does not supply a config, fall back to defaults.
        cfg = config or _CenterManifoldMapConfig()

        # Ensure injected engine is available even if base __init__ triggers compute()
        self._injected_engine: _CenterManifoldEngine | None = _engine

        super().__init__(cfg)

    @classmethod
    def with_default_engine(
        cls,
        cm: CenterManifold,
        energy: float,
        config: Optional[_CenterManifoldMapConfig] = None,
    ) -> "CenterManifoldMap":
        """Construct a map with a default-wired engine injected.

        This mirrors the DI-friendly facades (e.g., Connection) by creating
        a default engine using the current configuration and injecting it.
        The engine is wired for the default section coordinate in the config.
        """
        inst = cls(cm, energy, config)
        default_key = inst.config.section_coord
        backend = inst._build_backend(default_key)
        strategy = inst._build_seeding_strategy(default_key)
        engine = inst._build_engine(backend, strategy)
        inst._injected_engine = engine
        # Make it available immediately for the default key
        inst._engines[default_key] = engine
        return inst

    @property
    def energy(self) -> float:
        """Energy level of the center manifold.

        Returns
        -------
        float
            Energy level in nondimensional units.
        """
        return self._energy

    def _build_backend(self, section_coord: str):
        """Return (and cache) a center manifold backend via the owning CenterManifold.

        Parameters
        ----------
        section_coord : str
            Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend`
            Configured backend for center manifold computations.
        """

        return self.cm._get_or_create_backend(
            self._energy,
            section_coord,
            method=self.config.method,
            order=self.config.order,
            c_omega_heuristic=self.config.c_omega_heuristic,
        )

    def _build_seeding_strategy(self, section_coord: str):
        """Return a seeding strategy configured for the specified section coordinate.

        Parameters
        ----------
        section_coord : str
            Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase`
            Configured seeding strategy for generating initial conditions.
        """

        sec_cfg = _get_section_config(section_coord)

        strategy_kwargs: dict[str, object] = {}
        if self.config.seed_strategy == "single":
            strategy_kwargs["seed_axis"] = self.config.seed_axis

        strategy = _make_strategy(
            self.config.seed_strategy,
            sec_cfg,
            self.config,
            **strategy_kwargs,
        )

        return strategy

    def _build_engine(self, backend, strategy):
        """Instantiate the concrete engine for center manifold maps.

        Parameters
        ----------
        backend : :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend`
            Backend for numerical computations.
        strategy : :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase`
            Seeding strategy for initial conditions.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.engine._CenterManifoldEngine`
            Configured engine for center manifold map computation.
        """
        return _CenterManifoldEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=self.config,
            interface=_CenterManifoldInterface(),
        )

    def compute(
        self,
        section_coord: str | None = None,
        *,
        # runtime integration/iteration overrides
        dt: float | None = None,
        n_iter: int | None = None,
        n_workers: int | None = None,
        # runtime backend overrides
        method: Literal["fixed", "adaptive", "symplectic"] | None = None,
        order: int | None = None,
        c_omega_heuristic: float | None = None,
        # runtime seeding overrides
        seed_strategy: str | None = None,
        seed_axis: str | None = None,
        n_seeds: int | None = None,
    ) -> np.ndarray:
        """Compute the section, supporting runtime overrides without mutating config.

        If no overrides are provided, this defers to the cached, default setup
        and persists the result. If any overrides are provided, a temporary
        engine is assembled for this call and the result is returned without
        polluting the persistent cache. In all cases, this method returns the
        2-D points of the section.
        """
        key = section_coord or self.config.section_coord

        # Fast path: no overrides → use existing lazy/cached pipeline
        if (
            dt is None and n_iter is None and n_workers is None
            and method is None and order is None and c_omega_heuristic is None
            and seed_strategy is None and seed_axis is None and n_seeds is None
        ):
            # Reuse existing engine/cache machinery
            if key not in self._sections:
                self._solve_and_cache(key)
            self._section = self._sections[key]
            return self._section.points

        # Build a temporary backend honoring runtime backend overrides
        backend = self.cm._get_or_create_backend(
            self._energy,
            key,
            method=(method or self.config.method),
            order=(order or self.config.order),
            c_omega_heuristic=(
                c_omega_heuristic if c_omega_heuristic is not None else self.config.c_omega_heuristic  # type: ignore[arg-type]
            ),
        )

        # Build a temporary seeding strategy (honor runtime seeding overrides)
        sec_cfg = _get_section_config(key)
        final_seed_strategy = (seed_strategy or self.config.seed_strategy)
        final_seed_axis: str | None
        if final_seed_strategy == "single":
            final_seed_axis = seed_axis or self.config.seed_axis  # may remain None if not provided
        else:
            final_seed_axis = None

        # Clone current config to preserve all values, then override selected fields
        tmp_cfg = replace(
            self.config,
            section_coord=key,
            seed_strategy=final_seed_strategy,
            seed_axis=final_seed_axis,
            n_seeds=(int(n_seeds) if n_seeds is not None else self.config.n_seeds),
        )

        strategy = _make_strategy(tmp_cfg.seed_strategy, sec_cfg, tmp_cfg, seed_axis=tmp_cfg.seed_axis)

        # Assemble a one-off engine and solve with runtime iteration/integration overrides
        engine = _CenterManifoldEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=tmp_cfg,
            interface=_CenterManifoldInterface(),
        )

        results: CenterManifoldMapResults = engine.solve(
            dt=dt, n_iter=n_iter, n_workers=n_workers
        )

        # Do not pollute section cache when overrides are used, but expose via _section
        self._section = results
        return results.points

    def ic(self, pt: np.ndarray, *, section_coord: str | None = None) -> np.ndarray:
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
        key = section_coord or self.config.section_coord
        return self.cm.ic(pt, self._energy, section_coord=key)

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
        ic = self.cm.ic(cm_point, energy, section_coord=self.config.section_coord)
        logger.info("Initial conditions: %s", ic)
        orbit = GenericOrbit(self.cm.point, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    def plot(
        self,
        section_coord: str | None = None,
        *,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        axes: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot the Poincare map.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        dark_mode : bool, default=True
            If True, use dark mode styling.
        save : bool, default=False
            If True, save the plot to file.
        filepath : str, default='poincare_map.svg'
            File path for saving the plot.
        axes : Sequence[str], optional
            Axes to plot. If None, uses the section plane coordinates.
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        # Determine section
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self._solve_and_cache(section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self._solve_and_cache(None)
            section = self._section

        # Decide projection
        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )

    def plot_interactive(
        self,
        *,
        steps=1000,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order=6,
        frame="rotating",
        dark_mode: bool = True,
        axes: Sequence[str] | None = None,
        section_coord: str | None = None,
    ):
        """Create an interactive plot of the Poincare map.

        Parameters
        ----------
        steps : int, default=1000
            Number of integration steps for trajectory propagation.
        method : {'fixed', 'symplectic', 'adaptive'}, default='adaptive'
            Integration method for trajectory propagation.
        order : int, default=6
            Integration order for Runge-Kutta methods.
        frame : str, default='rotating'
            Reference frame for trajectory visualization.
        dark_mode : bool, default=True
            If True, use dark mode styling.
        axes : Sequence[str], optional
            Axes to plot. If None, uses the section plane coordinates.
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.

        Returns
        -------
        matplotlib.figure.Figure
            The interactive plot figure.

        Notes
        -----
        Clicking on points in the plot will propagate trajectories from
        those points and display the resulting orbits.
        """
        # Ensure section exists
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self._solve_and_cache(section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self._solve_and_cache(None)
            section = self._section

        def _on_select(pt_np: np.ndarray):
            if axes is None:
                section_pt = pt_np
            else:
                prev_sec = self._section
                self._section = section
                try:
                    proj_pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
                finally:
                    self._section = prev_sec
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                section_pt = section.points[np.argmin(distances)]

            orbit = self._propagate_from_point(
                section_pt,
                self.energy,
                steps=steps,
                method=method,
                order=order,
            )

            orbit.plot(frame=frame, dark_mode=dark_mode, block=False, close_after=False)

            return orbit

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map_interactive(
            points=pts,
            labels=lbls,
            on_select=_on_select,
            dark_mode=dark_mode,
        )

    def get_points(
        self,
        *,
        section_coord: str | None = None,
        axes: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Return 2-D projection of the Poincare map points.

        This method extends the base implementation to allow projections
        mixing plane coordinates with the missing coordinate by using the
        stored 4-D center manifold states.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        axes : tuple[str, str], optional
            Two coordinate axes to project onto. If None, uses the section
            plane coordinates. Allowed values are 'q2', 'p2', 'q3', 'p3'.

        Returns
        -------
        ndarray, shape (n_points, 2)
            Projected points in the specified coordinate system.

        Raises
        ------
        ValueError
            If an invalid axis name is specified.

        Notes
        -----
        The base implementation only knows about the two coordinates that span
        the section plane. This method extends it to permit projections mixing
        the plane coordinates with the missing coordinate by falling back to
        the stored 4-D center manifold states.
        """

        if axes is None:
            return super().get_points(section_coord=section_coord)

        key = section_coord or self.config.section_coord

        # Compute on-demand if missing
        if key not in self._sections:
            self._solve_and_cache(key)

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

    def _solve_and_cache(self, section_coord: str | None) -> None:
        key = section_coord or self.config.section_coord
        if key not in self._engines:
            # Prefer an injected engine if its backend section matches
            if (
                self._injected_engine is not None
                and getattr(self._injected_engine, "_backend", None) is not None
                and getattr(self._injected_engine._backend, "_section_cfg", None) is not None
                and self._injected_engine._backend._section_cfg.section_coord == key
            ):
                self._engines[key] = self._injected_engine
            else:
                backend = self._build_backend(key)
                strategy = self._build_seeding_strategy(key)
                self._engines[key] = self._build_engine(backend, strategy)

        engine = self._engines[key]
        results: CenterManifoldMapResults = engine.solve()
        self._section = results
        self._sections[key] = results

    def save(self, filepath: str, **kwargs) -> None:
        """Save the Poincare map to file.

        Parameters
        ----------
        filepath : str
            Path to save the map data.
        **kwargs
            Additional keyword arguments passed to the save function.
        """
        save_poincare_map(self, filepath, **kwargs)

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load Poincare map data from file in place.

        Parameters
        ----------
        filepath : str
            Path to load the map data from.
        **kwargs
            Additional keyword arguments passed to the load function.
        """
        load_poincare_map_inplace(self, filepath, **kwargs)

    @classmethod
    def load(
        cls,
        filepath: str,
        cm: CenterManifold,
        **kwargs,
    ) -> "CenterManifoldMap":
        """Load a Poincare map from file.

        Parameters
        ----------
        filepath : str
            Path to load the map data from.
        cm : :class:`~hiten.system.center.CenterManifold`
            Center manifold object for the loaded map.
        **kwargs
            Additional keyword arguments passed to the load function.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
            Loaded center manifold map instance.
        """
        return load_poincare_map(filepath, cm, **kwargs)
