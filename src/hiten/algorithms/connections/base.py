"""Provide a user-facing interface for discovering connections between manifolds in CR3BP.

This module provides the main :class:`~hiten.algorithms.connections.base.Connection` class, which serves as a
high-level facade for the connection discovery algorithm. It wraps the lower-level
connection engine and provides convenient methods for solving connection problems
and visualizing results.

The connection discovery process finds ballistic and impulsive transfers between
two manifolds by intersecting them with a common synodic section and analyzing
the geometric and dynamical properties of potential transfer points.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.connections.engine`
    Lower-level connection engine implementation.
:mod:`~hiten.algorithms.connections.results`
    Result classes for connection data.
:mod:`~hiten.system.manifold`
    Manifold classes for CR3BP invariant structures.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hiten.algorithms.connections.backends import _ConnectionsBackend
from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.engine import _ConnectionEngine
from hiten.algorithms.connections.types import (ConnectionResults,
                                                _ConnectionResult)
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.utils.exceptions import EngineError
from hiten.system.manifold import Manifold
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_poincare_connections_map


@dataclass
class Connection:
    """Provide a user-facing facade for connection discovery and plotting in CR3BP.

    This class provides a high-level interface for discovering ballistic and
    impulsive transfers between manifolds in the Circular Restricted Three-Body
    Problem. It wraps the lower-level connection engine and provides convenient
    methods for solving connection problems and visualizing results.

    Parameters
    ----------
    section : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        Configuration for the synodic section where manifolds are intersected.
    direction : {1, -1, None}, optional
        Direction for section crossings. 1 for positive, -1 for negative,
        None for both directions (default: None).
    search_cfg : :class:`~hiten.algorithms.connections.config._SearchConfig`, optional
        Configuration for connection search parameters including tolerances
        and geometric constraints (default: None).

    Examples
    --------

    >>> from hiten.algorithms.connections import Connection, SearchConfig
    >>> from hiten.algorithms.poincare import SynodicMapConfig
    >>> from hiten.system import System
    >>>
    >>> system = System.from_bodies("earth", "moon")
    >>> mu = system.mu

    >>> l1 = system.get_libration_point(1)
    >>> l2 = system.get_libration_point(2)
    >>> 
    >>> halo_l1 = l1.create_orbit('halo', amplitude_z=0.5, zenith='southern')
    >>> halo_l1.correct()
    >>> halo_l1.propagate()
    >>> 
    >>> halo_l2 = l2.create_orbit('halo', amplitude_z=0.3663368, zenith='northern')
    >>> halo_l2.correct()
    >>> halo_l2.propagate()
    >>> 
    >>> manifold_l1 = halo_l1.manifold(stable=True, direction='positive')
    >>> manifold_l1.compute(integration_fraction=0.9, step=0.005)
    >>> 
    >>> manifold_l2 = halo_l2.manifold(stable=False, direction='negative')
    >>> manifold_l2.compute(integration_fraction=1.0, step=0.005)
    >>> 
    >>> section_cfg = SynodicMapConfig(
    >>>     section_axis="x",
    >>>     section_offset=1 - mu,
    >>>     plane_coords=("y", "z"),
    >>>     interp_kind="cubic",
    >>>     segment_refine=30,
    >>>     tol_on_surface=1e-9,
    >>>     dedup_time_tol=1e-9,
    >>>     dedup_point_tol=1e-9,
    >>>     max_hits_per_traj=None,
    >>>     n_workers=None,
    >>> )
    >>> 
    >>> conn = Connection(
    >>>     section=section_cfg,
    >>>     direction=None,
    >>>     search_cfg=SearchConfig(delta_v_tol=1, ballistic_tol=1e-8, eps2d=1e-3),
    >>> )
    >>> 
    >>> conn.solve(manifold_l1, manifold_l2)
    >>> print(conn)
    >>> conn.plot(dark_mode=True)

    Notes
    -----
    The connection algorithm works by:
    1. Intersecting both manifolds with the specified synodic section
    2. Finding geometrically close points between the two intersection sets
    3. Refining matches using local segment geometry
    4. Computing Delta-V requirements and classifying transfers

    See Also
    --------
    :class:`~hiten.algorithms.connections.engine._ConnectionEngine`
        Lower-level engine that performs the actual computation.
    :class:`~hiten.algorithms.connections.types.ConnectionResults`
        Container for connection results with convenient access methods.
    """
    # User-provided single section configuration and direction
    section: _SynodicMapConfig
    direction: Literal[1, -1, None] | None = None

    # Optional search config
    search_cfg: _SearchConfig | None = None

    # Internal cache for plot convenience
    _last_source: Manifold | None = None
    _last_target: Manifold | None = None
    _last_results: list[_ConnectionResult] | None = None

    # Injected engine dependency (required)
    _engine: _ConnectionEngine | None = None

    @classmethod
    def with_default_engine(cls, *, section: _SynodicMapConfig, direction: Literal[1, -1, None] | None = None, search_cfg: _SearchConfig | None = None) -> "Connection":
        """Create a facade instance wired with the default engine and backend.

        The default engine uses :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`.

        Parameters
        ----------
        section : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
            Synodic section configuration.
        direction : {1, -1, None}, optional
            Crossing direction filter, by default None.
        search_cfg : :class:`~hiten.algorithms.connections.config._SearchConfig`, optional
            Search tolerances, by default None.

        Returns
        -------
        :class:`~hiten.algorithms.connections.base.Connection`
            A connection facade instance with a default engine injected.
        """
        backend = _ConnectionsBackend()
        engine = _ConnectionEngine(backend)
        return cls(section=section, direction=direction, search_cfg=search_cfg, _engine=engine)

    def solve(self, source: Manifold, target: Manifold) -> ConnectionResults:
        """Discover connections between two manifolds.

        This method finds ballistic and impulsive transfers between the source
        and target manifolds by intersecting them with the configured synodic
        section and analyzing potential connection points.

        Parameters
        ----------
        source : :class:`~hiten.system.manifold.Manifold`
            Source manifold (e.g., unstable manifold of a periodic orbit).
        target : :class:`~hiten.system.manifold.Manifold`
            Target manifold (e.g., stable manifold of another periodic orbit).

        Returns
        -------
        list of :class:`~hiten.algorithms.connections.types._ConnectionResult`
            Connection results sorted by increasing Delta-V requirement.
            Each result contains transfer type, Delta-V, intersection points,
            and 6D states at the connection.

        Notes
        -----
        Results are cached internally for convenient access via the 
        :attr:`~hiten.algorithms.connections.base.Connection.results`
        property and for plotting with the
        :meth:`~hiten.algorithms.connections.base.Connection.plot` method.

        The algorithm performs these steps:
        1. Convert manifolds to section interfaces
        2. Create connection problem specification
        3. Delegate to :class:`~hiten.algorithms.connections.engine._ConnectionEngine`
        4. Cache results for later use

        Examples
        --------
        >>> results = connection.solve(unstable_manifold, stable_manifold)
        >>> print(results)
        """
        logger.info(f"Searching for connection between {source} and {target}..")
        if self._engine is None:
            raise EngineError("Connection requires an injected _ConnectionEngine; provide via constructor.")
        results = self._engine.solve(
            source=source,
            target=target,
            section=self.section,
            direction=self.direction,
            search=self.search_cfg,
        )
        self._last_source = source
        self._last_target = target
        self._last_results = results
        logger.info(f"Found {len(results)} connection(s)")
        return ConnectionResults(results)

    @property
    def results(self) -> ConnectionResults:
        """Access the latest connection results with convenient formatting.

        Returns
        -------
        :class:`~hiten.algorithms.connections.types.ConnectionResults`
        :class:`~hiten.algorithms.connections.types.ConnectionResults` 
            A view over the latest results with friendly printing and
            convenient access methods. Returns an empty view if 
            :meth:`~hiten.algorithms.connections.base.Connection.solve`
            has not been called yet.

        Notes
        -----
        This property provides access to cached results from the most recent
        call to :meth:`~hiten.algorithms.connections.base.Connection.solve`. 
        The :class:`~hiten.algorithms.connections.results.ConnectionResults` 
        wrapper provides enhanced formatting and filtering capabilities.

        Examples
        --------
        >>> connection.solve(source, target)
        >>> print(connection.results)  # Pretty-printed summary
        >>> ballistic = connection.results.ballistic  # Filter by type
        """
        return ConnectionResults(self._last_results)

    def __repr__(self) -> str:
        n = len(self._last_results or [])
        sec = type(self.section).__name__ if self.section is not None else "None"
        return f"Connection(section={sec}, direction={self.direction}, results={n})"

    def __str__(self) -> str:
        header = self.__repr__()
        res_str = str(self.results)
        return f"{header}\n{res_str}"

    def plot(self, **kwargs):
        """Create a visualization of the connection results on the synodic section.

        This method generates a Poincare map showing the intersection points
        of both manifolds with the synodic section, highlighting discovered
        connections with color-coded Delta-V values.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            :func:`~hiten.utils.plots.plot_poincare_connections_map`.
            Common options include figure size, color maps, and styling parameters.

        Returns
        -------
        matplotlib figure or axes
            The plot object, which can be further customized or saved.

        Raises
        ------
        ValueError
            If :meth:`~hiten.algorithms.connections.base.Connection.solve` 
            has not been called yet (no cached data to plot).

        Notes
        -----
        The plot shows:
        - Source manifold intersection points (typically unstable manifold)
        - Target manifold intersection points (typically stable manifold)
        - Connection points with color-coded Delta-V requirements
        - Section coordinate labels and axes

        Examples
        --------
        >>> connection.solve(source, target)
        >>> fig = connection.plot(figsize=(10, 8), cmap='viridis')
        >>> fig.savefig('connections.png')

        See Also
        --------
        :func:`~hiten.utils.plots.plot_poincare_connections_map`
            Underlying plotting function with detailed parameter documentation.
        """
        # Use cached artifacts; user should call solve() first
        if self._last_source is None or self._last_target is None:
            raise EngineError("Nothing to plot: call solve(source, target) first.")
        from hiten.algorithms.connections.interfaces import \
            _ManifoldInterface  # internal
        src_if = _ManifoldInterface(manifold=self._last_source)
        tgt_if = _ManifoldInterface(manifold=self._last_target)

        # Build section hits for both manifolds on the configured synodic section
        sec_u = src_if.to_section(self.section, direction=self.direction)
        sec_s = tgt_if.to_section(self.section, direction=self.direction)

        pts_u = np.asarray(sec_u.points, dtype=float)
        pts_s = np.asarray(sec_s.points, dtype=float)
        labels = tuple(sec_u.labels)

        # Use cached results
        res_list = self._last_results or []

        if res_list:
            match_pts = np.asarray([r.point2d for r in res_list], dtype=float)
            match_vals = np.asarray([r.delta_v for r in res_list], dtype=float)
        else:
            match_pts = None
            match_vals = None

        return plot_poincare_connections_map(
            points_src=pts_u,
            points_tgt=pts_s,
            labels=labels,
            match_points=match_pts,
            match_values=match_vals,
            **kwargs,
        )
