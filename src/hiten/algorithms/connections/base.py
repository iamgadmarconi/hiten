"""Provide a user-facing interface for discovering connections between manifolds in CR3BP.

This module provides the main :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
class, which serves as a high-level facade for the connection discovery algorithm. It wraps 
the lower-level connection engine and provides convenient methods for solving connection 
problemsand visualizing results.

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

from typing import TYPE_CHECKING, Generic, Optional

import numpy as np

from hiten.algorithms.connections.types import (ConnectionDomainPayload,
                                                ConnectionResults)
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)
from hiten.algorithms.types.exceptions import EngineError
from hiten.utils.plots import (plot_heteroclinic_connection,
                               plot_poincare_connections_map)

if TYPE_CHECKING:
    from hiten.algorithms.connections.backends import _ConnectionsBackend
    from hiten.algorithms.connections.engine import _ConnectionEngine
    from hiten.algorithms.connections.options import ConnectionOptions
    from hiten.algorithms.connections.types import (Connections,
                                                    _ConnectionResult)

class ConnectionPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """Provide a user-facing facade for connection discovery and plotting in CR3BP.

    This class provides a high-level interface for discovering ballistic and
    impulsive transfers between manifolds in the Circular Restricted Three-Body
    Problem. It wraps the lower-level connection engine and provides convenient
    methods for solving connection problems and visualizing results.

    Parameters
    ----------
    config : :class:`~hiten.algorithms.connections.config.ConnectionConfig`
        Configuration object containing section, direction, and search parameters.
    interface : :class:`~hiten.algorithms.connections.interfaces._ManifoldConnectionInterface`
        Interface for translating between domain objects and backend inputs.
    engine : :class:`~hiten.algorithms.connections.engine._ConnectionEngine`, optional
        Engine instance to use for connection discovery. If None, must be set later
        or use with_default_engine() factory method.

    Examples
    --------

    >>> from hiten.algorithms.connections import ConnectionPipeline, SearchConfig
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
    >>> conn = ConnectionPipeline.with_default_engine(
    >>>     config=ConnectionConfig(
    >>>         section=section_cfg,
    >>>         direction=None,
    >>>         delta_v_tol=1,
    >>>         ballistic_tol=1e-8,
    >>>         eps2d=1e-3,
    >>>     )
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
    :class:`~hiten.algorithms.connections.types.Connections`
        Container for connection results with convenient access methods.
    """

    def __init__(self, config: ConfigT, engine: "_ConnectionEngine", interface: InterfaceT = None, backend: "_ConnectionsBackend" = None) -> None:
        super().__init__(config, engine, interface, backend)
        
        self._last_source = None
        self._last_target = None
        self._last_results: list["_ConnectionResult"] | None = None

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None, backend: Optional["_ConnectionsBackend"] = None) -> "ConnectionPipeline[DomainT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).

        The default engine uses :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`.

        Parameters
        ----------
        config : :class:`~hiten.algorithms.connections.config.ConnectionConfig`
            Configuration object containing section, direction, and search parameters.
        interface : :class:`~hiten.algorithms.connections.interfaces._ManifoldConnectionInterface`, optional
            Interface for translating between domain objects and backend inputs.
            If None, uses the default _ManifoldConnectionInterface.
        backend : :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`, optional
            Backend instance for the connection algorithm. If None, uses the default _ConnectionsBackend.
        Returns
        -------
        :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
            A connection facade instance with a default engine injected.
        """
        from hiten.algorithms.connections.backends import _ConnectionsBackend
        from hiten.algorithms.connections.engine import _ConnectionEngine
        from hiten.algorithms.connections.interfaces import \
            _ManifoldConnectionInterface
        backend = backend or _ConnectionsBackend()
        intf = interface or _ManifoldConnectionInterface()
        engine = _ConnectionEngine(backend=backend, interface=intf)
        return cls(config, engine, intf, backend)

    def solve(self, source: DomainT, target: DomainT, options: "ConnectionOptions") -> "Connections":
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
        options : :class:`~hiten.algorithms.connections.options.ConnectionOptions`
            Runtime options for the connection search.

        Returns
        -------
        :class:`~hiten.algorithms.connections.types.Connections`
            ConnectionPipeline results sorted by increasing Delta-V requirement.
            Each result contains transfer type, Delta-V, intersection points,
            and 6D states at the connection.

        Notes
        -----
        Results are cached internally for convenient access via the 
        :attr:`~hiten.algorithms.connections.base.ConnectionPipeline.results`
        property and for plotting with the
        :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.plot` method.

        The algorithm performs these steps:
        1. Convert manifolds to section interfaces
        2. Create connection problem specification
        3. Delegate to :class:`~hiten.algorithms.connections.engine._ConnectionEngine`
        4. Cache results for later use

        Examples
        --------
        >>> from hiten.algorithms.connections.options import ConnectionOptions
        >>> options = ConnectionOptions()
        >>> results = connection.solve(unstable_manifold, stable_manifold, options)
        >>> print(results)
        """
        domain_obj = (source, target)
        
        problem = self._create_problem(domain_obj=domain_obj, options=options)
        engine = self._get_engine()
        engine_result = engine.solve(problem)
        payload = ConnectionDomainPayload._from_mapping(
            {
                "connections": engine_result.connections,
                "source": source,
                "target": target,
            }
        )
        source.services.dynamics.apply_connections(payload)
        target.services.dynamics.apply_connections(payload)
        self._last_source = source
        self._last_target = target
        self._last_results = payload.connections
        return ConnectionResults(list(payload.connections))

    @property
    def results(self) -> "Connections":
        """Access the latest connection results with convenient formatting.

        Returns
        -------
        :class:`~hiten.algorithms.connections.types.Connections`
            A view over the latest results with friendly printing and
            convenient access methods. Returns an empty view if 
            :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.solve`
            has not been called yet.

        Notes
        -----
        This property provides access to cached results from the most recent
        call to :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.solve`. 
        The :class:`~hiten.algorithms.connections.types.Connections` 
        wrapper provides enhanced formatting and filtering capabilities.

        Examples
        --------
        >>> connection.solve(source, target)
        >>> print(connection.results)  # Pretty-printed summary
        >>> ballistic = connection.results.ballistic  # Filter by type
        """
        from hiten.algorithms.connections.types import Connections
        return Connections(self._last_results)

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
            If :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.solve` 
            has not been called yet (no cached data to plot).

        Notes
        -----
        The plot shows:
        - Source manifold intersection points (typically unstable manifold)
        - Target manifold intersection points (typically stable manifold)
        - ConnectionPipeline points with color-coded Delta-V requirements
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

        manifold_if = self._get_interface()
        config = self._get_config()

        # Apply direction correction via interface (handles time-reversal for stable manifolds)
        direction_u = manifold_if._apply_direction_correction(self._last_source, config.direction)
        direction_s = manifold_if._apply_direction_correction(self._last_target, config.direction)
        
        sec_u = manifold_if.to_section(manifold=self._last_source, config=config.section, direction=direction_u)
        sec_s = manifold_if.to_section(manifold=self._last_target, config=config.section, direction=direction_s)

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

    def plot_connection(self, index: int = 0, **kwargs):
        """Plot a specific heteroclinic connection showing the connecting trajectories.

        This method visualizes a single connection by plotting the portions of
        trajectories from both manifolds that lead to the connection point,
        including flow direction arrows and a Delta-V arrow at the connection.

        Parameters
        ----------
        index : int, default=0
            Index of the connection to plot (0 = best/lowest Delta-V connection).
        **kwargs
            Additional keyword arguments passed to
            :func:`~hiten.utils.plots.plot_heteroclinic_connection`.
            Common options include:
            
            - figsize : tuple, default (10, 8)
            - save : bool, default False
            - dark_mode : bool, default True
            - filepath : str, default 'heteroclinic_connection.svg'
            - src_color : str, default 'red'
            - tgt_color : str, default 'blue'
            - dv_arrow_scale : float, default 0.05
            - flow_arrow_spacing : int, default 10

        Returns
        -------
        tuple
            (fig, ax) containing the matplotlib figure and axis objects.

        Raises
        ------
        EngineError
            If :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.solve` 
            has not been called yet.
        IndexError
            If the specified index is out of range for the available connections.

        Notes
        -----
        The plot shows:
        
        - Trajectory portions from source manifold (unstable) in red
        - Trajectory portions from target manifold (stable) in blue
        - Flow direction arrows along both trajectories
        - Connection point marked with a yellow circle
        - Delta-V vector as a magenta arrow (if non-negligible)
        - Text annotation with connection type and Delta-V magnitude
        - Parent periodic orbits (naturally included in manifold trajectories)

        Examples
        --------
        >>> # Plot the best connection (index 0)
        >>> connection.solve(source, target)
        >>> connection.plot_connection()
        >>> 
        >>> # Plot a specific connection with custom styling
        >>> connection.plot_connection(
        ...     index=5,
        ...     figsize=(12, 10),
        ...     dark_mode=True,
        ...     src_color='cyan',
        ...     tgt_color='magenta'
        ... )

        See Also
        --------
        :func:`~hiten.utils.plots.plot_heteroclinic_connection`
            Underlying plotting function with detailed parameter documentation.
        :meth:`~hiten.algorithms.connections.base.ConnectionPipeline.plot`
            Plot all connections on a Poincare section.
        """
        # Check that we have cached results
        if self._last_source is None or self._last_target is None or self._last_results is None:
            raise EngineError("Nothing to plot: call solve(source, target) first.")

        # Check index bounds
        if not self._last_results:
            raise EngineError("No connections found to plot.")
        
        if index < 0 or index >= len(self._last_results):
            raise IndexError(
                f"Connection index {index} out of range. "
                f"Available connections: 0 to {len(self._last_results) - 1}"
            )

        # Get the specific connection result
        connection_result = self._last_results[index]

        # Extract trajectory data for plotting
        traj_data = self._extract_connection_trajectories(connection_result)
        
        # Call the plotting function with prepared data
        return plot_heteroclinic_connection(
            trajectory_data=traj_data,
            connection_result=connection_result,
            bodies=[self._last_source.system.primary, self._last_source.system.secondary],
            system_distance=self._last_source.system.distance,
            **kwargs
        )

    def _extract_connection_trajectories(self, connection_result: "_ConnectionResult") -> dict:
        """Extract trajectory data for a specific connection.

        This helper method finds the correct trajectory segments and connection
        point indices for visualization. It uses section timing information for
        accurate trajectory trimming.

        Parameters
        ----------
        connection_result : :class:`~hiten.algorithms.connections.types._ConnectionResult`
            The connection result to extract trajectory data from.

        Returns
        -------
        dict
            Dictionary containing:
            
            - 'states_u' : ndarray, shape (n, 6) - Source trajectory states up to connection
            - 'states_s' : ndarray, shape (m, 6) - Target trajectory states up to connection
            - 'state_u_conn' : ndarray, shape (6,) - Source state at connection
            - 'state_s_conn' : ndarray, shape (6,) - Target state at connection

        Notes
        -----
        This method uses section crossing times when available for robust trajectory
        trimming. Falls back to position-based search if timing info is unavailable.
        """
        # Extract trajectory indices from connection result
        traj_idx_u = connection_result.trajectory_index_u
        traj_idx_s = connection_result.trajectory_index_s

        # Get the specific trajectories
        traj_u = self._last_source.trajectories[traj_idx_u]
        traj_s = self._last_target.trajectories[traj_idx_s]

        # Get states at connection point
        state_u_conn = connection_result.state_u
        state_s_conn = connection_result.state_s

        # Find closest points in full trajectories to the interpolated connection states
        dist_u = np.linalg.norm(traj_u.states[:, :3] - state_u_conn[:3], axis=1)
        idx_u_conn = int(np.argmin(dist_u))
        
        dist_s = np.linalg.norm(traj_s.states[:, :3] - state_s_conn[:3], axis=1)
        idx_s_conn = int(np.argmin(dist_s))

        states_u = np.vstack([
            traj_u.states[:idx_u_conn + 1, :],
            state_u_conn
        ])
        states_s = np.vstack([
            traj_s.states[:idx_s_conn + 1, :],
            state_s_conn
        ])

        return {
            'states_u': states_u,
            'states_s': states_s,
            'state_u_conn': state_u_conn,
            'state_s_conn': state_s_conn,
        }

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.connections.config.ConnectionConfig`
            The configuration object to validate.
            
        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        super()._validate_config(config)
        
        if hasattr(config, 'section') and config.section is None:
            raise ValueError("Section configuration is required")