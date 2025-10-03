"""Provide interface classes for manifold data access in connection discovery.

This module provides interface classes that abstract manifold data access
for the connection discovery system. These interfaces handle the conversion
between manifold representations and the synodic section intersections
needed for connection analysis.

The interfaces serve as adapters between the manifold system and the
connection discovery algorithms, providing a clean separation of concerns
and enabling flexible data access patterns.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.system.manifold`
    Manifold classes that these interfaces wrap.
:mod:`~hiten.algorithms.poincare.synodic.base`
    Synodic map functionality used for section intersections.
:mod:`~hiten.algorithms.connections.engine`
    ConnectionPipeline engine that uses these interfaces.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from hiten.algorithms.connections.config import ConnectionConfig
from hiten.algorithms.connections.options import ConnectionOptions
from hiten.algorithms.connections.types import (
    ConnectionDomainPayload,
    ConnectionResults,
    ConnectionsBackendRequest,
    ConnectionsBackendResponse,
    _ConnectionProblem,
)
from hiten.algorithms.poincare.core.types import _Section
from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
from hiten.algorithms.types.core import _HitenBaseInterface
from hiten.algorithms.types.exceptions import EngineError
from hiten.system.maps.synodic import SynodicMap

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold


class _ManifoldConnectionInterface(
    _HitenBaseInterface[
        ConnectionConfig,
        _ConnectionProblem,
        ConnectionResults,
        ConnectionsBackendResponse,
    ]
):
    """Provide an interface for accessing manifold data in connection discovery.

    This class provides a clean interface for extracting synodic section
    intersections from manifolds. It handles the conversion between manifold
    trajectory data and the section intersection data needed for connection
    analysis.

    Notes
    -----
    This interface serves as an adapter between the manifold system and
    the connection discovery algorithms. It encapsulates the logic for:
    
    - Validating that manifold data is available
    - Converting manifold trajectories to synodic section intersections
    - Handling different crossing direction filters with time-reversal correction
    - Providing appropriate error messages for invalid states

    The interface ensures that manifolds are properly computed before
    attempting to extract section data, preventing runtime errors in
    the connection discovery process.

    Examples
    --------
    >>> from hiten.system.manifold import Manifold
    >>> from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
    >>> 
    >>> # Assuming manifold is computed
    >>> interface = _ManifoldConnectionInterface()
    >>> section_cfg = SynodicMapConfig(x=0.8)
    >>> section = interface.to_section(manifold=computed_manifold, config=section_cfg, direction=1)
    >>> print(f"Found {len(section.points)} intersection points")

    See Also
    --------
    :class:`~hiten.system.manifold.Manifold`
        Manifold class that this interface wraps.
    :class:`~hiten.algorithms.poincare.synodic.base.SynodicMap`
        Synodic map used for computing section intersections.
    :class:`~hiten.algorithms.connections.engine._ConnectionProblem`
        Problem specification that uses these interfaces.
    """

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: tuple["Manifold", "Manifold"],
        config: ConnectionConfig,
        options: ConnectionOptions,
    ) -> _ConnectionProblem:
        """Create a connection problem specification.
        
        Parameters
        ----------
        domain_obj : tuple of :class:`~hiten.system.manifold.Manifold`
            The source and target manifolds.
        config : :class:`~hiten.algorithms.connections.config.ConnectionConfig`
            The compile-time configuration (section structure, direction).
        options : :class:`~hiten.algorithms.connections.options.ConnectionOptions`, optional
            Runtime options (tolerances, search parameters). If None, defaults are used.

        Returns
        -------
        :class:`~hiten.algorithms.connections.types._ConnectionProblem`
            The connection problem combining config and options.
        """
        source, target = domain_obj
        
        return _ConnectionProblem(
            source=source,
            target=target,
            section_axis=config.section.section_axis,
            section_offset=config.section.section_offset,
            plane_coords=config.section.plane_coords,
            direction=config.direction,
            delta_v_tol=options.delta_v_tol,
            ballistic_tol=options.ballistic_tol,
            eps2d=options.eps2d,
            n_workers=options.n_workers,
        )

    def to_backend_inputs(self, problem: _ConnectionProblem) -> tuple:
        """Convert problem to backend inputs.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.connections.types._ConnectionProblem`
            The problem to convert to backend inputs.

        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend inputs.
            
        Notes
        -----
        Handles direction parameter carefully: stable manifolds are integrated
        backward in time, so we flip the crossing direction to ensure source
        and target manifolds use compatible crossing directions in physical space.
        """
        # Determine crossing directions for each manifold
        # Apply direction correction based on each object's time integration direction
        # Stable manifolds are integrated backward in time and need direction flipping
        # Unstable manifolds are integrated forward in time (no flip needed)
        direction_u = self._apply_direction_correction(problem.source, problem.direction)
        direction_s = self._apply_direction_correction(problem.target, problem.direction)
        
        # Create section configs with appropriate directions
        section_config_u = SynodicMapConfig(
            section_axis=problem.section_axis,
            section_offset=problem.section_offset,
            plane_coords=problem.plane_coords,
            direction=direction_u
        )
        
        section_config_s = SynodicMapConfig(
            section_axis=problem.section_axis,
            section_offset=problem.section_offset,
            plane_coords=problem.plane_coords,
            direction=direction_s
        )
        
        # Extract section data from both manifolds with appropriate directions
        pu, Xu, traj_indices_u = self.to_numeric(problem.source, section_config_u, direction=direction_u)
        ps, Xs, traj_indices_s = self.to_numeric(problem.target, section_config_s, direction=direction_s)
        
        # Extract search parameters from the problem
        eps = float(problem.eps2d)
        dv_tol = float(problem.delta_v_tol)
        bal_tol = float(problem.ballistic_tol)
        
        request = ConnectionsBackendRequest(
            points_u=pu,
            points_s=ps,
            states_u=Xu,
            states_s=Xs,
            traj_indices_u=traj_indices_u,
            traj_indices_s=traj_indices_s,
            eps=eps,
            dv_tol=dv_tol,
            bal_tol=bal_tol,
        )
        from hiten.algorithms.types.core import _BackendCall
        return _BackendCall(request=request)

    def to_domain(self, outputs: ConnectionsBackendResponse, *, problem: _ConnectionProblem) -> ConnectionDomainPayload:
        return ConnectionDomainPayload._from_mapping(
            {
                "connections": tuple(outputs.results),
                "source": problem.source,
                "target": problem.target,
                "metadata": dict(outputs.metadata),
            }
        )

    def to_results(self, outputs: ConnectionsBackendResponse, *, problem: _ConnectionProblem, domain_payload=None) -> ConnectionResults:
        payload = domain_payload or self.to_domain(outputs, problem=problem)
        return ConnectionResults(list(payload.connections))

    def to_section(
        self,
        manifold: "Manifold",
        config: SynodicMapConfig,
        *,
        direction: Literal[1, -1, None] | None = None,
    ) -> _Section:
        """Extract synodic section intersection data from the manifold.

        This method computes the intersections between the manifold trajectories
        and a specified synodic section, returning the intersection points,
        states, and timing information needed for connection analysis.

        Parameters
        ----------
        manifold : :class:`~hiten.system.manifold.Manifold`
            The manifold object containing computed trajectory data.
        config : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`, optional
            Configuration for the synodic section geometry and detection settings.
            Includes section axis, offset, coordinate system, interpolation method,
            and numerical tolerances. If not provided, default settings are used.
        direction : {1, -1, None}, optional
            Filter for section crossing direction. 1 selects positive crossings
            (increasing coordinate), -1 selects negative crossings (decreasing
            coordinate), None accepts both directions (default: None).

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.types._Section`
            Section object containing intersection data with attributes:
            
            - points : 2D coordinates on the section plane
            - states : 6D phase space states at intersections  
            - times : intersection times along trajectories
            - labels : coordinate labels for the section plane

        Raises
        ------
        :class:`~hiten.algorithms.types.exceptions.EngineError`
            If the manifold has not been computed (manifold.result is None).
            Call manifold.compute() before using this method.

        Notes
        -----
        This method delegates to :class:`~hiten.system.maps.synodic.SynodicMap`
        for the actual intersection computation. The synodic map handles:
        
        - Trajectory interpolation and root finding
        - Section crossing detection and refinement
        - Coordinate transformation to section plane
        - Deduplication of nearby intersection points
        
        The resulting section data is suitable for geometric analysis in
        the connection discovery algorithms.

        Examples
        --------
        >>> # Basic usage with default section
        >>> section = interface.to_section(manifold)
        >>> 
        >>> # Custom section at x = 0.8 with positive crossings only
        >>> from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
        >>> config = SynodicMapConfig(
        ...     section_axis="x",
        ...     section_offset=0.8,
        ...     plane_coords=("y", "z")
        ... )
        >>> section = interface.to_section(manifold, config=config, direction=1)
        >>> print(f"Points: {section.points.shape}")
        >>> print(f"States: {section.states.shape}")

        See Also
        --------
        :class:`~hiten.system.maps.synodic.SynodicMap`
            Underlying synodic map implementation.
        :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
            Configuration class for section parameters.
        :meth:`~hiten.system.manifold.Manifold.compute`
            Method to compute manifold data before section extraction.
        """

        if manifold.result is None:
            raise EngineError("Manifold must be computed before extracting section hits")

        # Create synodic map from the manifold
        synodic_map = SynodicMap(manifold)
    
        section_axis = config.section_axis
        section_offset = config.section_offset
        plane_coords = config.plane_coords
    
        # Compute the section (runtime options will use SynodicMap's defaults)
        synodic_map.compute(
            section_axis=section_axis,
            section_offset=section_offset,
            plane_coords=plane_coords,
            direction=direction,
        )
        
        # Fetch the most recently computed section directly to avoid reliance on key format
        return synodic_map.dynamics.get_section()

    def to_numeric(self, manifold: "Manifold", config: SynodicMapConfig, *, direction: Literal[1, -1, None] = None):
        """Return (points2d, states6d, trajectory_indices) arrays for this manifold on a section.

        Parameters
        ----------
        manifold : :class:`~hiten.system.manifold.Manifold`
            The manifold object containing computed trajectory data.
        config : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`, optional
            Configuration for the synodic section geometry and detection settings.
        direction : {1, -1, None}, optional
            Filter for section crossing direction. 1 selects positive crossings
            (increasing coordinate), -1 selects negative crossings (decreasing
            coordinate), None accepts both directions (default: None).

        Returns
        -------
        tuple
            A tuple containing (points2d, states6d, trajectory_indices) where:
            - points2d : ndarray, shape (n, 2)
                2D coordinates on the section plane
            - states6d : ndarray, shape (n, 6) 
                6D phase space states at intersections
            - trajectory_indices : ndarray, shape (n,) or None
                Indices of trajectories that produced each intersection point
        """
        sec = self.to_section(manifold=manifold, config=config, direction=direction)
        trajectory_indices = getattr(sec, 'trajectory_indices', None)
        if trajectory_indices is not None:
            trajectory_indices = np.asarray(trajectory_indices, dtype=int)
        return (np.asarray(sec.points, dtype=float), np.asarray(sec.states, dtype=float), trajectory_indices)

    @staticmethod
    def _apply_direction_correction(domain_obj, direction: Literal[1, -1, None]) -> Literal[1, -1, None]:
        """Apply direction correction based on object's time integration direction.

        This method accounts for the fact that stable manifolds are integrated
        backward in time while unstable manifolds are integrated forward in time.
        When filtering section crossings by direction, we need to flip the direction
        for backward-time integrations to get compatible physical crossings.

        Parameters
        ----------
        domain_obj : object
            Domain object (e.g., Manifold) that may have time direction info.
        direction : {1, -1, None}
            The requested crossing direction filter.

        Returns
        -------
        {1, -1, None}
            Corrected direction for this specific object, accounting for its
            time integration direction. Returns None if no direction filter.

        Notes
        -----
        For Manifold objects:
        - Stable manifolds (stable=1): integrated backward → flip direction
        - Unstable manifolds (stable=-1): integrated forward → no flip

        For other objects without time direction info: no correction applied.

        This translation logic properly belongs in the interface layer, as it
        translates domain object properties (stable/unstable) into backend
        parameters (crossing direction).

        Examples
        --------
        >>> # Stable manifold with direction=-1
        >>> corrected = _ManifoldConnectionInterface._apply_direction_correction(stable_manifold, -1)
        >>> # Returns +1 (flipped for backward time integration)
        >>> 
        >>> # Unstable manifold with direction=-1
        >>> corrected = _ManifoldConnectionInterface._apply_direction_correction(unstable_manifold, -1)
        >>> # Returns -1 (no flip for forward time integration)
        """
        if direction is None:
            return None

        # Check if object has manifold-like time direction info
        if hasattr(domain_obj, 'stable'):
            # stable = 1 means backward time integration (flip direction)
            # stable = -1 means forward time integration (no flip)
            is_backward_time = (domain_obj.stable == 1)
            if is_backward_time:
                return -direction
        
        # Default: no correction for objects without time direction info
        return direction
