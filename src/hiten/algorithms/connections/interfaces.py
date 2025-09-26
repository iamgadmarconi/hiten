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
    Connection engine that uses these interfaces.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from hiten.algorithms.connections.types import (ConnectionResults,
                                                _ConnectionProblem)
from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.synodic.base import SynodicMap
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.types.core import _HitenBaseInterface
from hiten.algorithms.types.exceptions import EngineError

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold


class _ManifoldInterface(
    _HitenBaseInterface[
        _SynodicMapConfig,
        _ConnectionProblem,
        ConnectionResults,
        list,
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
    - Handling different crossing direction filters
    - Providing appropriate error messages for invalid states

    The interface ensures that manifolds are properly computed before
    attempting to extract section data, preventing runtime errors in
    the connection discovery process.

    Examples
    --------
    >>> from hiten.system.manifold import Manifold
    >>> from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
    >>> 
    >>> # Assuming manifold is computed
    >>> interface = _ManifoldInterface()
    >>> section_cfg = _SynodicMapConfig(x=0.8)
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
        """Initialize the manifold interface."""
        super().__init__()

    def create_problem(
        self,
        *,
        source: "Manifold",
        target: "Manifold",
        section: _SynodicMapConfig,
        direction: Literal[1, -1, None] | None = None,
        search: dict | None = None,
    ) -> _ConnectionProblem:
        """Create a connection problem specification."""
        return _ConnectionProblem(
            source=source,
            target=target,
            section=section,
            direction=direction,
            search=search,
        )

    def to_backend_inputs(self, problem: _ConnectionProblem) -> tuple:
        """Convert problem to backend inputs."""
        # Extract section data from both manifolds
        pu, Xu = self.to_numeric(problem.source, problem.section, direction=problem.direction)
        ps, Xs = self.to_numeric(problem.target, problem.section, direction=problem.direction)
        
        # Extract search parameters
        eps = float(getattr(problem.search, "eps2d", 1e-4)) if problem.search else 1e-4
        dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
        bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8
        
        from hiten.algorithms.types.core import _BackendCall
        return _BackendCall(
            args=(pu, ps, Xu, Xs),
            kwargs={"eps": eps, "dv_tol": dv_tol, "bal_tol": bal_tol}
        )

    def to_results(self, outputs: list, *, problem: _ConnectionProblem) -> ConnectionResults:
        """Convert backend outputs to connection results."""
        return ConnectionResults(outputs)

    def to_section(
        self,
        manifold: "Manifold",
        config: _SynodicMapConfig | None = None,
        *,
        direction: Literal[1, -1, None] | None = None,
    ) -> _Section:
        """Extract synodic section intersection data from the manifold.

        This method computes the intersections between the manifold trajectories
        and a specified synodic section, returning the intersection points,
        states, and timing information needed for connection analysis.

        Parameters
        ----------
        config : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`, optional
            Configuration for the synodic section geometry and detection settings.
            Includes section axis, offset, coordinate system, interpolation method,
            and numerical tolerances. If not provided, default settings are used.
        direction : {1, -1, None}, optional
            Filter for section crossing direction. 1 selects positive crossings
            (increasing coordinate), -1 selects negative crossings (decreasing
            coordinate), None accepts both directions (default: None).

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            Section object containing intersection data with attributes:
            
            - points : 2D coordinates on the section plane
            - states : 6D phase space states at intersections  
            - times : intersection times along trajectories
            - labels : coordinate labels for the section plane

        Raises
        ------
        ValueError
            If the manifold has not been computed (manifold_result is None).
            Call manifold.compute() before using this method.

        Notes
        -----
        This method delegates to :class:`~hiten.algorithms.poincare.synodic.base.SynodicMap`
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
        >>> section = interface.to_section()
        >>> 
        >>> # Custom section at x = 0.8 with positive crossings only
        >>> from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
        >>> config = _SynodicMapConfig(
        ...     section_axis="x",
        ...     section_offset=0.8,
        ...     plane_coords=("y", "z")
        ... )
        >>> section = interface.to_section(config=config, direction=1)
        >>> print(f"Points: {section.points.shape}")
        >>> print(f"States: {section.states.shape}")

        See Also
        --------
        :class:`~hiten.algorithms.poincare.synodic.base.SynodicMap`
            Underlying synodic map implementation.
        :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
            Configuration class for section parameters.
        :meth:`~hiten.system.manifold.Manifold.compute`
            Method to compute manifold data before section extraction.
        """

        if manifold.manifold_result is None:
            raise EngineError("Manifold must be computed before extracting section hits")

        cfg = config or _SynodicMapConfig()
        syn = SynodicMap(cfg)
        return syn.from_manifold(manifold, direction=direction)

    def to_numeric(self, manifold: "Manifold", config: _SynodicMapConfig | None = None, *, direction: Literal[1, -1, None] | None = None):
        """Return (points2d, states6d) arrays for this manifold on a section.

        Parameters
        ----------
        manifold : :class:`~hiten.system.manifold.Manifold`
            The manifold object containing computed trajectory data.
        config : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`, optional
            Configuration for the synodic section geometry and detection settings.
        """
        sec = self.to_section(manifold=manifold, config=config, direction=direction)
        return (np.asarray(sec.points, dtype=float), np.asarray(sec.states, dtype=float))
