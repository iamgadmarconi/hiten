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

from dataclasses import dataclass
from typing import Literal

from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.synodic.base import SynodicMap
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.utils.exceptions import EngineError
from hiten.system.manifold import Manifold


@dataclass
class _ManifoldInterface:
    """Provide an interface for accessing manifold data in connection discovery.

    This class provides a clean interface for extracting synodic section
    intersections from manifolds. It handles the conversion between manifold
    trajectory data and the section intersection data needed for connection
    analysis.

    Parameters
    ----------
    manifold : :class:`~hiten.system.manifold.Manifold`
        The manifold object containing computed trajectory data.

    Attributes
    ----------
    manifold : :class:`~hiten.system.manifold.Manifold`
        The wrapped manifold object.

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
    >>> interface = _ManifoldInterface(manifold=computed_manifold)
    >>> section_cfg = _SynodicMapConfig(x=0.8)
    >>> section = interface.to_section(config=section_cfg, direction=1)
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
    manifold: Manifold

    def to_section(
        self,
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

        if self.manifold.manifold_result is None:
            raise EngineError("Manifold must be computed before extracting section hits")

        cfg = config or _SynodicMapConfig()
        syn = SynodicMap(cfg)
        return syn.from_manifold(self.manifold, direction=direction)

