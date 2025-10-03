"""Configuration classes for connection discovery (compile-time structure).

This module provides configuration classes that define the algorithm structure
for connection discovery between manifolds. These parameters define WHAT
problem is being solved and should be set once when creating a pipeline.

For runtime tuning parameters (tolerances, search radii), see options.py.

All distance and velocity tolerances are in nondimensional CR3BP rotating-frame units.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
from hiten.algorithms.types.configs import _HitenBaseConfig


@dataclass(frozen=True)
class ConnectionConfig(_HitenBaseConfig):
    """Configuration for connection discovery (compile-time structure).

    This dataclass encapsulates compile-time configuration parameters that
    define the problem structure for connection discovery. These parameters
    define WHAT connections to look for, not HOW WELL to find them.

    Parameters
    ----------
    section : SynodicMapConfig
        Configuration for the synodic section where manifolds are intersected.
        This defines the geometric structure of the problem - which section
        to use for finding connections.
    direction : {1, -1, None}, default=None
        Direction for section crossings to consider. This defines the problem
        structure by filtering which crossings are candidates:
        
        - 1: Only positive crossings (increasing coordinate)
        - -1: Only negative crossings (decreasing coordinate)
        - None: Both directions (default)

    Notes
    -----
    For runtime tuning parameters like `delta_v_tol`, `ballistic_tol`, `eps2d`,
    and `n_workers`, use ConnectionOptions instead.

    The `section` parameter is compile-time because it fundamentally defines
    WHAT geometric structure you're using to find connections. The `direction`
    parameter is compile-time because it defines WHICH physical crossings are
    candidates for connections.

    Examples
    --------
    >>> from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
    >>> 
    >>> # Compile-time: Define problem structure
    >>> section_cfg = SynodicMapConfig(
    ...     section_axis="x",
    ...     section_offset=0.8,
    ...     plane_coords=("y", "z")
    ... )
    >>> config = ConnectionConfig(
    ...     section=section_cfg,
    ...     direction=1  # Only positive crossings
    ... )
    >>> 
    >>> # Runtime: Set search tolerances
    >>> from hiten.algorithms.connections.options import ConnectionOptions
    >>> options = ConnectionOptions(
    ...     delta_v_tol=1e-3,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-4
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
        Synodic section configuration.
    :class:`~hiten.algorithms.connections.options.ConnectionOptions`
        Runtime tuning parameters for connection search.
    :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
        Main class that uses this configuration.
    """
    section: SynodicMapConfig = SynodicMapConfig()
    direction: Optional[Literal[1, -1]] = None

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.direction is not None and self.direction not in [1, -1]:
            raise ValueError(
                f"Invalid direction: {self.direction}. Must be 1, -1, or None."
            )
