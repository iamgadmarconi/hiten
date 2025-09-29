"""Provide configuration classes for connection discovery parameters in CR3BP.

This module provides configuration classes that control the behavior of the
connection discovery algorithm. These classes define tolerances, search parameters,
and computational settings used when finding ballistic and impulsive transfers
between manifolds.

All distance and velocity tolerances are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.connections.base`
    Main ConnectionPipeline class that uses these configuration objects.
:mod:`~hiten.algorithms.connections.engine`
    ConnectionPipeline engine that applies these parameters during computation.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig


@dataclass(frozen=True)
class _SearchConfig:
    """Define search parameters and tolerances for connection discovery.

    This class defines the tolerances and geometric parameters used during
    the connection discovery process. It controls which candidate connections
    are accepted and how they are classified.

    Parameters
    ----------
    delta_v_tol : float, default 1e-3
        Maximum Delta-V tolerance for accepting a connection, in nondimensional
        CR3BP velocity units. Connections with ||Delta-V|| > delta_v_tol are rejected.
    ballistic_tol : float, default 1e-8
        Threshold for classifying connections as ballistic vs impulsive, in
        nondimensional CR3BP velocity units. Connections with ||Delta-V|| <= ballistic_tol
        are classified as "ballistic", others as "impulsive".
    eps2d : float, default 1e-4
        Radius for initial 2D pairing of points on the synodic section, in
        nondimensional CR3BP distance units. Points closer than this distance
        in the section plane are considered potential connection candidates.

    Notes
    -----
    The search process uses a multi-stage filtering approach:
    1. Initial 2D geometric pairing using `eps2d`
    2. Mutual-nearest-neighbor filtering
    3. Geometric refinement using local segments
    4. Final Delta-V computation and filtering using `delta_v_tol`
    5. Classification using `ballistic_tol`

    Typical values:
    - For loose searches: delta_v_tol=1e-2, eps2d=1e-3
    - For precise searches: delta_v_tol=1e-4, eps2d=1e-5
    - For ballistic-only: delta_v_tol=ballistic_tol=1e-8

    Examples
    --------
    >>> # Default configuration
    >>> config = _SearchConfig()
    >>> 
    >>> # Loose search for preliminary analysis
    >>> loose_config = _SearchConfig(
    ...     delta_v_tol=1e-2,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-3
    ... )
    >>> 
    >>> # Tight search for high-precision connections
    >>> tight_config = _SearchConfig(
    ...     delta_v_tol=1e-5,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-5
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.connections.config._ConnectionConfig`
        Extended configuration including computational parameters.
    :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
        Main class that uses this configuration.
    """
    delta_v_tol: float = 1e-3
    ballistic_tol: float = 1e-8
    eps2d: float = 1e-4


@dataclass(frozen=True)
class _ConnectionConfig(_SearchConfig):
    """Define configuration for connection discovery including section and search parameters.

    This class combines the synodic section configuration with search parameters
    to provide a complete configuration for connection discovery between manifolds.

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
    >>> from hiten.algorithms.connections.config import _ConnectionConfig, _SearchConfig
    >>> from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
    >>> 
    >>> section_cfg = _SynodicMapConfig(section_axis="x", section_offset=0.8)
    >>> search_cfg = _SearchConfig(delta_v_tol=1e-3, ballistic_tol=1e-8, eps2d=1e-4)
    >>> 
    >>> config = _ConnectionConfig(
    ...     section=section_cfg,
    ...     direction=1,
    ...     search_cfg=search_cfg
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.connections.config._SearchConfig`
        Search parameters and tolerances.
    :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        Synodic section configuration.
    :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
        Main class that uses this configuration.
    """
    section: _SynodicMapConfig = _SynodicMapConfig()
    direction: Optional[Literal[1, -1]] = None
    n_workers: Optional[int] = 1

