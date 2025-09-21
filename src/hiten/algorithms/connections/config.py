"""Provide configuration classes for connection discovery parameters in CR3BP.

This module provides configuration classes that control the behavior of the
connection discovery algorithm. These classes define tolerances, search parameters,
and computational settings used when finding ballistic and impulsive transfers
between manifolds.

All distance and velocity tolerances are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.connections.base`
    Main Connection class that uses these configuration objects.
:mod:`~hiten.algorithms.connections.engine`
    Connection engine that applies these parameters during computation.
"""

from dataclasses import dataclass


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
    :class:`~hiten.algorithms.connections.config.ConnectionConfig`
        Extended configuration including computational parameters.
    :class:`~hiten.algorithms.connections.base.Connection`
        Main class that uses this configuration.
    """

    # Accept if ||Delta-V|| <= delta_v_tol
    delta_v_tol: float = 1e-3
    # Classify ballistic if ||Delta-V|| <= ballistic_tol
    ballistic_tol: float = 1e-8
    # Pairing radius on the section plane
    eps2d: float = 1e-4

@dataclass(frozen=True)
class ConnectionConfig(_SearchConfig):
    """Define an extended configuration including computational parameters.

    This class extends :class:`~hiten.algorithms.connections.config._SearchConfig` with additional parameters
    for controlling the computational aspects of connection discovery,
    such as parallel processing.

    Parameters
    ----------
    n_workers : int, default 1
        Number of worker processes to use for parallel computation.
        Set to 1 for serial processing, or a higher value to enable
        parallel processing of manifold intersections and connection searches.
        
    **kwargs
        All parameters from :class:`~hiten.algorithms.connections.config._SearchConfig` are also available:
        delta_v_tol, ballistic_tol, eps2d.

    Notes
    -----
    Parallel processing can significantly speed up connection discovery
    for large manifolds, but may not be beneficial for small problems
    due to overhead. The optimal number of workers depends on the system
    and problem size.

    Examples
    --------
    >>> # Serial processing with custom tolerances
    >>> config = ConnectionConfig(
    ...     delta_v_tol=1e-3,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-4,
    ...     n_workers=1
    ... )
    >>> 
    >>> # Parallel processing for large problems
    >>> parallel_config = ConnectionConfig(
    ...     delta_v_tol=1e-3,
    ...     n_workers=4
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.connections.config._SearchConfig`
        Base class with search and tolerance parameters.
    :class:`~hiten.algorithms.connections.base.Connection`
        Main class that uses this configuration.
    """
    n_workers: int = 1
