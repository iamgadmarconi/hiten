"""Runtime options for connection discovery.

These classes define runtime tuning parameters that control HOW WELL the
connection discovery algorithm performs. They can vary between method calls
without changing the algorithm structure.

For compile-time configuration (problem structure), see config.py.

All distance and velocity tolerances are in nondimensional CR3BP rotating-frame units.
"""

from dataclasses import dataclass
from typing import Optional

from hiten.algorithms.types.options import _HitenBaseOptions


@dataclass(frozen=True)
class ConnectionOptions(_HitenBaseOptions):
    """Runtime options for connection discovery.
    
    These parameters tune HOW WELL the connection search performs and can vary
    between method calls without changing the problem structure.
    
    Parameters
    ----------
    delta_v_tol : float, default=1e-3
        Maximum Delta-V tolerance for accepting a connection, in nondimensional
        CR3BP velocity units. Connections with ||Delta-V|| > delta_v_tol are
        rejected. This is a runtime parameter because it tunes the acceptance
        criteria without changing WHAT connections are being searched for.
    ballistic_tol : float, default=1e-8
        Threshold for classifying connections as ballistic vs impulsive, in
        nondimensional CR3BP velocity units. Connections with ||Delta-V|| <=
        ballistic_tol are classified as "ballistic", others as "impulsive".
        This is a runtime parameter because it only affects classification,
        not the algorithm structure.
    eps2d : float, default=1e-4
        Radius for initial 2D pairing of points on the synodic section, in
        nondimensional CR3BP distance units. Points closer than this distance
        in the section plane are considered potential connection candidates.
        This is a runtime parameter because it tunes the geometric search
        radius without changing the fundamental algorithm.
    n_workers : int or None, default=1
        Number of parallel workers for computation. Runtime tuning parameter
        for resource allocation.

    Notes
    -----
    These are runtime options because they control HOW WELL the search performs
    (acceptance criteria, classification thresholds, search radius, parallelism)
    rather than WHAT problem is being solved (which section, which direction).

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
    >>> # Default runtime options
    >>> options = ConnectionOptions()
    >>> 
    >>> # Loose search for preliminary analysis
    >>> loose_options = ConnectionOptions(
    ...     delta_v_tol=1e-2,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-3
    ... )
    >>> 
    >>> # Tight search for high-precision connections
    >>> tight_options = ConnectionOptions(
    ...     delta_v_tol=1e-5,
    ...     ballistic_tol=1e-8,
    ...     eps2d=1e-5
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.connections.config.ConnectionConfig`
        Compile-time configuration for problem structure.
    :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
        Main class that uses these options.
    """
    _version: float = 1.0

    delta_v_tol: float = 1e-3
    ballistic_tol: float = 1e-8
    eps2d: float = 1e-4
    n_workers: Optional[int] = 1

    def _validate(self) -> None:
        """Validate the options."""
        if self.delta_v_tol <= 0:
            raise ValueError("delta_v_tol must be positive.")
        if self.ballistic_tol <= 0:
            raise ValueError("ballistic_tol must be positive.")
        if self.eps2d <= 0:
            raise ValueError("eps2d must be positive.")
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError("n_workers must be positive or None.")

