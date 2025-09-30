"""Configuration classes for synodic Poincare sections.

This module provides configuration classes for synodic Poincare section
detection and refinement. It defines the parameters needed for section
geometry, detection algorithms, and numerical settings.

The implementation supports both explicit normal vector specification
and axis-based section definitions, with comprehensive numerical
settings for detection and refinement algorithms.
"""

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

from hiten.algorithms.poincare.core.config import (_RefineConfig,
                                                   _ReturnMapBaseConfig)


@dataclass(frozen=True)
class _SynodicMapConfig(_ReturnMapBaseConfig, _RefineConfig):
    """Configuration for synodic Poincare map detection and refinement.

    This configuration class extends the base return map configuration
    with specialized parameters for synodic Poincare section detection
    on precomputed trajectories. It includes both geometric parameters
    for section definition and numerical parameters for detection algorithms.

    Parameters
    ----------
    section_axis : str or int or None, default "x"
        Axis for section definition (ignored if section_normal provided).
        Can be a string ("x", "y", "z", "vx", "vy", "vz") or integer index.
    section_offset : float, default 0.0
        Offset for the section hyperplane (nondimensional units).
    section_normal : sequence of float or None, optional
        Explicit normal vector for section definition (length 6).
        If provided, overrides section_axis. Must be in synodic coordinates.
    plane_coords : tuple[str, str], default ("y", "vy")
        Coordinate labels for 2D projection of section points.

    Detection Parameters
    -------------------
    interp_kind : {"linear", "cubic"}, default "linear"
        Interpolation method for crossing refinement.
        "cubic" provides higher accuracy but requires more computation.
    segment_refine : int, default 0
        Number of refinement segments for dense crossing detection.
        Higher values detect more crossings but increase computation.
    tol_on_surface : float, default 1e-12
        Tolerance for considering a point to be on the surface.
    dedup_time_tol : float, default 1e-9
        Time tolerance for deduplicating nearby crossings.
    dedup_point_tol : float, default 1e-12
        Point tolerance for deduplicating nearby crossings.
    max_hits_per_traj : int or None, default None
        Maximum number of hits per trajectory (None for unlimited).
    newton_max_iter : int, default 4
        Maximum Newton iterations for root refinement.

    Notes
    -----
    This configuration class provides comprehensive control over synodic
    Poincare section detection. The geometric parameters define the
    section hyperplane, while the detection parameters control the
    numerical algorithms used for crossing detection and refinement.

    The class automatically sets `compute_on_init = False` since synodic
    maps require precomputed trajectories to be supplied explicitly via
    :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_orbit`, 
    :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_manifold`, 
    or :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_trajectories`.

    All tolerances and offsets are in nondimensional units unless
    otherwise specified.
    """

    section_axis: str | int | None = "x"  # ignored if section_normal provided
    section_offset: float = 0.0
    section_normal: Sequence[float] | None = None  # length-6; overrides section_axis
    plane_coords: Tuple[str, str] = ("y", "vy")
    direction: Literal[1, -1, None] | None = None
