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

import numpy as np

from hiten.algorithms.poincare.core.config import (_ReturnMapBaseConfig,
                                                   _SectionConfig)
from hiten.algorithms.poincare.synodic.events import _AffinePlaneEvent


@dataclass(frozen=True)
class _SynodicMapConfig(_ReturnMapBaseConfig):
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

    # Detection/runtime knobs
    interp_kind: Literal["linear", "cubic"] = "linear"
    segment_refine: int = 0
    tol_on_surface: float = 1e-12
    dedup_time_tol: float = 1e-9
    dedup_point_tol: float = 1e-12
    max_hits_per_traj: int | None = None
    newton_max_iter: int = 4

    def __post_init__(self) -> None:
        """Post-initialization processing for synodic map configuration.

        Notes
        -----
        This method automatically sets `compute_on_init = False` since
        synodic maps require precomputed trajectories to be supplied
        explicitly via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_orbit`, 
        :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_manifold`, 
        or :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_trajectories`.
        The user-provided value for `compute_on_init` is ignored.
        """
        # Synodic maps do not support computing on init because trajectories
        # must be supplied via from_orbit/from_manifold. Ignore any user value.
        # For frozen dataclasses, use object.__setattr__ for post-init fixes.
        object.__setattr__(self, "compute_on_init", False)


class _SynodicSectionConfig(_SectionConfig):
    """Synodic affine-plane section specification.

    This configuration class defines the geometric section (hyperplane) used for
    crossings and the 2D projection axes used to report points in the section
    plane. It extends the base section configuration with specialized validation
    and event building for synodic Poincare sections.

    Parameters
    ----------
    normal : array_like, shape (6,)
        Hyperplane normal vector in synodic coordinates (nondimensional units).
        Must be a 1D array of length 6 with finite values.
    offset : float, default 0.0
        Hyperplane offset so that the section is defined by n * state = offset
        (nondimensional units).
    plane_coords : tuple[str, str], default ("y", "vy")
        Names of the 2D axes used for reporting section points.
        Must be a tuple of two coordinate labels.

    Attributes
    ----------
    normal : ndarray, shape (6,)
        The hyperplane normal vector (normalized).
    offset : float
        The hyperplane offset value.
    plane_coords : tuple[str, str]
        The 2D projection coordinate labels.

    Notes
    -----
    This configuration class provides the geometric specification for synodic
    Poincare sections. The section is defined as a hyperplane in the 6D
    state space of the circular restricted three-body problem.

    The class performs validation to ensure:
    - Normal vector has correct dimensions (6D)
    - All values are finite
    - Plane coordinates are properly specified

    All geometric parameters are in nondimensional units unless otherwise
    specified.
    """

    def __init__(
        self,
        *,
        normal: Sequence[float] | np.ndarray,
        offset: float = 0.0,
        plane_coords: Tuple[str, str] = ("y", "vy"),
    ) -> None:
        """Initialize the synodic section configuration.

        Parameters
        ----------
        normal : array_like, shape (6,)
            Hyperplane normal vector in synodic coordinates.
            Must be a 1D array of length 6 with finite values.
        offset : float, default 0.0
            Hyperplane offset value (nondimensional units).
        plane_coords : tuple[str, str], default ("y", "vy")
            Names of the 2D axes for section point projection.
            Must be a tuple of two coordinate labels.

        Raises
        ------
        ValueError
            If normal vector has incorrect dimensions or contains non-finite values.
        ValueError
            If plane_coords is not a tuple of two strings.

        Notes
        -----
        This constructor initializes the section configuration with validation
        of all geometric parameters. The normal vector is converted to a
        numpy array and validated for correct dimensions and finite values.

        The section is defined by the equation n * state = offset, where
        n is the normal vector and state is the 6D state vector.
        """

        n_arr = np.asarray(normal, dtype=float)
        if n_arr.ndim != 1 or n_arr.size != 6:
            raise ValueError("normal must be a 1-D array of length 6")
        if not np.all(np.isfinite(n_arr)):
            raise ValueError("normal must contain only finite values")
        self.normal: np.ndarray = n_arr
        self.offset: float = float(offset)

        if not (isinstance(plane_coords, tuple) and len(plane_coords) == 2):
            raise ValueError("plane_coords must be a tuple of two axis names")
        self.plane_coords: Tuple[str, str] = (str(plane_coords[0]), str(plane_coords[1]))

    def build_event(self, *, direction: Literal[1, -1, None] = None) -> _AffinePlaneEvent:
        """Build an affine plane event from this section configuration.

        Parameters
        ----------
        direction : {1, -1, None}, optional
            Crossing direction filter. If None, no direction filtering
            is applied. If 1 or -1, only crossings in the specified
            direction are detected.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.events._AffinePlaneEvent`
            An affine plane event configured with this section's geometry
            and the specified crossing direction.

        Notes
        -----
        This method creates an affine plane event that can be used for
        crossing detection. The event is configured with the section's
        normal vector, offset, and optional crossing direction.

        The event can be used by the detection backend to identify
        trajectory-section intersections during Poincare section computation.
        """
        return _AffinePlaneEvent(normal=self.normal, offset=self.offset, direction=direction)


_SECTION_CACHE: dict[tuple[tuple[float, ...], float, tuple[str, str]], _SynodicSectionConfig] = {}
"""Cache for section configurations to avoid duplicate objects.

This module-level cache stores section configuration objects keyed by
their geometry parameters (normal vector, offset, plane coordinates).
It provides efficient lookup and reuse of identical configurations.
"""


def _get_section_config(
    *,
    normal: Sequence[float] | np.ndarray,
    offset: float,
    plane_coords: Tuple[str, str],
) -> _SynodicSectionConfig:
    """Get a cached section configuration for the given geometry and projection.

    Parameters
    ----------
    normal : array_like, shape (6,)
        Hyperplane normal vector in synodic coordinates.
    offset : float
        Hyperplane offset value (nondimensional units).
    plane_coords : tuple[str, str]
        Names of the 2D axes for section point projection.

    Returns
    -------
    :class:`~hiten.algorithms.poincare.synodic.config._SynodicSectionConfig`
        A cached section configuration object with the specified geometry.

    Notes
    -----
    This function provides a caching mechanism for section configurations
    to avoid creating duplicate objects with identical geometry. The cache
    key is based on the normal vector, offset, and plane coordinates.

    The function automatically creates a new configuration if one with
    the specified geometry doesn't exist in the cache, and caches it
    for future use.

    All geometric parameters are in nondimensional units unless otherwise
    specified.
    """
    n_arr = np.asarray(normal, dtype=float)
    key = (tuple(n_arr.tolist()), float(offset), (str(plane_coords[0]), str(plane_coords[1])))
    if key not in _SECTION_CACHE:
        _SECTION_CACHE[key] = _SynodicSectionConfig(normal=n_arr, offset=offset, plane_coords=plane_coords)
    return _SECTION_CACHE[key]
