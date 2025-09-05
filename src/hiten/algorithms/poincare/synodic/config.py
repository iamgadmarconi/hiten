from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np

from hiten.algorithms.poincare.core.config import (_ReturnMapBaseConfig,
                                                   _SectionConfig)
from hiten.algorithms.poincare.synodic.events import _AffinePlaneEvent


@dataclass
class _SynodicMapConfig(_ReturnMapBaseConfig):

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
        # Synodic maps do not support computing on init because trajectories
        # must be supplied via from_orbit/from_manifold. Ignore any user value.
        self.compute_on_init = False


class _SynodicSectionConfig(_SectionConfig):
    """
    Synodic affine-plane section specification.

    This configuration declares the geometric section (hyperplane) used for
    crossings and the 2-D projection axes used to report points in the section
    plane. Numerical detection settings are configured separately.

    Parameters
    ----------
    normal : array_like, shape (6,)
        Hyperplane normal vector in synodic coordinates.
    offset : float
        Hyperplane offset so that the section is defined by ``n · state = offset``.
    plane_coords : tuple[str, str]
        Names of the 2-D axes used for reporting section points (default ("y","vy")).
    """

    def __init__(
        self,
        *,
        normal: Sequence[float] | np.ndarray,
        offset: float = 0.0,
        plane_coords: Tuple[str, str] = ("y", "vy"),
    ) -> None:

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
        """Return an :class:`AffinePlaneEvent` with this geometry and *direction*."""
        return _AffinePlaneEvent(normal=self.normal, offset=self.offset, direction=direction)


_SECTION_CACHE: dict[tuple[tuple[float, ...], float, tuple[str, str]], _SynodicSectionConfig] = {}


def _get_section_config(
    *,
    normal: Sequence[float] | np.ndarray,
    offset: float,
    plane_coords: Tuple[str, str],
) -> _SynodicSectionConfig:
    """Return a cached _SynodicSectionConfig for the given geometry/projection."""
    n_arr = np.asarray(normal, dtype=float)
    key = (tuple(n_arr.tolist()), float(offset), (str(plane_coords[0]), str(plane_coords[1])))
    if key not in _SECTION_CACHE:
        _SECTION_CACHE[key] = _SynodicSectionConfig(normal=n_arr, offset=offset, plane_coords=plane_coords)
    return _SECTION_CACHE[key]
