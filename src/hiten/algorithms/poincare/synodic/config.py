"""Configuration classes for synodic Poincare sections (compile-time structure).

This module provides configuration classes for synodic Poincare section
geometry and structure. These define WHAT section to use, not HOW WELL to
detect crossings.

For runtime tuning parameters (tolerances, refinement settings), see options.py.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

from hiten.algorithms.poincare.core.config import _ReturnMapConfig
from hiten.algorithms.types.configs import RefineConfig


@dataclass(frozen=True)
class SynodicMapConfig(_ReturnMapConfig):
    """Configuration for synodic Poincare map geometry (compile-time structure).

    This configuration class defines the geometric structure of the synodic
    Poincare section. These parameters define WHAT section to use, not HOW
    WELL to detect crossings on it.

    Parameters
    ----------
    section_axis : str or int or None, default "x"
        Axis for section definition (ignored if section_normal provided).
        Can be a string ("x", "y", "z", "vx", "vy", "vz") or integer index.
        This is compile-time because it defines WHAT geometric structure to use.
    section_offset : float, default 0.0
        Offset for the section hyperplane (nondimensional units).
        This is compile-time because it defines WHERE the section is located.
    section_normal : sequence of float or None, optional
        Explicit normal vector for section definition (length 6).
        If provided, overrides section_axis. Must be in synodic coordinates.
        This is compile-time because it defines WHAT geometric structure to use.
    plane_coords : tuple[str, str], default ("y", "vy")
        Coordinate labels for 2D projection of section points.
        This is compile-time because it defines WHAT coordinates to visualize.
    direction : {1, -1, None}, default=None
        Crossing direction filter. This is compile-time because it defines
        WHICH physical crossings are candidates.
    interp_kind : {'linear', 'cubic'}, default='cubic'
        Interpolation method for section crossing refinement (inherited from RefineConfig).
        This is compile-time because it defines WHAT interpolation algorithm to use.

    Notes
    -----
    For runtime tuning parameters like `segment_refine`, tolerances, and 
    `n_workers`, use SynodicMapOptions instead.

    The section geometry is compile-time because it fundamentally defines
    WHAT problem you're solving. The detection/refinement parameters are
    runtime because they tune HOW WELL you find crossings.

    Examples
    --------
    >>> # Compile-time: Define section geometry and method
    >>> from hiten.algorithms.types.configs import RefineConfig
    >>> config = SynodicMapConfig(
    ...     section_axis="x",
    ...     section_offset=0.8,
    ...     plane_coords=("y", "vy"),
    ...     direction=1,
    ...     interp_kind="cubic"
    ... )
    >>> 
    >>> # Runtime: Tune detection quality
    >>> from hiten.algorithms.poincare.synodic.options import SynodicMapOptions
    >>> from hiten.algorithms.types.options import RefineOptions
    >>> options = SynodicMapOptions(
    ...     refine=RefineOptions(segment_refine=10)
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
        Runtime tuning parameters for detection and refinement.
    :class:`~hiten.algorithms.poincare.synodic.base.SynodicMapPipeline`
        Main class that uses this configuration.
    :class:`~hiten.algorithms.types.configs.RefineConfig`
        Base configuration for refinement method selection.
    """
    section_axis: str | int | None = "x"
    section_offset: float = 0.0
    section_normal: Sequence[float] | None = None
    plane_coords: Tuple[str, str] = ("y", "vy")
    direction: Optional[Literal[1, -1]] = None
    interp_kind: RefineConfig = RefineConfig(interp_kind="cubic")

    def _validate(self) -> None:
        """Validate the configuration."""
        if len(self.plane_coords) != 2:
            raise ValueError("plane_coords must have exactly 2 elements")
        if self.direction is not None and self.direction not in [1, -1]:
            raise ValueError("direction must be 1, -1, or None")
