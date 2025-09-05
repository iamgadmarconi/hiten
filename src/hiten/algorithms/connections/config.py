from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, Optional


@dataclass
class _SectionUseConfig:
    """Geometry and projection for the working Poincaré section.

    mode selects which map family is used to generate sections.
    Either an axis (with offset) or a 6-D normal can be specified.
    """

    mode: Literal["synodic", "center_manifold"] = "synodic"
    plane_coords: Tuple[str, str] = ("y", "vy")
    direction: Literal[1, -1, None] = None
    axis: str | int | None = "x"
    normal: Sequence[float] | None = None
    offset: Optional[float] = 0.0

    def __post_init__(self):
        # If a 6D normal is not provided but an axis is, infer a canonical one-hot normal
        if self.normal is None and (self.axis is not None):
            idx_map = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5}
            try:
                if isinstance(self.axis, str):
                    k = idx_map.get(self.axis.lower(), None)
                else:
                    k = int(self.axis)
                if k is not None and 0 <= int(k) <= 5:
                    n = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    n[int(k)] = 1.0
                    self.normal = tuple(n)
            except Exception:
                # Do not fail; leave normal as None
                pass

@dataclass
class _SearchConfig:
    """Planner/search configuration that generates/filters candidates."""

    strategy: Literal["grid", "graph", "ga"] = "grid"
    budget: int = 500
    neighborhood_radius: float = 1e-3
    seed_grid: int = 100
    top_k: int = 5
    use_segments: bool = True
    segment_step_source: int = 1
    segment_step_target: int = 1

class ConnectionConfig(_SectionUseConfig, _SearchConfig):
    n_workers: int = 1
