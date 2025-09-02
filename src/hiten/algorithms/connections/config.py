from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple


@dataclass
class _ConnectionEngineConfig:
    """Engine-level execution settings.

    - n_workers: degree of parallelism
    - cache: enable memoization of intermediate artifacts
    - random_seed: seed for stochastic planners (e.g., GA)
    - log_level: optional logging verbosity hint
    """

    n_workers: int = 1
    cache: bool = True
    random_seed: Optional[int] = None
    log_level: Optional[str] = None


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
    offset: float = 0.0
    offset_sweep: Sequence[float] | None = None


@dataclass
class _BallisticConfig:
    """Settings for ballistic (no-impulse) connections."""

    same_energy_required: bool = True
    tol_intersection: float = 1e-8
    tol_refine: float = 1e-8
    max_candidates: int = 256
    newton_max_iters: int = 5
    newton_tol: float = 1e-8
    tau_min: float = 1e-4
    tau_max: float | None = None  # default to arc_time if None


@dataclass
class _ImpulsiveConfig:
    """Settings for impulsive connections (single or two impulses)."""

    impulse_model: Literal["one", "two"] = "one"
    max_total_dv: float = 0.0
    dv_weight: float = 1.0
    tof_weight: float = 0.0
    coast_time_bounds: Tuple[float, float] | None = None


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


