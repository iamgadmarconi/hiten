from abc import ABC
from dataclasses import dataclass
from typing import Literal, Protocol, Tuple, runtime_checkable


@dataclass
class _ReturnMapBaseConfig(ABC):
    """Minimal map configuration shared by all return-map flavors.

    Contains only orchestration fields that are universally applicable.
    """
    compute_on_init: bool = False
    n_workers: int | None = None


@dataclass
class _IntegrationConfig(ABC):
    """Integration-related knobs used by propagating backends/engines."""
    dt: float = 1e-2
    method: Literal["scipy", "rk", "symplectic", "adaptive"] = "rk"
    order: int = 4
    c_omega_heuristic: float = 20.0


@dataclass
class _IterationConfig(ABC):
    """Iteration control for return-map steps."""
    n_iter: int = 40


@dataclass
class _SeedingConfig(ABC):
    """Seeding control used by strategies that generate initial seeds."""
    n_seeds: int = 20


# Backward-compatible umbrella config combining all mixins
@dataclass
class _ReturnMapConfig(_ReturnMapBaseConfig, _IntegrationConfig, _IterationConfig, _SeedingConfig):
    pass


class _SectionConfig(ABC):

    section_coord: str
    plane_coords: Tuple[str, str]


@runtime_checkable
class _EngineConfigLike(Protocol):
    dt: float
    n_iter: int
    n_workers: int | None


@runtime_checkable
class _SeedingConfigLike(Protocol):
    n_seeds: int
