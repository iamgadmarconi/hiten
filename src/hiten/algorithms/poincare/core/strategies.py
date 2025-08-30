from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from hiten.algorithms.poincare.core.config import (_SeedingConfigLike,
                                                   _SectionConfig)


class _SeedingStrategyBase(ABC):

    _cached_limits: dict[tuple[float, int], list[float]] = {}
    
    def __init__(self, section_cfg: _SectionConfig, map_cfg: _SeedingConfigLike) -> None:
        self._section_cfg = section_cfg
        self._map_cfg = map_cfg

    @property
    def config(self) -> "_SectionConfig":
        return self._section_cfg

    @property
    def map_config(self) -> "_SeedingConfigLike":
        return self._map_cfg

    @property
    def n_seeds(self) -> int:
        return self._map_cfg.n_seeds
    
    @property
    def plane_coords(self) -> Tuple[str, str]:
        return self._section_cfg.plane_coords
    
    @abstractmethod
    def generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any, find_turning_fn: Any) -> List[Tuple[float, float, float, float]]:
        pass

    def __call__(self, **kwargs):
        return self.generate(**kwargs)
