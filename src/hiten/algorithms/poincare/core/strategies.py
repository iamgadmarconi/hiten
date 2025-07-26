from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from hiten.algorithms.poincare.core.config import _SectionConfig

class _SeedingStrategyBase(ABC):

    _cached_limits: dict[tuple[float, int], list[float]] = {}
    
    def __init__(self, config: _SectionConfig) -> None:
        self._cfg = config

    @property
    def config(self) -> "_SectionConfig":
        return self._cfg

    @property
    def plane_coords(self) -> Tuple[str, str]:
        return self._cfg.plane_coords
    
    @abstractmethod
    def generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any, find_turning_fn: Any) -> List[Tuple[float, float, float, float]]:
        pass

    def __call__(self, **kwargs):
        return self.generate(**kwargs)
