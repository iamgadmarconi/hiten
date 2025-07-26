import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _ReturnMapConfig
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase

if TYPE_CHECKING:
    from hiten.algorithms.poincare.core.base import _Section

class _ReturnMapEngine(ABC):
    

    def __init__(self, backend: _ReturnMapBackend, 
                 seed_strategy: _SeedingStrategyBase,
                 map_config: _ReturnMapConfig,
                n_workers: int | None = None) -> None:
        self._backend = backend
        self._strategy = seed_strategy
        self._map_config = map_config
        self._n_iter = int(self._map_config.n_iter)
        self._dt = float(self._map_config.dt)
        self._n_workers = n_workers or os.cpu_count() or 1

        self._section_cache: "_Section" | None = None

    @abstractmethod
    def compute_section(self, *, recompute: bool = False) -> "_Section":
        pass

    def clear_cache(self):
        self._section_cache = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_iter={self._n_iter}, dt={self._dt}, n_workers={self._n_workers})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n_iter={self._n_iter}, dt={self._dt}, n_workers={self._n_workers})"