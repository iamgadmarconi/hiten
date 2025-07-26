from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _ReturnMapConfig
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _Section:
    """Lightweight immutable container for a single 2-D return-map slice."""

    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str]):
        self.points: np.ndarray = points       # (n, 2) plane coordinates
        self.states: np.ndarray = states       # (n, k) backend-specific state vectors
        self.labels: tuple[str, str] = labels  # axis labels (e.g. ("q2", "p2"))

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self):
        return f"_Section(points={len(self)}, labels={self.labels})"


class _ReturnMapBase(ABC):
    """Reference-frame-agnostic façade for discrete Poincaré maps.

    Concrete subclasses supply ONLY four pieces of information:

    1. `_build_backend(section_coord)` -> _ReturnMapBackend
    2. `_build_seeding_strategy(section_coord)` -> _SeedingStrategyBase
    3. `ic(pt, section_coord)` -> 6-D initial conditions in the problem frame
    4. *(optionally)* overrides for plotting or advanced projections.
    """

    def __init__(self, config: _ReturnMapConfig) -> None:
        self.config: _ReturnMapConfig = config

        # Run-time caches
        self._sections: dict[str, _Section] = {}
        self._engines: dict[str, "_ReturnMapEngine"] = {}
        self._section: Optional[_Section] = None  # most-recently accessed

        if self.config.compute_on_init:
            self.compute()

    @abstractmethod
    def _build_backend(self, section_coord: str) -> _ReturnMapBackend:
        """Return a backend capable of single-step propagation to *section_coord*."""

    @abstractmethod
    def _build_seeding_strategy(self, section_coord: str) -> _SeedingStrategyBase:
        """Return a seeding strategy suitable for *section_coord*."""

    def _build_engine(self, backend: _ReturnMapBackend, strategy: _SeedingStrategyBase) -> "_ReturnMapEngine":

        if _ReturnMapEngine.__abstractmethods__:
            raise TypeError("Sub-class must implement _build_engine to return a concrete _ReturnMapEngine")
        return _ReturnMapEngine(backend=backend, seed_strategy=strategy, map_config=self.config)

    def compute(self, *, section_coord: str | None = None):
        """Compute (or retrieve from cache) the return map on `section_coord`."""

        key: str = section_coord or self.config.section_coord

        # Fast path - already cached
        if key in self._sections:
            self._section = self._sections[key]
            return self._section.points

        # Lazy-build engine if needed
        if key not in self._engines:
            backend = self._build_backend(key)
            strategy = self._build_seeding_strategy(key)

            # Let the subclass decide which engine to use.
            self._engines[key] = self._build_engine(backend, strategy)

        # Delegate compute to engine
        self._section = self._engines[key].compute_section()
        self._sections[key] = self._section
        return self._section.points

    def get_section(self, section_coord: str) -> _Section:
        if section_coord not in self._sections:
            raise KeyError(
                f"Section '{section_coord}' has not been computed. "
                f"Available: {list(self._sections.keys())}"
            )
        return self._sections[section_coord]

    def list_sections(self) -> list[str]:
        return list(self._sections.keys())

    def has_section(self, section_coord: str) -> bool:
        return section_coord in self._sections

    def clear_cache(self):
        self._sections.clear()
        self._engines.clear()
        self._section = None

    def get_points(self, *, section_coord: str | None = None) -> np.ndarray:
        """Return the stored 2-D points for *section_coord* (compute on-demand)."""

        key = section_coord or self.config.section_coord

        if key not in self._sections:
            self.compute(section_coord=key)

        return self._sections[key].points

    def __len__(self):  
        return 0 if self._section is None else len(self._section)

    def __repr__(self):  
        return (
            f"{self.__class__.__name__}(sections={len(self._sections)}, "
            f"config={self.config})"
        )