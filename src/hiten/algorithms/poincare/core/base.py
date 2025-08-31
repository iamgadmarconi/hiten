from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _ReturnMapBaseConfig
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _Section:
    """Lightweight immutable container for a single 2-D return-map slice."""

    def __init__(self, points: np.ndarray, states: np.ndarray, labels: tuple[str, str], times: np.ndarray | None = None):
        self.points: np.ndarray = points       # (n, 2) plane coordinates
        self.states: np.ndarray = states       # (n, k) backend-specific state vectors
        self.labels: tuple[str, str] = labels  # axis labels (e.g. ("q2", "p2"))
        self.times: np.ndarray | None = times  # (n,) absolute integration times (optional)

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self):
        return f"_Section(points={len(self)}, labels={self.labels}, times={'yes' if self.times is not None else 'no'})"


class _ReturnMapBase(ABC):
    """Reference-frame-agnostic façade for discrete Poincaré maps.

    Concrete subclasses supply ONLY four pieces of information:

    1. `_build_backend(section_coord)` -> _ReturnMapBackend
    2. `_build_seeding_strategy(section_coord)` -> _SeedingStrategyBase
    3. `ic(pt, section_coord)` -> 6-D initial conditions in the problem frame
    4. *(optionally)* overrides for plotting or advanced projections.
    """

    def __init__(self, config: _ReturnMapBaseConfig) -> None:
        self.config: _ReturnMapBaseConfig = config

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

    def _axis_index(self, section: "_Section", axis: str) -> int:
        """Return the column index corresponding to *axis*.

        The default implementation assumes a 1-1 mapping between the
        ``section.labels`` tuple and columns of ``section.points``.  Concrete
        subclasses can override this method if their mapping differs or if
        axis-based projection is not supported.
        """
        try:
            return section.labels.index(axis)
        except ValueError as exc:
            raise ValueError(
                f"Axis '{axis}' not available; valid labels are {section.labels}"
            ) from exc

    def get_points(
        self,
        *,
        section_coord: str | None = None,
        axes: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Return cached points for *section_coord* (compute on-demand).

        Parameters
        ----------
        section_coord
            Which stored section to retrieve (default ``self.config.section_coord``).
        axes
            Optional tuple of two axis labels (e.g. ("q3", "p2")) requesting a
            different 2-D projection of the stored state.  If *axes* is not
            given the raw stored projection is returned.
        """

        key = section_coord or self.config.section_coord

        if key not in self._sections:
            self.compute(section_coord=key)

        sec = self._sections[key]

        if axes is None:
            return sec.points

        idx1 = self._axis_index(sec, axes[0])
        idx2 = self._axis_index(sec, axes[1])

        return sec.points[:, (idx1, idx2)]

    def __len__(self):
        return 0 if self._section is None else len(self._section)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sections={len(self._sections)}, "
            f"config={self.config})"
        )