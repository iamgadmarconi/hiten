"""Base classes for Poincare return map implementations.

This module provides the foundational classes for implementing Poincare
return maps in the hiten framework. It defines the core interfaces and
data structures used across all Poincare map implementations.

This module provides the foundational classes for implementing Poincare
return maps in the hiten framework. It defines the core interfaces and
data structures used across all Poincare map implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _ReturnMapBaseConfig
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase
from hiten.algorithms.poincare.core.types import _Section


class _PoincareMapBase(ABC):
    """Common scaffolding for Poincare map facades."""

    def __init__(self, config: _ReturnMapBaseConfig) -> None:
        self.config: _ReturnMapBaseConfig = config
        self._sections: dict[str, _Section] = {}
        self._section: Optional[_Section] = None

    @abstractmethod
    def _get_or_compute_section(self, key: str) -> _Section:
        """Return the cached section for ``key``, computing it if necessary."""
        raise NotImplementedError

    def compute(self, *, section_coord: str | None = None):
        """Compute or retrieve the return map for the specified section.

        Parameters
        ----------
        section_coord : str, optional
            The section coordinate to compute. If None, uses the default
            section coordinate from the configuration.

        Returns
        -------
        ndarray, shape (n, 2)
            Array of 2D points in the section plane.

        Notes
        -----
        This method implements a caching strategy to avoid redundant
        computation. If the section has already been computed, it returns
        the cached result. Otherwise, it builds the necessary backend
        and engine, computes the section, and caches the result.

        The method handles lazy initialization of engines and provides
        a unified interface for section computation across different
        return map implementations.
        """
        key: str = section_coord or self.config.section_coord

        self._section = self._get_or_compute_section(key)
        self._sections[key] = self._section
        return self._section.points

    def get_section(self, section_coord: str) -> _Section:
        """Get a computed section by coordinate.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed section data.

        Raises
        ------
        KeyError
            If the section has not been computed.

        Notes
        -----
        This method returns the full section data including points,
        states, labels, and times. Use this method when you need
        access to the complete section information.
        """
        if section_coord not in self._sections:
            raise KeyError(
                f"Section '{section_coord}' has not been computed. "
                f"Available: {list(self._sections.keys())}"
            )
        return self._sections[section_coord]

    def list_sections(self) -> list[str]:
        """List all computed section coordinates.

        Returns
        -------
        list[str]
            List of section coordinate identifiers that have been computed.

        Notes
        -----
        This method returns the keys of the internal section cache,
        indicating which sections are available for access.
        """
        return list(self._sections.keys())

    def has_section(self, section_coord: str) -> bool:
        """Check if a section has been computed.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier to check.

        Returns
        -------
        bool
            True if the section has been computed, False otherwise.

        Notes
        -----
        This method provides a safe way to check section availability
        before attempting to access it.
        """
        return section_coord in self._sections

    def clear_cache(self):
        """Clear all cached sections and engines.

        Notes
        -----
        This method clears the internal caches for sections and engines,
        forcing recomputation on the next access. Use this method to
        free memory or force fresh computation with updated parameters.
        """
        self._sections.clear()
        self._section = None
        if hasattr(self, "_engines"):
            getattr(self, "_engines").clear()

    def _axis_index(self, section: "_Section", axis: str) -> int:
        """Return the column index corresponding to an axis label.

        Parameters
        ----------
        section : :class:`~hiten.algorithms.poincare.core.base._Section`
            The section containing the axis labels.
        axis : str
            The axis label to find.

        Returns
        -------
        int
            The column index of the axis in the section points.

        Raises
        ------
        ValueError
            If the axis label is not found in the section labels.

        Notes
        -----
        The default implementation assumes a 1-1 mapping between the
        section.labels tuple and columns of section.points. Concrete
        subclasses can override this method if their mapping differs
        or if axis-based projection is not supported.
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
        """Return cached points for a section with optional axis projection.

        Parameters
        ----------
        section_coord : str, optional
            Which stored section to retrieve. If None, uses the default
            section coordinate from the configuration.
        axes : tuple[str, str], optional
            Optional tuple of two axis labels (e.g., ("q3", "p2")) requesting
            a different 2D projection of the stored state. If None, returns
            the raw stored projection.

        Returns
        -------
        ndarray, shape (n, 2)
            Array of 2D points in the section plane, either the raw points
            or a projection onto the specified axes.

        Notes
        -----
        This method provides access to the computed section points with
        optional axis projection. If the section hasn't been computed,
        it triggers computation automatically. The axis projection allows
        viewing the section data from different coordinate perspectives.
        """
        key = section_coord or self.config.section_coord

        if key not in self._sections:
            self.compute(section_coord=key)

        sec = self._get_or_compute_section(key)
        self._sections[key] = sec

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


class _ReturnMapBase(_PoincareMapBase):
    """Propagation-based Poincare map base class."""

    def __init__(self, config: _ReturnMapBaseConfig) -> None:
        super().__init__(config)
        self._engines: dict[str, "_ReturnMapEngine"] = {}

        if self.config.compute_on_init:
            self.compute()

    @abstractmethod
    def _build_backend(self, section_coord: str) -> _ReturnMapBackend:
        ...

    @abstractmethod
    def _build_seeding_strategy(self, section_coord: str) -> _SeedingStrategyBase:
        ...

    def _build_engine(self, backend: _ReturnMapBackend, strategy: _SeedingStrategyBase) -> "_ReturnMapEngine":
        if _ReturnMapEngine.__abstractmethods__:
            raise TypeError("Sub-class must implement _build_engine to return a concrete _ReturnMapEngine")
        return _ReturnMapEngine(backend=backend, seed_strategy=strategy, map_config=self.config)

    def _get_or_compute_section(self, key: str) -> _Section:
        if key not in self._sections:
            if key not in self._engines:
                backend = self._build_backend(key)
                strategy = self._build_seeding_strategy(key)
                self._engines[key] = self._build_engine(backend, strategy)
            self._sections[key] = self._engines[key].solve()
        return self._sections[key]


class _DetectionMapBase(_PoincareMapBase):
    """Base class for Poincare maps that rely on detection-only backends.

    Detection facades do not propagate seeds via a predictor/corrector backend;
    instead they accept precomputed trajectories (or equivalent data) and run a
    detection engine. This base provides the same caching and projection helpers
    as :class:`_ReturnMapBase` without requiring subclasses to implement the
    propagation-specific hooks.
    """

    def __init__(self, config: _ReturnMapBaseConfig) -> None:
        # Detection maps do not support `compute_on_init` because they need
        # explicit trajectory input. Guard against accidental use.
        if getattr(config, "compute_on_init", False):
            raise NotImplementedError(
                "Detection-only maps do not support compute_on_init; call the "
                "facade-specific API explicitly."
            )
        super().__init__(config)

    def compute(self, *args, **kwargs):  # pragma: no cover - explicit guard
        raise NotImplementedError(
            "Detection-only maps do not implement compute(); use the facade-specific "
            "method (e.g., from_trajectories) to trigger detection."
        )

    def _get_or_compute_section(self, key: str) -> _Section:
        if key not in self._sections:
            raise KeyError(
                f"Section '{key}' has not been computed. "
                f"Available: {list(self._sections.keys())}"
            )
        return self._sections[key]