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


class _ReturnMapBase(ABC):
    """Abstract base class for Poincare return map implementations.

    This class provides a reference-frame-agnostic facade for discrete
    Poincare maps. It handles caching, section management, and provides
    a unified interface for different types of return maps.

    Concrete subclasses must implement four key methods:

    1. :meth:`~hiten.algorithms.poincare.core.base._ReturnMapBase._build_backend` 
        -> :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
    2. :meth:`~hiten.algorithms.poincare.core.base._ReturnMapBase._build_seeding_strategy` 
        -> :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
    3. :meth:`~hiten.algorithms.poincare.core.base._ReturnMapBase.ic` 
        -> 6D initial conditions in the problem frame
    4. *(optionally)* overrides for plotting or advanced projections

    Parameters
    ----------
    config : :class:`~hiten.algorithms.poincare.core.config._ReturnMapBaseConfig`
        Configuration object containing section parameters and settings.

    Attributes
    ----------
    config : :class:`~hiten.algorithms.poincare.core.config._ReturnMapBaseConfig`
        The configuration object.
    _sections : dict[str, :class:`~hiten.algorithms.poincare.core.base._Section`]
        Cache of computed sections, keyed by section coordinate.
    _engines : dict[str, :class:`~hiten.algorithms.poincare.core.engine._ReturnMapEngine`]
        Cache of engines, keyed by section coordinate.
    _section : :class:`~hiten.algorithms.poincare.core.base._Section` or None
        The most recently accessed section.

    Notes
    -----
    The class provides automatic caching of computed sections and engines
    to avoid redundant computation. Sections are computed on-demand and
    cached for subsequent access. The framework supports multiple section
    coordinates and provides projection capabilities for different views
    of the return map data.

    All time units are in nondimensional units unless otherwise specified.
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
        """Build a backend for the specified section coordinate.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier (e.g., "q2", "p2").

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
            A backend capable of single-step propagation to the section.

        Notes
        -----
        This method must be implemented by concrete subclasses to provide
        the appropriate backend for the given section coordinate. The
        backend handles the numerical integration and section crossing
        detection.
        """

    @abstractmethod
    def _build_seeding_strategy(self, section_coord: str) -> _SeedingStrategyBase:
        """Build a seeding strategy for the specified section coordinate.

        Parameters
        ----------
        section_coord : str
            The section coordinate identifier (e.g., "q2", "p2").

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
            A seeding strategy suitable for the section coordinate.

        Notes
        -----
        This method must be implemented by concrete subclasses to provide
        the appropriate seeding strategy for generating initial conditions
        on the section plane.
        """

    def _build_engine(self, backend: _ReturnMapBackend, strategy: _SeedingStrategyBase) -> "_ReturnMapEngine":
        """Build an engine from a backend and seeding strategy.

        Parameters
        ----------
        backend : :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
            The backend for numerical integration.
        strategy : :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
            The seeding strategy for generating initial conditions.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.engine._ReturnMapEngine`
            A concrete engine instance.

        Raises
        ------
        TypeError
            If the engine class is still abstract and must be implemented
            by a subclass.

        Notes
        -----
        This method creates an engine that coordinates the backend and
        seeding strategy to compute the return map. If the engine class
        is abstract, subclasses must override this method to provide
        a concrete implementation.
        """
        if _ReturnMapEngine.__abstractmethods__:
            raise TypeError("Sub-class must implement _build_engine to return a concrete _ReturnMapEngine")
        return _ReturnMapEngine(backend=backend, seed_strategy=strategy, map_config=self.config)

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
        self._section = self._engines[key].solve()
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
        self._engines.clear()
        self._section = None

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