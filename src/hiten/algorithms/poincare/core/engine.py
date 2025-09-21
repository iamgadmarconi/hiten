"""Abstract base class for Poincare return map engines.

This module provides the abstract base class for implementing Poincare
return map engines in the hiten framework. Engines coordinate backends
and seeding strategies to compute complete return maps.

The main class :class:`~hiten.algorithms.poincare.core.engine._ReturnMapEngine` 
defines the interface that all concrete engines must implement, including 
the core `solve` method and common functionality for caching and configuration.

The engine layer sits between the high-level return map interface
and the low-level numerical integration, providing a clean separation
of concerns and enabling different computational strategies.
"""

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _EngineConfigLike
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase

if TYPE_CHECKING:
    from hiten.algorithms.poincare.core.base import _Section

class _ReturnMapEngine(ABC):
    """Abstract base class for Poincare return map engines.

    This class defines the interface that all concrete return map
    engines must implement. It coordinates backends and seeding
    strategies to compute complete return maps efficiently.

    Parameters
    ----------
    backend : :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
        The backend for numerical integration and section crossing
        detection.
    seed_strategy : :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
        The seeding strategy for generating initial conditions
        on the section plane.
    map_config : :class:`~hiten.algorithms.poincare.core.config._EngineConfigLike`
        Configuration object containing engine parameters such as
        iteration count, time step, and worker count.

    Attributes
    ----------
    _backend : :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
        The numerical integration backend.
    _strategy : :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
        The seeding strategy for initial conditions.
    _map_config : :class:`~hiten.algorithms.poincare.core.config._EngineConfigLike`
        The engine configuration.
    _n_iter : int
        Number of return map iterations to compute.
    _dt : float
        Integration time step (nondimensional units).
    _n_workers : int
        Number of parallel workers for computation.
    _section_cache : :class:`~hiten.algorithms.poincare.core.base._Section` or None
        Cache for the computed section to avoid redundant computation.

    Notes
    -----
    The engine coordinates the computation process by:
    1. Using the seeding strategy to generate initial conditions
    2. Using the backend to integrate trajectories and find section crossings
    3. Iterating the process to build up the complete return map
    4. Managing caching and parallel computation for efficiency

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self, backend: _ReturnMapBackend, 
                 seed_strategy: _SeedingStrategyBase,
                 map_config: _EngineConfigLike) -> None:
        self._backend = backend
        self._strategy = seed_strategy
        self._map_config = map_config
        self._n_iter = int(self._map_config.n_iter)
        self._dt = float(self._map_config.dt)
        # Use configuration value for workers, falling back to CPU count
        self._n_workers = self._map_config.n_workers or os.cpu_count() or 1

        self._section_cache: "_Section" | None = None

    @abstractmethod
    def solve(self) -> "_Section":
        """Compute and return the section (or Results that inherit _Section)."""
        raise NotImplementedError

    def clear_cache(self):
        """Clear the cached section data.

        Notes
        -----
        This method clears the internal section cache, forcing
        recomputation on the next call to compute_section. Use
        this method to free memory or force fresh computation
        with updated parameters.
        """
        self._section_cache = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_iter={self._n_iter}, dt={self._dt}, n_workers={self._n_workers})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n_iter={self._n_iter}, dt={self._dt}, n_workers={self._n_workers})"