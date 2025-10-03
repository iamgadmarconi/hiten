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
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.config import _ReturnMapConfig
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase
from hiten.algorithms.types.core import (OutputsT, ProblemT, ResultT,
                                         _HitenBaseEngine)

if TYPE_CHECKING:
    from hiten.algorithms.poincare.core.types import _Section


class _ReturnMapEngine(_HitenBaseEngine[ProblemT, ResultT, OutputsT], Generic[ProblemT, ResultT, OutputsT]):
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
    map_config : :class:`~hiten.algorithms.poincare.core.config._ReturnMapConfig`
        Configuration object containing engine parameters such as
        iteration count, time step, and worker count.

    Attributes
    ----------
    _backend : :class:`~hiten.algorithms.poincare.core.backend._ReturnMapBackend`
        The numerical integration backend.
    _strategy : :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
        The seeding strategy for initial conditions.
    _map_config : :class:`~hiten.algorithms.poincare.core.config._ReturnMapConfig`
        The engine configuration.
    _n_iter : int
        Number of return map iterations to compute.
    _dt : float
        Integration time step (nondimensional units).
    _n_workers : int
        Number of parallel workers for computation.
    _section_cache : :class:`~hiten.algorithms.poincare.core.types._Section` or None
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

    def __init__(
        self,
        *,
        backend: _ReturnMapBackend,
        seed_strategy: _SeedingStrategyBase,
        map_config: _ReturnMapConfig,
        interface=None,
    ) -> None:
        super().__init__(backend=backend, interface=interface)
        self._strategy = seed_strategy
        self._map_config = map_config
        # NOTE: Runtime params (n_iter, dt, n_workers) should come from problem, not config
        # Engines now get these from the problem object created by the interface

    @abstractmethod
    def solve(self, problem) -> "_Section":
        """Compute and return the section (or Results that inherit _Section)."""
        raise NotImplementedError
