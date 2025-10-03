"""
Abstract base class for Poincare section seeding strategies.

This module provides the abstract base class for implementing seeding
strategies in the Poincare section framework. Seeding strategies determine
how initial conditions are generated for return map computation.

The main class :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase` 
defines the interface that all concrete seeding strategies must implement, providing common
all concrete seeding strategies must implement, providing common
functionality for configuration management and seed generation.

Different strategies are appropriate for different dynamical systems
and analysis goals. The base class provides a common interface while
allowing concrete implementations to define their specific seeding
logic.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from hiten.algorithms.poincare.core.config import (_ReturnMapConfig,
                                                   _SeedingConfig)


class _SeedingStrategyBase(ABC):
    """Abstract base class for Poincare section seeding strategies.

    This abstract base class defines the interface that all concrete
    seeding strategies must implement. It provides common functionality
    for configuration management and defines the contract for seed
    generation.

    Parameters
    ----------
    section_cfg : :class:`~hiten.algorithms.poincare.core.interfaces._SectionInterface`
        Section configuration containing section coordinate and plane
        coordinate information.
    map_cfg : :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`
        Seeding configuration containing parameters such as the
        number of seeds to generate.

    Attributes
    ----------
    _section_cfg : :class:`~hiten.algorithms.poincare.core.interfaces._SectionInterface`
        The section configuration.
    _map_cfg : :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`
        The seeding configuration.
    _cached_limits : dict[tuple[float, int], list[float]]
        Class-level cache for computed limits, keyed by (h0, n_seeds)
        to avoid redundant computation.

    Notes
    -----
    Concrete subclasses must implement the `generate` method to define
    how initial conditions are generated. The base class provides
    convenient access to configuration parameters and common functionality.

    All time units are in nondimensional units unless otherwise specified.
    """

    _cached_limits: dict[tuple[float, int], list[float]] = {}
    
    def __init__(self, map_cfg: _SeedingConfig) -> None:
        self._map_cfg = map_cfg

    @property
    def config(self) -> "_SeedingConfig":
        """Get the map configuration.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`
            The map configuration containing global parameters.

        Notes
        -----
        This property provides access to the map configuration
        which contains information about the global parameters
        used for seeding.
        """
        return self._map_cfg

    @property
    def map_config(self) -> "_ReturnMapConfig":
        """Get the seeding configuration.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.config._ReturnMapConfig`
            The seeding configuration containing parameters such as
            the number of seeds to generate.

        Notes
        -----
        This property provides access to the seeding configuration
        that controls the seed generation process.
        """
        return self._map_cfg

    @property
    def n_seeds(self) -> int:
        """Get the number of seeds to generate.

        Returns
        -------
        int
            The number of seeds to generate as specified in the
            seeding configuration.

        Notes
        -----
        This property provides convenient access to the number of
        seeds parameter from the configuration. Note: This now accesses
        from options, not config. Subclasses should override if using
        different options structure.
        """
        # For backward compatibility, try to get from options-like structure
        # Concrete strategies should set this properly
        return getattr(self._map_cfg, 'n_seeds', 20)
    
    @property
    def plane_coords(self) -> Tuple[str, str]:
        """Get the plane coordinate labels.

        Returns
        -------
        tuple[str, str]
            Tuple of two coordinate labels that define the section
            plane (e.g., ("q2", "p2")).

        Notes
        -----
        This property provides access to the plane coordinate labels
        from the section configuration.
        """
        return self._map_cfg.plane_coords
    
    @abstractmethod
    def generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any, find_turning_fn: Any) -> List[Tuple[float, float, float, float]]:
        """Generate initial conditions for the Poincare section.

        This abstract method must be implemented by concrete subclasses
        to define how initial conditions are generated for the specific
        seeding strategy.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO (Center Manifold Local Orbit) table for polynomial
            evaluation and coordinate transformation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given
            constraints and energy level.
        find_turning_fn : callable
            Function to find turning points for a given coordinate
            and energy level.

        Returns
        -------
        list[tuple[float, float, float, float]]
            List of initial conditions, each represented as a tuple
            of four coordinates (q2, p2, q3, p3) in the center manifold
            coordinate system.

        Notes
        -----
        This method must be implemented by concrete subclasses to
        define the specific seeding logic. The implementation should
        generate initial conditions that are likely to intersect
        with the Poincare section.

        The returned coordinates are in the center manifold coordinate
        system and should satisfy the energy constraint H(q,p) = h0.

        All coordinates are in nondimensional units.
        """
        pass

    def __call__(self, **kwargs):
        """Call the seeding strategy with the given parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the `generate` method.

        Returns
        -------
        list[tuple[float, float, float, float]]
            The result of calling the `generate` method with the
            provided parameters.

        Notes
        -----
        This method provides a callable interface to the seeding
        strategy, allowing it to be used as a function. It simply
        delegates to the `generate` method.
        """
        return self.generate(**kwargs)
