"""Base class for center manifold seeding strategies.

This module provides the base class for seeding strategies used to generate
initial conditions on center manifolds of collinear libration points in the
Circular Restricted Three-Body Problem (CR3BP).

The main class :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase` 
defines the interface for all seeding strategies and provides common functionality for Hill
boundary validation and seed generation.
"""
from typing import Any, Callable

from hiten.algorithms.poincare.centermanifold.config import \
    CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.interfaces import \
    _CenterManifoldSectionInterface
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _CenterManifoldSeedingBase(_SeedingStrategyBase):
    """Base class for center manifold seeding strategies.

    This class provides the common interface and functionality for all
    seeding strategies used to generate initial conditions on center
    manifolds. It handles Hill boundary validation and provides caching
    for turning point limits.

    Parameters
    ----------
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.

    Notes
    -----
    Subclasses must implement the `generate` method to define how initial
    conditions are generated. The base class provides:
    - Hill boundary validation using turning points
    - Caching of turning point limits per energy level
    - Common seed validation functionality

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(self, map_config: CenterManifoldMapConfig) -> None:
        super().__init__(map_config)

    @property
    def plane_coords(self) -> tuple[str, str]:
        """Get the plane coordinate labels from the config's section coordinate.
        
        Returns
        -------
        tuple[str, str]
            Tuple of two coordinate labels that define the section plane.
        """
        return self._get_plane_coords(self.config.section_coord)

    def _get_plane_coords(self, section_coord: str) -> tuple[str, str]:
        """Get the plane coordinates for a given section coordinate.
        
        Parameters
        ----------
        section_coord : str
            Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').
            
        Returns
        -------
        tuple[str, str]
            Tuple of two coordinate labels that define the section plane.
        """
        return _CenterManifoldSectionInterface.get_plane_coords(section_coord)

    def _hill_boundary_limits(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        find_turning_fn: Callable
    ) -> list[float]:
        """Return turning-point limits for the two plane coordinates.

        This method computes the maximum absolute values for the plane
        coordinates that define the Hill boundary of the center manifold.
        Results are cached per energy level to avoid recomputing when
        multiple strategies are used with identical parameters.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[float]
            Maximum absolute values for the two plane coordinates defining
            the Hill boundary.

        Notes
        -----
        The Hill boundary represents the physical limits of the center
        manifold in phase space. Turning points are found by solving
        for the maximum coordinate values where the energy constraint
        H(q,p) = h0 can be satisfied.
        """
        key = (self.plane_coords, float(h0), id(H_blocks))
        if key in self._cached_limits:
            return self._cached_limits[key]

        limits = [find_turning_fn(c) for c in self.plane_coords]
        self._cached_limits[key] = limits
        return limits

    def _build_seed(
        self,
        plane_vals: tuple[float, float],
        *,
        solve_missing_coord_fn,
    ) -> tuple[float, float] | None:
        """Validate plane coordinates against the Hill boundary.

        This method validates that the given plane coordinates lie within
        the Hill boundary by attempting to solve for the missing coordinate
        using the energy constraint H(q,p) = h0.

        Parameters
        ----------
        plane_vals : tuple[float, float]
            Values for the two plane coordinates (nondimensional units).
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.

        Returns
        -------
        tuple[float, float] or None
            The plane coordinates if valid (within Hill boundary), None if
            the point lies outside the Hill boundary.

        Notes
        -----
        The validation process:
        1. Build constraint dictionary with the plane coordinate values
        2. Attempt to solve for the missing coordinate using the energy constraint
        3. Return the plane coordinates if a solution exists, None otherwise

        Points outside the Hill boundary cannot satisfy the energy constraint
        and are therefore invalid for center manifold trajectories.
        """

        cfg = self.config
        plane_coords = self._get_plane_coords(cfg.section_coord)

        constraints = _CenterManifoldSectionInterface.build_constraint_dict(cfg.section_coord, **{
            plane_coords[0]: plane_vals[0],
            plane_coords[1]: plane_vals[1],
        })

        missing_coord = {
            "q3": "p3",
            "p3": "q3",
            "q2": "p2",
            "p2": "q2",
        }[cfg.section_coord]

        missing_val = solve_missing_coord_fn(missing_coord, constraints)

        if missing_val is None:
            # Point lies outside Hill boundary.
            return None

        return plane_vals
