"""Concrete implementations of center manifold seeding strategies.

This module provides concrete implementations of seeding strategies for
generating initial conditions on center manifolds of collinear libration
points in the Circular Restricted Three-Body Problem (CR3BP).
"""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.centermanifold.config import \
    CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.seeding import \
    _CenterManifoldSeedingBase
from hiten.algorithms.types.exceptions import EngineError
from hiten.utils.log_config import logger


def _make_strategy(map_config: "CenterManifoldMapConfig", **kwargs) -> "_CenterManifoldSeedingBase":
    """Factory returning a concrete seeding strategy.

    Parameters
    ----------
    kind : str
        Strategy identifier. Must be one of: 'single', 'axis_aligned', 
        'level_sets', 'radial', or 'random'.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Map-level configuration containing global parameters such as
        ``n_seeds`` and ``seed_axis``.
    **kwargs
        Additional keyword arguments forwarded to the concrete strategy
        constructor.

    Returns
    -------
    :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase`
        Concrete seeding strategy instance.

    Raises
    ------
    ValueError
        If ``kind`` is not a valid strategy identifier.

    Notes
    -----
    The available strategies are:
    - 'single': Single axis seeding along one coordinate direction
    - 'axis_aligned': Seeding aligned with coordinate axes
    - 'level_sets': Seeding based on level sets of the Hamiltonian
    - 'radial': Radial seeding pattern from the periodic orbit
    - 'random': Random seeding within specified bounds
    """
    _STRATEGY_MAP = {
        "single": _SingleAxisSeeding,
        "axis_aligned": _AxisAlignedSeeding,
        "level_sets": _LevelSetsSeeding,
        "radial": _RadialSeeding,
        "random": _RandomSeeding,
    }
    try:
        cls = _STRATEGY_MAP[map_config.seed_strategy]
    except KeyError as exc:
        raise EngineError(f"Unknown seed_strategy '{map_config.seed_strategy}'") from exc
    return cls(map_config, **kwargs)


class _SingleAxisSeeding(_CenterManifoldSeedingBase):
    """Single axis seeding strategy.

    This strategy generates seeds by varying only one coordinate of the
    section plane while keeping the other coordinate fixed at zero.
    This creates a line of seeds along one axis of the Poincare section.

    Parameters
    ----------
        Configuration for the Poincare section.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.
    seed_axis : str, optional
        Coordinate axis to vary ('q2', 'p2', 'q3', or 'p3'). If None,
        uses the seed_axis from map_config.

    Notes
    -----
    This strategy is useful for exploring the center manifold along
    specific coordinate directions. The seeds are distributed linearly
    along the chosen axis within the Hill boundary limits.

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(
        self,
        map_config: CenterManifoldMapConfig,
        *,
        seed_axis: Optional[str] = None,
    ) -> None:
        super().__init__(map_config)
        # seed_axis can be provided explicitly or taken from map_config
        self._seed_axis = seed_axis or map_config.seed_axis

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        """Generate seeds along a single axis of the section plane.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[tuple[float, float]]
            List of valid seed points on the section plane.

        Notes
        -----
        The seeds are distributed linearly along the chosen axis from
        -0.9 * axis_max to 0.9 * axis_max, where axis_max is the Hill
        boundary limit for that coordinate.
        """
        plane_coords = self._get_plane_coords(self.config.section_coord)
        axis_idx = 0
        if self._seed_axis is not None:
            try:
                axis_idx = plane_coords.index(self._seed_axis)
            except ValueError:
                logger.warning("seed_axis %s not in plane coords %s - defaulting to first", self._seed_axis, plane_coords)

        limits = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)
        axis_max = limits[axis_idx]
        values = np.linspace(-0.9 * axis_max, 0.9 * axis_max, self.n_seeds)

        seeds: list[tuple[float, float]] = []
        for v in values:
            plane = [0.0, 0.0]
            plane[axis_idx] = float(v)
            seed = self._build_seed(tuple(plane), solve_missing_coord_fn=solve_missing_coord_fn)
            if seed is not None:
                seeds.append(seed)

        return seeds


class _AxisAlignedSeeding(_CenterManifoldSeedingBase):
    """Axis-aligned seeding strategy.

    This strategy generates seeds along each coordinate axis of the
    section plane. It creates two lines of seeds: one along each axis,
    providing good coverage of the coordinate directions.

    Parameters
    ----------
        Configuration for the Poincare section.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.

    Notes
    -----
    This strategy is useful for exploring the center manifold along
    both coordinate directions. The seeds are distributed linearly
    along each axis within the Hill boundary limits, with approximately
    half the seeds on each axis.

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(self, map_config: CenterManifoldMapConfig) -> None:
        super().__init__(map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        """Generate seeds along both coordinate axes of the section plane.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[tuple[float, float]]
            List of valid seed points on the section plane.

        Notes
        -----
        The seeds are distributed along both axes, with approximately
        half the seeds on each axis within the Hill boundary limits.
        """
        plane_maxes = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)
        max1, max2 = plane_maxes
        seeds: list[tuple[float, float]] = []

        seeds_per_axis = max(1, self.n_seeds // 2)
        axis_vals1 = np.linspace(-0.9 * max1, 0.9 * max1, seeds_per_axis)
        for v in axis_vals1:
            seed = self._build_seed((v, 0.0), solve_missing_coord_fn=solve_missing_coord_fn)
            if seed is not None:
                seeds.append(seed)

        axis_vals2 = np.linspace(-0.9 * max2, 0.9 * max2, seeds_per_axis)
        for v in axis_vals2:
            seed = self._build_seed((0.0, v), solve_missing_coord_fn=solve_missing_coord_fn)
            if seed is not None:
                seeds.append(seed)

        return seeds[: self.n_seeds]


class _LevelSetsSeeding(_CenterManifoldSeedingBase):
    """Level sets seeding strategy.

    This strategy generates seeds along several non-zero level sets of
    each plane coordinate. It creates a grid-like pattern of seeds
    covering the section plane more uniformly than axis-aligned strategies.

    Parameters
    ----------
        Configuration for the Poincare section.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.

    Notes
    -----
    This strategy provides good coverage of the section plane by creating
    a grid of seeds. The number of level sets is determined by the square
    root of the total number of seeds, and near-zero levels are skipped
    to avoid redundancy with axis-aligned strategies.

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(self, map_config: CenterManifoldMapConfig) -> None:
        super().__init__(map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        """Generate seeds along level sets of each plane coordinate.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[tuple[float, float]]
            List of valid seed points on the section plane.

        Notes
        -----
        The seeds are distributed in a grid pattern along level sets
        of each coordinate, with near-zero levels skipped to avoid
        redundancy with axis-aligned strategies.
        """
        plane_coords = self._get_plane_coords(self.config.section_coord)
        plane_maxes = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)

        n_levels = max(2, int(np.sqrt(self.n_seeds)))
        seeds_per_level = max(1, self.n_seeds // (2 * n_levels))

        seeds: List[Tuple[float, float]] = []
        for i, _ in enumerate(plane_coords):
            other_coord_idx = 1 - i
            level_vals = np.linspace(
                -0.7 * plane_maxes[other_coord_idx],
                0.7 * plane_maxes[other_coord_idx],
                n_levels + 2,
            )[1:-1]  # exclude endpoints

            for level_val in level_vals:
                if abs(level_val) < 0.05 * plane_maxes[other_coord_idx]:
                    continue  # skip near-zero levels

                varying_vals = np.linspace(
                    -0.8 * plane_maxes[i],
                    0.8 * plane_maxes[i],
                    seeds_per_level,
                )
                for varying_val in varying_vals:
                    plane_vals: List[float] = [0.0, 0.0]
                    plane_vals[i] = float(varying_val)
                    plane_vals[other_coord_idx] = float(level_val)

                    seed = self._build_seed(tuple(plane_vals), solve_missing_coord_fn=solve_missing_coord_fn)
                    if seed is not None:
                        seeds.append(seed)

        return seeds


class _RadialSeeding(_CenterManifoldSeedingBase):
    """Radial seeding strategy.

    This strategy generates seeds distributed on concentric circles in the
    section plane. It creates a radial pattern of seeds that provides
    good coverage of the section plane in polar coordinates.

    Parameters
    ----------
        Configuration for the Poincare section.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.

    Notes
    -----
    This strategy is useful for exploring the center manifold in a radial
    pattern. The seeds are distributed on concentric circles with uniform
    angular spacing, providing good coverage of the section plane.

    The number of radial and angular points is determined by the total
    number of seeds, with the radial count based on the square root of
    the total seeds divided by 2*pi.

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(self, map_config: CenterManifoldMapConfig) -> None:
        super().__init__(map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        """Generate seeds distributed on concentric circles in the section plane.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[tuple[float, float]]
            List of valid seed points on the section plane.

        Notes
        -----
        The seeds are distributed on concentric circles with uniform
        angular spacing, providing good coverage in polar coordinates.
        """
        plane_maxes = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)
        max_radius = 0.8 * min(*plane_maxes)

        n_radial = max(1, int(np.sqrt(self.n_seeds / (2 * np.pi))))
        n_angular = max(4, self.n_seeds // n_radial)

        seeds: List[Tuple[float, float]] = []
        for i in range(n_radial):
            r = (i + 1) / n_radial * max_radius
            for j in range(n_angular):
                theta = 2 * np.pi * j / n_angular
                plane_val1 = r * np.cos(theta)
                plane_val2 = r * np.sin(theta)

                if not (
                    abs(plane_val1) < plane_maxes[0] and abs(plane_val2) < plane_maxes[1]
                ):
                    continue

                seed = self._build_seed((plane_val1, plane_val2), solve_missing_coord_fn=solve_missing_coord_fn)
                if seed is not None:
                    seeds.append(seed)

                if len(seeds) >= self.n_seeds:
                    return seeds

        return seeds


class _RandomSeeding(_CenterManifoldSeedingBase):
    """Random seeding strategy.

    This strategy generates seeds by uniform rejection sampling inside the
    rectangular Hill box. It creates a random distribution of seeds that
    provides good coverage of the section plane through statistical sampling.

    Parameters
    ----------
        Configuration for the Poincare section.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Configuration for the center manifold map.

    Notes
    -----
    This strategy uses rejection sampling to generate random seeds within
    the Hill boundary. It attempts to generate the requested number of
    seeds, but may produce fewer if many random points fall outside the
    valid region.

    The strategy uses a maximum of 10 * n_seeds attempts to avoid infinite
    loops when the valid region is small relative to the Hill box.

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(self, map_config: CenterManifoldMapConfig) -> None:
        super().__init__(map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        """Generate seeds using uniform rejection sampling within the Hill box.

        Parameters
        ----------
        h0 : float
            Energy level for the center manifold (nondimensional units).
        H_blocks : Any
            Hamiltonian polynomial blocks for energy computation.
        clmo_table : Any
            CLMO table for polynomial evaluation.
        solve_missing_coord_fn : callable
            Function to solve for the missing coordinate given constraints.
        find_turning_fn : callable
            Function to find turning points for a given coordinate.

        Returns
        -------
        list[tuple[float, float]]
            List of valid seed points on the section plane.

        Notes
        -----
        Uses rejection sampling with a maximum of 10 * n_seeds attempts
        to avoid infinite loops when the valid region is small.
        """
        plane_maxes = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)

        seeds: List[Tuple[float, float]] = []
        max_attempts = self.n_seeds * 10
        attempts = 0

        rng = np.random.default_rng()
        while len(seeds) < self.n_seeds and attempts < max_attempts:
            attempts += 1
            plane_val1 = rng.uniform(-0.9 * plane_maxes[0], 0.9 * plane_maxes[0])
            plane_val2 = rng.uniform(-0.9 * plane_maxes[1], 0.9 * plane_maxes[1])

            seed = self._build_seed((plane_val1, plane_val2), solve_missing_coord_fn=solve_missing_coord_fn)
            if seed is not None:
                seeds.append(seed)

        if len(seeds) < self.n_seeds:
            logger.warning(
                "Only generated %d out of %d requested random seeds",
                len(seeds),
                self.n_seeds,
            )

        return seeds
