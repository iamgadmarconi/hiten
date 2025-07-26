r"""
hiten.algorithms.poincare.seeding.strategies
===========================================

Implementation of various Poincaré section seeding strategies.

The module exposes concrete implementations of the :pyclass:`_CenterManifoldSeedingBase`
base class for different seeding strategies.
"""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.cm.config import (_CenterManifoldMapConfig,
                                                 _CenterManifoldSectionConfig)
from hiten.algorithms.poincare.cm.seeding import _CenterManifoldSeedingBase
from hiten.utils.log_config import logger


class _SingleAxisSeeding(_CenterManifoldSeedingBase):
    """Generate seeds varying only one coordinate of the section plane."""

    def __init__(
        self,
        section_config: "_CenterManifoldSectionConfig",
        map_config: _CenterManifoldMapConfig,
        *,
        seed_axis: Optional[str] = None,
    ) -> None:
        super().__init__(section_config, map_config)
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
        cfg = self.config

        axis_idx = 0
        if self._seed_axis is not None:
            try:
                axis_idx = cfg.plane_coords.index(self._seed_axis)
            except ValueError:
                logger.warning("seed_axis %s not in plane coords %s - defaulting to first", self._seed_axis, cfg.plane_coords)

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
    """Generate seeds along each coordinate axis in the section plane."""

    def __init__(self, section_config: "_CenterManifoldSectionConfig", map_config: _CenterManifoldMapConfig) -> None:
        super().__init__(section_config, map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        cfg = self.config

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
    """Generate seeds along several non-zero level-sets of each plane coordinate."""

    def __init__(self, section_config: "_CenterManifoldSectionConfig", map_config: _CenterManifoldMapConfig) -> None:
        super().__init__(section_config, map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        cfg = self.config
        plane_maxes = self._hill_boundary_limits(h0=h0, H_blocks=H_blocks, clmo_table=clmo_table, find_turning_fn=find_turning_fn)

        n_levels = max(2, int(np.sqrt(self.n_seeds)))
        seeds_per_level = max(1, self.n_seeds // (2 * n_levels))

        seeds: List[Tuple[float, float]] = []
        for i, varying_coord in enumerate(cfg.plane_coords):
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
    """Generate seeds distributed on concentric circles in the section plane."""

    def __init__(self, section_config: "_CenterManifoldSectionConfig", map_config: _CenterManifoldMapConfig) -> None:
        super().__init__(section_config, map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        cfg = self.config
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
    """Generate seeds by uniform rejection sampling inside the rectangular Hill box."""

    def __init__(self, section_config: "_CenterManifoldSectionConfig", map_config: _CenterManifoldMapConfig) -> None:
        super().__init__(section_config, map_config)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
        find_turning_fn: "Callable",
    ) -> List[Tuple[float, float]]:
        cfg = self.config
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
