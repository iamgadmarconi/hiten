r"""
hiten.algorithms.poincare.seeding.strategies
===========================================

Implementation of various Poincaré section seeding strategies.

The module exposes concrete implementations of the :pyclass:`_SeedingStrategy`
base class for different seeding strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.seeding.base import _SeedingStrategy
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from hiten.algorithms.poincare.config import _PoincareSectionConfig


class _SingleAxisSeeding(_SeedingStrategy):
    """Generate seeds varying only one coordinate of the section plane."""

    def __init__(
        self,
        section_config: "_PoincareSectionConfig",
        *,
        n_seeds: int = 20,
        seed_axis: Optional[str] = None,
    ) -> None:
        super().__init__(section_config, n_seeds)
        self._seed_axis = seed_axis

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self.config

        if self._seed_axis is None:
            logger.warning(
                "seed_axis not specified for 'single' strategy; using the first plane coordinate (%s)",
                cfg.plane_coords[0],
            )
            axis_idx = 0
        else:
            try:
                axis_idx = cfg.plane_coords.index(self._seed_axis)
            except ValueError:
                logger.warning(
                    "seed_axis '%s' not found in plane coordinates %s. Using first coordinate.",
                    self._seed_axis,
                    cfg.plane_coords,
                )
                axis_idx = 0

        plane_maxes: List[float] = []
        for coord in cfg.plane_coords:
            plane_maxes.append(self.find_turning(coord, h0, H_blocks, clmo_table))

        coord_vals = np.linspace(
            -0.9 * plane_maxes[axis_idx], 0.9 * plane_maxes[axis_idx], self.n_seeds
        )

        seeds: List[Tuple[float, float, float, float]] = []
        for coord_val in coord_vals:
            plane_vals: List[float] = [0.0, 0.0]
            plane_vals[axis_idx] = float(coord_val)

            constraints = cfg.build_constraint_dict(**{
                cfg.plane_coords[0]: plane_vals[0],
                cfg.plane_coords[1]: plane_vals[1],
            })

            missing_val = solve_missing_coord_fn(
                cfg.missing_coord, constraints, h0, H_blocks, clmo_table
            )
            if missing_val is None:
                continue  # point lies outside Hill region

            other_vals: List[float] = [0.0, 0.0]
            missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
            other_vals[missing_idx] = missing_val

            seeds.append(cfg.build_state(tuple(plane_vals), tuple(other_vals)))

        return seeds


class _AxisAlignedSeeding(_SeedingStrategy):
    """Generate seeds along each coordinate axis in the section plane."""

    def __init__(self, section_config: "_PoincareSectionConfig", *, n_seeds: int = 20) -> None:
        super().__init__(section_config, n_seeds)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self.config
        seeds_per_axis = max(1, self.n_seeds // 2)

        plane_maxes = [
            self.find_turning(coord, h0, H_blocks, clmo_table) for coord in cfg.plane_coords
        ]

        seeds: List[Tuple[float, float, float, float]] = []
        for i, coord in enumerate(cfg.plane_coords):
            axis_vals = np.linspace(
                -0.9 * plane_maxes[i], 0.9 * plane_maxes[i], seeds_per_axis
            )
            for val in axis_vals:
                plane_vals: List[float] = [0.0, 0.0]
                plane_vals[i] = float(val)

                constraints = cfg.build_constraint_dict(**{
                    cfg.plane_coords[0]: plane_vals[0],
                    cfg.plane_coords[1]: plane_vals[1],
                })

                missing_val = solve_missing_coord_fn(
                    cfg.missing_coord, constraints, h0, H_blocks, clmo_table
                )
                if missing_val is None:
                    continue

                other_vals: List[float] = [0.0, 0.0]
                missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
                other_vals[missing_idx] = missing_val

                seeds.append(cfg.build_state(tuple(plane_vals), tuple(other_vals)))

        return seeds


class _LevelSetsSeeding(_SeedingStrategy):
    """Generate seeds along several non-zero level-sets of each plane coordinate."""

    def __init__(self, section_config: "_PoincareSectionConfig", *, n_seeds: int = 20) -> None:
        super().__init__(section_config, n_seeds)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self.config

        plane_maxes = [
            self.find_turning(coord, h0, H_blocks, clmo_table) for coord in cfg.plane_coords
        ]

        n_levels = max(2, int(np.sqrt(self.n_seeds)))
        seeds_per_level = max(1, self.n_seeds // (2 * n_levels))

        seeds: List[Tuple[float, float, float, float]] = []
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

                    constraints = cfg.build_constraint_dict(**{
                        cfg.plane_coords[0]: plane_vals[0],
                        cfg.plane_coords[1]: plane_vals[1],
                    })
                    missing_val = solve_missing_coord_fn(
                        cfg.missing_coord, constraints, h0, H_blocks, clmo_table
                    )
                    if missing_val is None:
                        continue

                    other_vals: List[float] = [0.0, 0.0]
                    missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
                    other_vals[missing_idx] = missing_val
                    seeds.append(cfg.build_state(tuple(plane_vals), tuple(other_vals)))

        return seeds


class _RadialSeeding(_SeedingStrategy):
    """Generate seeds distributed on concentric circles in the section plane."""

    def __init__(self, section_config: "_PoincareSectionConfig", *, n_seeds: int = 20) -> None:
        super().__init__(section_config, n_seeds)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self.config

        plane_maxes = [
            self.find_turning(coord, h0, H_blocks, clmo_table) for coord in cfg.plane_coords
        ]
        max_radius = 0.8 * min(*plane_maxes)

        n_radial = max(1, int(np.sqrt(self.n_seeds / (2 * np.pi))))
        n_angular = max(4, self.n_seeds // n_radial)

        seeds: List[Tuple[float, float, float, float]] = []
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

                constraints = cfg.build_constraint_dict(**{
                    cfg.plane_coords[0]: plane_val1,
                    cfg.plane_coords[1]: plane_val2,
                })
                missing_val = solve_missing_coord_fn(
                    cfg.missing_coord, constraints, h0, H_blocks, clmo_table
                )
                if missing_val is None:
                    continue

                other_vals: List[float] = [0.0, 0.0]
                missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
                other_vals[missing_idx] = missing_val
                seeds.append(cfg.build_state((plane_val1, plane_val2), tuple(other_vals)))

                if len(seeds) >= self.n_seeds:
                    return seeds

        return seeds


class _RandomSeeding(_SeedingStrategy):
    """Generate seeds by uniform rejection sampling inside the rectangular Hill box."""

    def __init__(self, section_config: "_PoincareSectionConfig", *, n_seeds: int = 20) -> None:
        super().__init__(section_config, n_seeds)

    def generate(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn: "Callable",
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self.config

        plane_maxes = [
            self.find_turning(coord, h0, H_blocks, clmo_table) for coord in cfg.plane_coords
        ]

        seeds: List[Tuple[float, float, float, float]] = []
        max_attempts = self.n_seeds * 10
        attempts = 0

        rng = np.random.default_rng()
        while len(seeds) < self.n_seeds and attempts < max_attempts:
            attempts += 1
            plane_val1 = rng.uniform(-0.9 * plane_maxes[0], 0.9 * plane_maxes[0])
            plane_val2 = rng.uniform(-0.9 * plane_maxes[1], 0.9 * plane_maxes[1])

            constraints = cfg.build_constraint_dict(**{
                cfg.plane_coords[0]: plane_val1,
                cfg.plane_coords[1]: plane_val2,
            })
            missing_val = solve_missing_coord_fn(
                cfg.missing_coord, constraints, h0, H_blocks, clmo_table
            )
            if missing_val is None:
                continue

            other_vals: List[float] = [0.0, 0.0]
            missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
            other_vals[missing_idx] = missing_val
            seeds.append(cfg.build_state((plane_val1, plane_val2), tuple(other_vals)))

        if len(seeds) < self.n_seeds:
            logger.warning(
                "Only generated %d out of %d requested random seeds",
                len(seeds),
                self.n_seeds,
            )

        return seeds
