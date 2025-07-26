r"""
hiten.algorithms.poincare.seeding.base
====================================

Base class for Poincaré section seeding strategies.

The module exposes a base class :pyclass:`_CenterManifoldSeedingBase` that defines the
interface for all seeding strategies.
"""
from typing import Any, Callable

from hiten.algorithms.poincare.cm.config import _CenterManifoldSectionConfig, _CenterManifoldMapConfig
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _CenterManifoldSeedingBase(_SeedingStrategyBase):
    def __init__(self, section_config: _CenterManifoldSectionConfig, map_config: _CenterManifoldMapConfig) -> None:
        super().__init__(section_config, map_config)

    def _hill_boundary_limits(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        find_turning_fn: Callable
    ) -> list[float]:
        """Return turning-point limits (max absolute) for the two plane coords.

        Results are cached per energy level to avoid recomputing when multiple
        strategies are used with identical parameters.
        """
        key = (float(h0), id(H_blocks))
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
        """Validate *plane_vals* against the Hill boundary.
        """

        cfg = self.config

        constraints = cfg.build_constraint_dict(**{
            cfg.plane_coords[0]: plane_vals[0],
            cfg.plane_coords[1]: plane_vals[1],
        })

        missing_val = solve_missing_coord_fn(cfg.missing_coord, constraints)

        if missing_val is None:
            # Point lies outside Hill boundary.
            return None

        return plane_vals
