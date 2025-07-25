r"""
hiten.algorithms.poincare.seeding.base
====================================

Base class for Poincaré section seeding strategies.

The module exposes a base class :pyclass:`_CenterManifoldSeedingBase` that defines the
interface for all seeding strategies.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.poincare.config import _CenterManifoldSectionConfig


class _CenterManifoldSeedingBase(ABC):
    def __init__(self, section_config: "_CenterManifoldSectionConfig", n_seeds: int = 20) -> None:
        self._cfg = section_config
        self.n_seeds = n_seeds

    @property
    def config(self) -> "_CenterManifoldSectionConfig":
        return self._cfg

    @property
    def plane_coords(self) -> Tuple[str, str]:
        return self._cfg.plane_coords

    def find_turning(self, q_or_p: str, h0: float, H_blocks: List[np.ndarray], clmo: List[np.ndarray], initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40) -> float:
        return _find_turning(q_or_p, h0, H_blocks, clmo, initial_guess, expand_factor, max_expand)

    @abstractmethod
    def _generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any) -> List[Tuple[float, float, float, float]]:
        pass

    def __call__(self, **kwargs):
        return self._generate(**kwargs)

    _cached_limits: dict[tuple[float, int], list[float]] = {}

    def _hill_boundary_limits(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
    ) -> list[float]:
        """Return turning-point limits (max absolute) for the two plane coords.

        Results are cached per energy level to avoid recomputing when multiple
        strategies are used with identical parameters.
        """
        key = (float(h0), id(H_blocks))
        if key in self._cached_limits:
            return self._cached_limits[key]

        limits = [
            self.find_turning(c, h0, H_blocks, clmo_table) for c in self.plane_coords
        ]
        self._cached_limits[key] = limits
        return limits

    def _build_seed(
        self,
        plane_vals: tuple[float, float],
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        solve_missing_coord_fn,
    ) -> tuple[float, float, float, float] | None:
        """Try to construct full 4-tuple seed from plane coordinates.

        Returns *None* when the point lies outside Hill region.
        """
        cfg = self.config
        constraints = cfg.build_constraint_dict(**{
            cfg.plane_coords[0]: plane_vals[0],
            cfg.plane_coords[1]: plane_vals[1],
        })
        missing_val = solve_missing_coord_fn(
            cfg.missing_coord, constraints, h0, H_blocks, clmo_table
        )
        if missing_val is None:
            return None
        other_vals = [0.0, 0.0]
        missing_idx = 0 if cfg.missing_coord == cfg.other_coords[0] else 1
        other_vals[missing_idx] = missing_val
        return cfg.build_state(plane_vals, tuple(other_vals))


def _find_turning(q_or_p: str, h0: float, H_blocks: List[np.ndarray], clmo: List[np.ndarray], initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40) -> float:
    # Deferred import to avoid circular dependency and ensure availability at runtime
    from hiten.algorithms.poincare.map import _solve_missing_coord

    fixed_vals = {coord: 0.0 for coord in ("q2", "p2", "q3", "p3") if coord != q_or_p}
    
    root = _solve_missing_coord(
        q_or_p, fixed_vals, h0, H_blocks, clmo, 
        initial_guess, expand_factor, max_expand
    )
    
    if root is None:
        raise RuntimeError("Root finding for Hill boundary did not converge.")

    return root