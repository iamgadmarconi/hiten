r"""
hiten.algorithms.poincare.seeding.base
====================================

Base class for Poincaré section seeding strategies.

The module exposes a base class :pyclass:`_SeedingStrategy` that defines the
interface for all seeding strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.poincare.config import _PoincareSectionConfig


class _SeedingStrategy(ABC):
    def __init__(self, section_config: "_PoincareSectionConfig", n_seeds: int = 20) -> None:
        self._cfg = section_config
        self.n_seeds = n_seeds

    @property
    def config(self) -> "_PoincareSectionConfig":
        return self._cfg

    @property
    def plane_coords(self) -> Tuple[str, str]:
        return self._cfg.plane_coords

    def find_turning(self, q_or_p: str, h0: float, H_blocks: List[np.ndarray], clmo: List[np.ndarray], initial_guess: float = 1e-3, expand_factor: float = 2.0, max_expand: int = 40) -> float:
        return _find_turning(q_or_p, h0, H_blocks, clmo, initial_guess, expand_factor, max_expand)

    @abstractmethod
    def generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any) -> List[Tuple[float, float, float, float]]:
        pass

    def __call__(self, **kwargs):
        return self.generate(**kwargs)


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