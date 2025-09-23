"""Interfaces (Adapters) for linalg engines.

Currently provides a CR3BP interface that turns a position into a Jacobian
matrix suitable for eigen-structure classification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.types import _EigenDecompositionProblem

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint

@dataclass
class _EigenDecompositionInterface:
    """Direct adapter for matrix-based stability problems (domain-agnostic)."""

    A: np.ndarray | None
    config: _EigenDecompositionConfig

    def create_problem(self) -> _EigenDecompositionProblem:
        return _EigenDecompositionProblem(A=self.A, config=self.config)   

@dataclass
class _LibrationPointInterface(_EigenDecompositionInterface):
    """Create linalg problems from CR3BP domain data."""

    point: "LibrationPoint"

    def __post_init__(self):
        if self.A is None:
            self.A = _jacobian_crtbp(
                self.point.position[0],
                self.point.position[1],
                self.point.position[2],
                self.point.mu
            )
