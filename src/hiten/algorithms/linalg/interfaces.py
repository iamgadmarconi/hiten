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


@dataclass(frozen=True)
class _EigenDecompositionInterface:
    """Stateless adapter that builds numeric eigen problems."""

    config: _EigenDecompositionConfig

    def create_problem(self, matrix: np.ndarray) -> _EigenDecompositionProblem:
        return _EigenDecompositionProblem(A=matrix, config=self.config)


@dataclass(frozen=True)
class _LibrationPointInterface:
    config: _EigenDecompositionConfig

    def create_problem(self, point: "LibrationPoint") -> _EigenDecompositionProblem:
        jac = _jacobian_crtbp(
            point.position[0],
            point.position[1],
            point.position[2],
            point.mu,
        )
        return _EigenDecompositionProblem(A=jac, config=self.config)
