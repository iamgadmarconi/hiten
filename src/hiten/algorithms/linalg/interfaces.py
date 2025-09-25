"""Interfaces (Adapters) for linalg engines.

Currently provides a CR3BP interface that turns a position into a Jacobian
matrix suitable for eigen-structure classification.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _EigenDecompositionProblem)
from hiten.algorithms.types.core import BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint


class _EigenDecompositionInterface(
    _HitenBaseInterface[
        np.ndarray,
        _EigenDecompositionConfig,
        _EigenDecompositionProblem,
        EigenDecompositionResults,
        EigenDecompositionResults,
    ]
):
    """Adapter producing eigen-decomposition problems from matrices."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        matrix: np.ndarray,
        config: Optional[_EigenDecompositionConfig] = None,
    ) -> _EigenDecompositionProblem:
        if config is None:
            raise ValueError("config must be provided for eigen decomposition problem")
        matrix_arr = np.asarray(matrix, dtype=float)
        return _EigenDecompositionProblem(A=matrix_arr, config=config)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> BackendCall:
        return BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem) -> EigenDecompositionResults:
        return outputs


class _LibrationPointInterface(
    _HitenBaseInterface[
        "LibrationPoint",
        _EigenDecompositionConfig,
        _EigenDecompositionProblem,
        EigenDecompositionResults,
        EigenDecompositionResults,
    ]
):

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        point: "LibrationPoint",
        config: Optional[_EigenDecompositionConfig] = None,
    ) -> _EigenDecompositionProblem:
        if config is None:
            raise ValueError("config must be provided for eigen decomposition problem")
        jac = _jacobian_crtbp(
            point.position[0],
            point.position[1],
            point.position[2],
            point.mu,
        )
        return _EigenDecompositionProblem(A=jac, config=config)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> BackendCall:
        return BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem) -> EigenDecompositionResults:
        return outputs
