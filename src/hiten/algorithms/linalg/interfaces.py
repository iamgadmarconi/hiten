"""Interfaces (Adapters) for linalg engines.

Currently provides a CR3BP interface that turns a position into a Jacobian
matrix suitable for eigen-structure classification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _EigenDecompositionProblem)
from hiten.algorithms.utils.core import BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint


@dataclass
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

    config: _EigenDecompositionConfig

    def __post_init__(self) -> None:
        super().__init__(domain_object=None)
        self._config = self.config

    def create_problem(
        self,
        *,
        config: Optional[_EigenDecompositionConfig] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> _EigenDecompositionProblem:
        cfg = config or self.config
        if matrix is not None:
            self._domain_object = np.asarray(matrix, dtype=float)
        if self._domain_object is None:
            raise ValueError("matrix must be provided for eigen decomposition problem")
        self._config = cfg
        return _EigenDecompositionProblem(A=self._domain_object, config=cfg)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> BackendCall:
        return BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem) -> EigenDecompositionResults:
        return outputs


@dataclass
class _LibrationPointInterface(
    _HitenBaseInterface[
        "LibrationPoint",
        _EigenDecompositionConfig,
        _EigenDecompositionProblem,
        EigenDecompositionResults,
        EigenDecompositionResults,
    ]
):
    config: _EigenDecompositionConfig

    def __post_init__(self) -> None:
        super().__init__(domain_object=None)
        self._config = self.config

    def create_problem(
        self,
        *,
        config: Optional[_EigenDecompositionConfig] = None,
        point: Optional["LibrationPoint"] = None,
    ) -> _EigenDecompositionProblem:
        cfg = config or self.config
        if point is not None:
            self._domain_object = point
        if self._domain_object is None:
            raise ValueError("libration point required to build stability problem")
        point_obj = self._domain_object
        jac = _jacobian_crtbp(
            point_obj.position[0],
            point_obj.position[1],
            point_obj.position[2],
            point_obj.mu,
        )
        self._config = cfg
        return _EigenDecompositionProblem(A=jac, config=cfg)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> BackendCall:
        return BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem) -> EigenDecompositionResults:
        return outputs
