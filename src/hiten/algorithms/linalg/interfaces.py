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
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint


class _EigenDecompositionInterface(
    _HitenBaseInterface[
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
        domain_obj: np.ndarray,
        config: Optional[_EigenDecompositionConfig] = None,
    ) -> _EigenDecompositionProblem:
        if config is None:
            raise ValueError("config must be provided for eigen decomposition problem")
        matrix_arr = np.asarray(domain_obj, dtype=float)
        return _EigenDecompositionProblem(A=matrix_arr, config=config)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> _BackendCall:
        return _BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        return outputs


class _LibrationPointInterface(
    _HitenBaseInterface[
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
        domain_obj: "LibrationPoint",
        config: Optional[_EigenDecompositionConfig] = None,
    ) -> _EigenDecompositionProblem:
        if config is None:
            raise ValueError("config must be provided for eigen decomposition problem")
        jac = _jacobian_crtbp(
            domain_obj.position[0],
            domain_obj.position[1],
            domain_obj.position[2],
            domain_obj.mu,
        )
        return _EigenDecompositionProblem(A=jac, config=config)

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> _BackendCall:
        return _BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        return outputs
