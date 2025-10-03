"""Interfaces (Adapters) for linalg engines.

Currently provides a CR3BP interface that turns a position into a Jacobian
matrix suitable for eigen-structure classification.
"""

from typing import TYPE_CHECKING

import numpy as np

from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp
from hiten.algorithms.linalg.config import EigenDecompositionConfig
from hiten.algorithms.linalg.options import EigenDecompositionOptions
from hiten.algorithms.linalg.types import (
    EigenDecompositionResults,
    LinalgBackendRequest,
    LinalgBackendResponse,
    _EigenDecompositionProblem,
)
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint


class _EigenDecompositionInterface(
    _HitenBaseInterface[
        EigenDecompositionConfig,
        _EigenDecompositionProblem,
        EigenDecompositionResults,
        LinalgBackendResponse,
    ]
):
    """Adapter producing eigen-decomposition problems from matrices.
    """

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: np.ndarray,
        config: EigenDecompositionConfig,
        options: EigenDecompositionOptions,
    ) -> _EigenDecompositionProblem:
        """Create a eigen-decomposition problem.
        
        Parameters
        ----------
        domain_obj : np.ndarray
            Matrix to decompose.
        config : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            Compile-time configuration (problem type, system type).
        options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`, optional
            Runtime options (tolerances). If None, defaults are used.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            Eigen decomposition problem combining config and options.
        """
        matrix_arr = np.asarray(domain_obj, dtype=float)
        return _EigenDecompositionProblem(
            A=matrix_arr, 
            problem_type=config.problem_type,
            system_type=config.system_type,
            delta=options.delta,
            tol=options.tol
        )

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> _BackendCall:
        """Convert a eigen-decomposition problem to backend inputs.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            The eigen-decomposition problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend inputs.
        """
        request = LinalgBackendRequest(
            matrix=problem.A,
            problem_type=problem.problem_type,
            system_type=problem.system_type,
            delta=problem.delta,
            tol=problem.tol,
        )
        return _BackendCall(request=request)

    def to_results(self, outputs: LinalgBackendResponse, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        return outputs.results


class _LibrationPointInterface(
    _HitenBaseInterface[
        EigenDecompositionConfig,
        _EigenDecompositionProblem,
        EigenDecompositionResults,
        LinalgBackendResponse,
    ]
):

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: "LibrationPoint",
        config: EigenDecompositionConfig,
        options: EigenDecompositionOptions,
    ) -> _EigenDecompositionProblem:
        """Create a eigen-decomposition problem.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.libration.base.LibrationPoint`
            The domain object.
        config : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            Compile-time configuration (problem type, system type).
        options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`, optional
            Runtime options (tolerances). If None, defaults are used.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            Eigen decomposition problem combining config and options.
        """
        jac = _jacobian_crtbp(
            domain_obj.position[0],
            domain_obj.position[1],
            domain_obj.position[2],
            domain_obj.mu,
        )
        return _EigenDecompositionProblem(
            A=jac, 
            problem_type=config.problem_type,
            system_type=config.system_type,
            delta=options.delta,
            tol=options.tol
        )

    def to_backend_inputs(self, problem: _EigenDecompositionProblem) -> _BackendCall:
        """Convert a eigen-decomposition problem to backend inputs.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            The eigen-decomposition problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend inputs.
        """
        request = LinalgBackendRequest(
            matrix=problem.A,
            problem_type=problem.problem_type,
            system_type=problem.system_type,
            delta=problem.delta,
            tol=problem.tol,
        )
        return _BackendCall(request=request)

    def to_results(self, outputs: LinalgBackendResponse, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        return outputs.results
