"""Interfaces (Adapters) for linalg engines.

Currently provides a CR3BP interface that turns a position into a Jacobian
matrix suitable for eigen-structure classification.
"""

from typing import TYPE_CHECKING

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
    """Adapter producing eigen-decomposition problems from matrices.
    """

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        domain_obj: np.ndarray,
        config: _EigenDecompositionConfig,
    ) -> _EigenDecompositionProblem:
        """Create a eigen-decomposition problem.
        
        Parameters
        ----------
        domain_obj : np.ndarray
            Matrix to decompose.
        config : :class:`~hiten.algorithms.linalg.config._EigenDecompositionConfig`
            Configuration for the eigenvalue decomposition.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            Eigen decomposition problem.
        """
        matrix_arr = np.asarray(domain_obj, dtype=float)
        return _EigenDecompositionProblem(
            A=matrix_arr, 
            problem_type=config.problem_type,
            system_type=config.system_type,
            delta=config.delta,
            tol=config.tol
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
        return _BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        """Convert backend outputs to eigen-decomposition results.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            The backend outputs.
        problem : :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            The eigen-decomposition problem.
        domain_payload : Any, optional
            The domain payload.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            The eigen-decomposition results.
        """
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
        config: _EigenDecompositionConfig,
    ) -> _EigenDecompositionProblem:
        """Create a eigen-decomposition problem.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.libration.base.LibrationPoint`
            The domain object.
        config : :class:`~hiten.algorithms.linalg.config._EigenDecompositionConfig`
            Configuration for the eigenvalue decomposition.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            Eigen decomposition problem.
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
            delta=config.delta,
            tol=config.tol
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
        return _BackendCall(args=(problem,))

    def to_results(self, outputs: EigenDecompositionResults, *, problem: _EigenDecompositionProblem, domain_payload: any = None) -> EigenDecompositionResults:
        """Convert backend outputs to eigen-decomposition results.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            The backend outputs.
        problem : :class:`~hiten.algorithms.linalg.types._EigenDecompositionProblem`
            The eigen-decomposition problem.
        domain_payload : Any, optional
            The domain payload.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            The eigen-decomposition results.
        """
        return outputs
