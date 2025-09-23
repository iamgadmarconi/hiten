"""Engines orchestrating linalg backends and interfaces."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           StabilityIndicesResults,
                                           _EigenDecompositionProblem,
                                           _ProblemType)



@dataclass
class _LinearStabilityEngine:
    """Engine: compute linear stability via Jacobian + eigen classification.
    
    Parameters
    ----------
    interface : :class:`~hiten.algorithms.linalg.interfaces._EigenDecompositionInterface`
        Interface for the eigenvalue decomposition.
    backend : :class:`~hiten.algorithms.linalg.backend._LinalgBackend`
        Backend for the eigenvalue decomposition.
    """
    interface: _EigenDecompositionInterface
    backend: _LinalgBackend

    def __post_init__(self):
        self.backend.system_type = self.interface.config.system_type

    def solve(self, problem: _EigenDecompositionProblem) -> Tuple[EigenDecompositionResults, StabilityIndicesResults]:
        self.backend.system_type = problem.config.system_type
        problem_type = problem.config.problem_type

        # Initialize empty/default results to keep return type consistent
        n = problem.A.shape[0]
        empty_vals = np.array([], dtype=np.complex128)
        empty_mat = np.zeros((n, 0), dtype=np.complex128)
        eig_results = EigenDecompositionResults(
            stable=empty_vals,
            unstable=empty_vals,
            center=empty_vals,
            Ws=empty_mat,
            Wu=empty_mat,
            Wc=empty_mat,
        )

        empty_indices = np.array([], dtype=np.complex128)
        empty_vecs = np.zeros((0, 0), dtype=np.complex128)
        si_results = StabilityIndicesResults(
            nu=empty_indices,
            eigvals=empty_indices,
            eigvecs=empty_vecs,
        )

        if problem_type == _ProblemType.EIGENVALUE_DECOMPOSITION:
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(
                problem.A,
                problem.config.delta,
            )
            eig_results = EigenDecompositionResults(sn, un, cn, Ws, Wu, Wc)

        elif problem_type == _ProblemType.STABILITY_INDICES:
            nu, eigvals, eigvecs = self.backend.stability_indices(problem.A, problem.config.tol)
            si_results = StabilityIndicesResults(nu, eigvals, eigvecs)

        elif problem_type == _ProblemType.ALL:
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(
                problem.A,
                problem.config.delta,
            )
            eig_results = EigenDecompositionResults(sn, un, cn, Ws, Wu, Wc)
            nu, eigvals, eigvecs = self.backend.stability_indices(problem.A, problem.config.tol)
            si_results = StabilityIndicesResults(nu, eigvals, eigvecs)

        return eig_results, si_results
