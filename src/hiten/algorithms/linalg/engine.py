"""Engines orchestrating linalg backends and interfaces."""

from dataclasses import dataclass

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _EigenDecompositionProblem,
                                           _ProblemType)


@dataclass
class _LinearStabilityEngine:
    """Engine: compute linear stability via Jacobian + eigen classification."""

    backend: _LinalgBackend

    def solve(self, problem: _EigenDecompositionProblem) -> EigenDecompositionResults:
        self.backend.system_type = problem.config.system_type
        problem_type = problem.config.problem_type

        # Initialize empty/default results to keep return type consistent
        n = problem.A.shape[0]
        empty_vals = np.array([], dtype=np.complex128)
        empty_mat = np.zeros((n, 0), dtype=np.complex128)
        empty_indices = np.array([], dtype=np.complex128)
        empty_vecs = np.zeros((0, 0), dtype=np.complex128)
        results = EigenDecompositionResults(
            stable=empty_vals,
            unstable=empty_vals,
            center=empty_vals,
            Ws=empty_mat,
            Wu=empty_mat,
            Wc=empty_mat,
            nu=empty_indices,
            eigvals=empty_indices,
            eigvecs=empty_vecs,
        )

        if problem_type == _ProblemType.EIGENVALUE_DECOMPOSITION:
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(
                problem.A,
                problem.config.delta,
            )
            results = EigenDecompositionResults(sn, un, cn, Ws, Wu, Wc,
                                                results.nu,
                                                results.eigvals,
                                                results.eigvecs)
        elif problem_type == _ProblemType.STABILITY_INDICES:
            nu, eigvals, eigvecs = self.backend.stability_indices(problem.A, problem.config.tol)
            results = EigenDecompositionResults(results.stable,
                                                results.unstable,
                                                results.center,
                                                results.Ws,
                                                results.Wu,
                                                results.Wc,
                                                nu,
                                                eigvals,
                                                eigvecs)
        elif problem_type == _ProblemType.ALL:
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(
                problem.A,
                problem.config.delta,
            )
            nu, eigvals, eigvecs = self.backend.stability_indices(problem.A, problem.config.tol)
            results = EigenDecompositionResults(sn, un, cn, Ws, Wu, Wc, nu, eigvals, eigvecs)

        return results
