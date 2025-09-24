"""Engines orchestrating linalg backends and interfaces."""

from dataclasses import dataclass

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _EigenDecompositionProblem,
                                           _ProblemType)
from hiten.algorithms.types.core import BackendCall, _HitenBaseEngine


@dataclass
class _LinearStabilityEngine(_HitenBaseEngine[_EigenDecompositionProblem, EigenDecompositionResults, EigenDecompositionResults]):
    backend: _LinalgBackend

    def __init__(self, backend: _LinalgBackend) -> None:
        super().__init__(backend=backend, backend_method="solve")

    def _invoke_backend(self, call: BackendCall) -> EigenDecompositionResults:
        problem = call.args[0]
        self.backend.system_type = problem.config.system_type
        problem_type = problem.config.problem_type

        n = problem.A.shape[0]
        empty_vals = np.array([], dtype=np.complex128)
        empty_vecs = np.zeros((n, 0), dtype=np.complex128)
        empty_complex = np.array([], dtype=np.complex128)
        results = EigenDecompositionResults(
            stable=empty_vals,
            unstable=empty_vals,
            center=empty_vals,
            Ws=empty_vecs,
            Wu=empty_vecs,
            Wc=empty_vecs,
            nu=empty_complex,
            eigvals=empty_complex,
            eigvecs=np.zeros((0, 0), dtype=np.complex128),
        )

        if problem_type in (_ProblemType.EIGENVALUE_DECOMPOSITION, _ProblemType.ALL):
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(problem.A, problem.config.delta)
            results = results.__class__(sn, un, cn, Ws, Wu, Wc, results.nu, results.eigvals, results.eigvecs)

        if problem_type in (_ProblemType.STABILITY_INDICES, _ProblemType.ALL):
            nu, eigvals, eigvecs = self.backend.stability_indices(problem.A, problem.config.tol)
            results = results.__class__(
                results.stable,
                results.unstable,
                results.center,
                results.Ws,
                results.Wu,
                results.Wc,
                nu,
                eigvals,
                eigvecs,
            )

        return results
