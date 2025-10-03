"""Engines orchestrating linalg backends and interfaces."""

from dataclasses import dataclass

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           LinalgBackendRequest,
                                           LinalgBackendResponse,
                                           _EigenDecompositionProblem,
                                           _ProblemType)
from hiten.algorithms.types.core import _BackendCall, _HitenBaseEngine


class _LinearStabilityEngine(_HitenBaseEngine[_EigenDecompositionProblem, EigenDecompositionResults, EigenDecompositionResults]):
    """Engine orchestrating linalg backends and interfaces.
    
    Parameters
    ----------
    backend : :class:`~hiten.algorithms.linalg.backend._LinalgBackend`
        Backend responsible for the computational steps of eigenvalue decomposition.
    interface : :class:`~hiten.algorithms.linalg.interfaces._EigenDecompositionInterface`, optional
        Interface for handling eigenvalue decomposition problems. If None, a default interface is used.
    """

    def __init__(self, backend: _LinalgBackend, interface: _EigenDecompositionInterface | None = None) -> None:
        super().__init__(backend=backend, interface=interface)

    def _invoke_backend(self, call: _BackendCall) -> LinalgBackendResponse:
        request: LinalgBackendRequest = call.request
        self.backend.system_type = request.system_type
        problem_type = request.problem_type

        n = request.matrix.shape[0]
        empty_vals = np.array([], dtype=np.complex128)
        empty_vecs = np.zeros((n, 0), dtype=np.complex128)
        empty_complex = np.array([], dtype=np.complex128)
        current = EigenDecompositionResults(
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
        metadata: dict[str, object] = {}

        if problem_type in (_ProblemType.EIGENVALUE_DECOMPOSITION, _ProblemType.ALL):
            sn, un, cn, Ws, Wu, Wc = self.backend.eigenvalue_decomposition(request.matrix, request.delta)
            current = EigenDecompositionResults(sn, un, cn, Ws, Wu, Wc, current.nu, current.eigvals, current.eigvecs)

        if problem_type in (_ProblemType.STABILITY_INDICES, _ProblemType.ALL):
            nu, eigvals, eigvecs = self.backend.stability_indices(request.matrix, request.tol)
            current = EigenDecompositionResults(
                current.stable,
                current.unstable,
                current.center,
                current.Ws,
                current.Wu,
                current.Wc,
                nu,
                eigvals,
                eigvecs,
            )
            metadata.update({
                "nu": nu,
                "eigvals": eigvals,
            })

        response = LinalgBackendResponse(results=current, metadata=metadata)
        return response
