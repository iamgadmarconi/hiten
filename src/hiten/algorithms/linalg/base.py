"""Base types and protocols for the linear algebra module."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.engine import _LinearStabilityEngine
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _ProblemType, _SystemType)
from hiten.algorithms.types.exceptions import EngineError


@dataclass
class StabilityProperties:
    """Facade exposing linear stability results on demand."""

    _engine: _LinearStabilityEngine
    _interface: _EigenDecompositionInterface
    _result: EigenDecompositionResults | None = None

    @classmethod
    def with_default_engine(cls, *, config: _EigenDecompositionConfig) -> "StabilityProperties":
        interface = _EigenDecompositionInterface(config=config)
        backend = _LinalgBackend()
        engine = _LinearStabilityEngine(backend=backend).with_interface(interface)
        return cls(engine, interface)

    @property
    def config(self) -> _EigenDecompositionConfig:
        return self._interface.config

    def compute(
        self,
        matrix: np.ndarray,
        *,
        system_type: _SystemType | None = None,
        problem_type: _ProblemType | None = None,
    ) -> EigenDecompositionResults:
        """Compose a problem from *matrix* and run the engine."""

        cfg = self._interface.config
        if system_type is not None or problem_type is not None:
            cfg = _EigenDecompositionConfig(
                system_type=system_type or cfg.system_type,
                problem_type=problem_type or cfg.problem_type,
                delta=cfg.delta,
                tol=cfg.tol,
            )
            self._interface = _EigenDecompositionInterface(config=cfg)
            self._engine.with_interface(self._interface)

        problem = self._interface.create_problem(matrix=np.asarray(matrix, dtype=float))
        self._engine.with_interface(self._interface)
        self._result = self._engine.solve(problem)
        return self._result

    def require_result(self) -> EigenDecompositionResults:
        if self._result is None:
            raise EngineError("Stability results not computed; call compute() first")
        return self._result

    @property
    def is_stable(self) -> bool:
        result = self.require_result()
        return len(result.unstable) == 0

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = self.require_result()
        return result.stable, result.unstable, result.center

    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = self.require_result()
        return result.Ws, result.Wu, result.Wc

    def get_real_eigenvectors(self, vectors: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isreal(values)
        real_vals_arr = values[mask].astype(np.complex128)
        if np.any(mask):
            real_vecs_arr = vectors[:, mask]
        else:
            real_vecs_arr = np.zeros((vectors.shape[0], 0), dtype=np.complex128)
        return real_vals_arr, real_vecs_arr
