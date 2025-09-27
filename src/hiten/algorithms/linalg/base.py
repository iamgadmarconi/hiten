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
    _result: EigenDecompositionResults | None = None
    _config: _EigenDecompositionConfig | None = None

    @classmethod
    def with_default_engine(cls) -> "StabilityProperties":
        from hiten.algorithms.linalg.config import _EigenDecompositionConfig
        from hiten.algorithms.linalg.types import _ProblemType, _SystemType
        config = _EigenDecompositionConfig(
            problem_type=_ProblemType.ALL,
            system_type=_SystemType.DISCRETE,
        )
        interface = _EigenDecompositionInterface()
        backend = _LinalgBackend()
        engine = _LinearStabilityEngine(backend=backend).with_interface(interface)
        return cls(engine, _config=config)

    @property
    def config(self) -> _EigenDecompositionConfig:
        if self._config is None:
            raise ValueError("Configuration not set")
        return self._config

    def compute(
        self,
        matrix: np.ndarray,
        *,
        system_type: _SystemType | None = None,
        problem_type: _ProblemType | None = None,
    ) -> EigenDecompositionResults:
        """Compose a problem from *matrix* and run the engine."""

        cfg = self._config
        if system_type is not None or problem_type is not None:
            cfg = _EigenDecompositionConfig(
                system_type=system_type or cfg.system_type,
                problem_type=problem_type or cfg.problem_type,
                delta=cfg.delta,
                tol=cfg.tol,
            )

        # Create a new interface instance for this computation (stateless)
        interface = _EigenDecompositionInterface()
        self._engine.with_interface(interface)
        
        problem = interface.create_problem(domain_obj=np.asarray(matrix, dtype=float), config=cfg)
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
