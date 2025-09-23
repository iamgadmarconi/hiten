"""Base types and protocols for the linear algebra module."""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.engine import _LinearStabilityEngine
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           StabilityIndicesResults,
                                           _SystemType,
                                           _ProblemType)
from hiten.algorithms.utils.exceptions import EngineError


@dataclass
class StabilityProperties:

    _engine: _LinearStabilityEngine
    _interface: _EigenDecompositionInterface
    _stability_info: Tuple[EigenDecompositionResults, StabilityIndicesResults] | None = None

    @classmethod
    def with_default_engine(cls, *, interface: _EigenDecompositionInterface) -> "StabilityProperties":
        backend = _LinalgBackend()
        intf = interface
        engine = _LinearStabilityEngine(backend=backend, interface=intf)
        return cls(_engine=engine, _interface=interface)

    def compute_linear_stability(
        self,
        *,
        system_type: _SystemType = _SystemType.CONTINUOUS,
        problem_type: _ProblemType = _ProblemType.EIGENVALUE_DECOMPOSITION,
        delta: float = 1e-4,
        tol: float = 1e-8,
    ) -> Tuple[EigenDecompositionResults, StabilityIndicesResults]:
        """Compute linear stability for a linear system."""
        cfg = replace(self._interface.config, system_type=system_type, delta=delta, tol=tol, problem_type=problem_type)
        self._interface = replace(self._interface, config=cfg)
        problem = self._interface.create_problem()

        if self._engine is None:
            raise EngineError("StabilityProperties requires an injected _LinearStabilityEngine; provide via constructor.")

        return self._engine.solve(problem)

    def __post_init__(self):
        self._stability_info = self.compute_linear_stability(
            system_type=self._interface.config.system_type,
            delta=self._interface.config.delta,
            tol=self._interface.config.tol,
        )

    @property
    def is_stable(self) -> bool:
        """Check if the linear system is stable."""
        unstable_eigenvalues = self._stability_info[0].unstable
        return len(unstable_eigenvalues) == 0

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvalues of the linear system.

        Returns
        -------
        tuple
            (stable_eigenvalues, unstable_eigenvalues, center_eigenvalues)
            Each array contains eigenvalues in nondimensional units.
        """
        results = self._stability_info[0]
        return results.stable, results.unstable, results.center

    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvectors of the linear system.

        Returns
        -------
        tuple
            (stable_eigenvectors, unstable_eigenvectors, center_eigenvectors)
            Each array contains eigenvectors as column vectors.
        """
        results = self._stability_info[0]
        return results.Ws, results.Wu, results.Wc
