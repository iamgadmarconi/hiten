"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.types import (OrbitCorrectionResult,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.types.exceptions import (BackendError, ConvergenceError,
                                               EngineError)


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface."""

    def __init__(self, *, backend: _NewtonBackend, interface: _PeriodicOrbitCorrectorInterface) -> None:
        super().__init__(backend=backend, interface=interface)

    def _handle_backend_failure(
        self,
        exc: Exception,
        *,
        problem: _OrbitCorrectionProblem,
        call,
        interface,
    ) -> None:
        if isinstance(exc, (ConvergenceError, BackendError)):
            raise EngineError("Orbit correction failed") from exc
        raise EngineError("Unexpected error during orbit correction") from exc

    def _invoke_backend(self, call):
        return self._backend.run(*call.args, **call.kwargs)

    def _after_backend_success(self, outputs, *, problem, domain_payload, interface) -> None:
        x_corr, iterations, residual_norm = outputs
        try:
            self._backend.on_success(
                x_corr,
                iterations=int(iterations),
                residual_norm=float(residual_norm),
            )
            
        except Exception:
            pass
