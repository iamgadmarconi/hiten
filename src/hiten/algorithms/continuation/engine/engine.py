"""Orbit-specific continuation engine wiring backend and interface closures."""

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.engine.base import _ContinuationEngine
from hiten.algorithms.continuation.interfaces import (
    _PeriodicOrbitContinuationInterface,
)
from hiten.algorithms.continuation.types import _ContinuationProblem
from hiten.algorithms.utils.exceptions import EngineError


class _OrbitContinuationEngine(_ContinuationEngine):
    """Engine orchestrating periodic orbit continuation via backend and interface."""

    def __init__(
        self,
        *,
        backend: _ContinuationBackend,
        interface: _PeriodicOrbitContinuationInterface | None = None,
    ) -> None:
        super().__init__(backend=backend, interface=interface)

    def _handle_backend_failure(
        self,
        exc: Exception,
        *,
        problem: _ContinuationProblem,
        call,
        interface,
    ) -> None:
        raise EngineError("Orbit continuation failed") from exc

    def _invoke_backend(self, call):
        interface = self._interface
        interface.bind_backend(self._backend)
        return self._backend.solve(*call.args, **call.kwargs)

    def _after_backend_success(self, outputs, *, problem, domain_payload, interface) -> None:
        family_repr, info = outputs
        try:
            last_repr = family_repr[-1] if family_repr else interface._representation(interface.domain_object)
            self._backend.on_success(
                np.asarray(last_repr, dtype=float),
                iterations=int(info.get("iterations", 0)),
                residual_norm=float(info.get("residual_norm", float("nan"))),
            )
        except Exception:
            pass
