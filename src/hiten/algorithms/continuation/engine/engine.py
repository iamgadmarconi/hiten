"""Orbit-specific continuation engine wiring backend and interface closures.

This module provides the engine for the orbit continuation module.
"""

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.engine.base import _ContinuationEngine
from hiten.algorithms.continuation.interfaces import (
    _PeriodicOrbitContinuationInterface,
)
from hiten.algorithms.continuation.types import _ContinuationProblem
from hiten.algorithms.types.exceptions import EngineError


class _OrbitContinuationEngine(_ContinuationEngine):
    """Engine orchestrating periodic orbit continuation via backend and interface.
    
    Parameters
    ----------
    backend : :class:`~hiten.algorithms.continuation.backends.base._ContinuationBackend`
        The backend for the orbit continuation.
    interface : :class:`~hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface` | None
        The interface for the orbit continuation.
    """

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
        """Handle the backend failure.
        
        Parameters
        ----------
        exc : Exception
            The exception raised by the backend.
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The problem being solved.
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.
        interface : :class:`~hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface`
            The interface to the backend.
        """
        raise EngineError("Orbit continuation failed") from exc

    def _invoke_backend(self, call):
        """Invoke the backend.
        
        Parameters
        ----------
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.
        """
        return self._backend.run(*call.args, **call.kwargs)

    def _after_backend_success(self, outputs, *, problem, domain_payload, interface) -> None:
        """Handle the backend success.
        
        Parameters
        ----------
        outputs : tuple
            The outputs from the backend.
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The problem being solved.
        domain_payload : Any
            The domain payload.
        interface : :class:`~hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface`
            The interface to the backend.
        """
        family_repr, info = outputs
        try:
            last_repr = family_repr[-1] if family_repr else interface._representation(problem.initial_solution)
            self._backend.on_success(
                np.asarray(last_repr, dtype=float),
                iterations=int(info.get("iterations", 0)),
                residual_norm=float(info.get("residual_norm", float("nan"))),
            )
        except Exception:
            pass
