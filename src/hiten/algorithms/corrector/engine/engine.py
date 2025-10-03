"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import \
    _OrbitCorrectionInterface
from hiten.algorithms.corrector.types import _OrbitCorrectionProblem
from hiten.algorithms.types.exceptions import (BackendError, ConvergenceError,
                                               EngineError)


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface.
    
    Parameters
    ----------
    backend : :class:`~hiten.algorithms.corrector.backends.newton._NewtonBackend`
        The backend for the orbit correction.
    interface : :class:`~hiten.algorithms.corrector.interfaces._OrbitCorrectionInterface`
        The interface for the orbit correction.
    """

    def __init__(self, *, backend: _NewtonBackend, interface: _OrbitCorrectionInterface) -> None:
        super().__init__(backend=backend, interface=interface)

    def _handle_backend_failure(
        self,
        exc: Exception,
        *,
        problem: _OrbitCorrectionProblem,
        call,
        interface,
    ) -> None:
        """Handle backend failure.
        
        Parameters
        ----------
        exc : Exception
            The exception.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The problem.
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call.
        interface : :class:`~hiten.algorithms.corrector.interfaces._OrbitCorrectionInterface`
            The interface.
        
        Raises
        ------
        :class:`~hiten.algorithms.types.exceptions.EngineError`
            If the orbit correction failed.
        """
        if isinstance(exc, (ConvergenceError, BackendError)):
            raise EngineError("Orbit correction failed") from exc
        raise EngineError("Unexpected error during orbit correction") from exc

    def _invoke_backend(self, call):
        """Invoke the backend.
        
        Parameters
        ----------
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.
        """
        request = call.request
        if request is None:
            return self._backend.run(**call.kwargs)
        return self._backend.run(request=request, **call.kwargs)

    def _after_backend_success(self, outputs, *, problem, domain_payload, interface) -> None:
        """Handle backend success.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.corrector.types.CorrectorOutput`
            The outputs from the backend.
        problem : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionProblem`
            The problem.
        domain_payload : :class:`~hiten.algorithms.corrector.types._OrbitCorrectionDomainPayload`
            The domain payload.
        interface : :class:`~hiten.algorithms.corrector.interfaces._OrbitCorrectionInterface`
            The interface.
        """
        x_corr = outputs.x_corrected
        iterations = outputs.iterations
        residual_norm = outputs.residual_norm
        try:
            self._backend.on_success(
                x_corr,
                iterations=int(iterations),
                residual_norm=float(residual_norm),
            )
            
        except Exception:
            pass
