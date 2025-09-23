"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.types import (OrbitCorrectionResult,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.utils.exceptions import (BackendError, ConvergenceError,
                                               EngineError)


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface."""

    def __init__(self, *, backend: _NewtonBackend, interface: _PeriodicOrbitCorrectorInterface | None = None) -> None:
        self._backend = backend
        self._interface = _PeriodicOrbitCorrectorInterface() if interface is None else interface

    def solve(self, problem: _OrbitCorrectionProblem) -> OrbitCorrectionResult:
        """Run correction from a composed Problem and return engine results.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.corrector.types._CorrectionProblem`
            The problem to solve.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.OrbitCorrectionResult`
            Engine-level correction results containing backend output and half-period.
        """

        try:
            x_corr, info = self._backend.correct(
                x0=problem.initial_guess,
                residual_fn=problem.residual_fn,
                jacobian_fn=problem.jacobian_fn,
                norm_fn=problem.norm_fn,
                stepper_factory=problem.stepper_factory,
                tol=problem.cfg.tol,
                max_attempts=problem.cfg.max_attempts,
                max_delta=problem.cfg.max_delta,
                fd_step=problem.cfg.fd_step,
            )
        except (ConvergenceError, BackendError) as exc:
            raise EngineError("Orbit correction failed") from exc
        except Exception as exc:
            raise EngineError("Unexpected error during orbit correction") from exc

        corrected_state = self._interface._to_full_state(problem.orbit.initial_state, list(problem.cfg.control_indices), x_corr)
        half_period = self._interface.compute_half_period(problem.orbit, corrected_state, problem.cfg, problem.cfg.forward)

        try:
            self._backend.on_success(
                corrected_state,
                iterations=int(info.get("iterations", 0)),
                residual_norm=float(info.get("residual_norm", np.nan)),
            )
        except Exception:
            pass

        return OrbitCorrectionResult(
            converged=True,
            x_corrected=corrected_state,
            residual_norm=float(info.get("residual_norm", np.nan)),
            iterations=int(info.get("iterations", 0)),
            half_period=half_period,
        )
