"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

from typing import Tuple

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.types import (CorrectionResult,
                                              _OrbitCorrectionProblem)
from hiten.algorithms.utils.exceptions import (BackendError, ConvergenceError,
                                               EngineError)


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface."""

    def __init__(self, *, backend: _NewtonBackend, interface: _PeriodicOrbitCorrectorInterface | None = None) -> None:
        self._backend = backend
        self._interface = _PeriodicOrbitCorrectorInterface() if interface is None else interface

    def solve(self, problem: _OrbitCorrectionProblem) -> Tuple[CorrectionResult, float]:
        """Run correction from a composed Problem and return backend result, half-period.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.corrector.types._CorrectionProblem`
            The problem to solve.

        Returns
        -------
        Tuple[:class:`~hiten.algorithms.corrector.types.CorrectionResult`, float]
            The correction result and half-period.
        """

        try:
            x_corr, info = self._backend.correct(
                x0=problem.initial_guess,
                residual_fn=problem.residual_fn,
                jacobian_fn=problem.jacobian_fn,
                norm_fn=problem.norm_fn,
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

        result = CorrectionResult(
            converged=True,
            x_corrected=corrected_state,
            residual_norm=float(info.get("residual_norm", np.nan)),
            iterations=int(info.get("iterations", 0)),
        )

        try:
            self._backend.on_success(
                corrected_state,
                iterations=result.iterations,
                residual_norm=result.residual_norm,
            )
        except Exception:
            pass

        return result, half_period
