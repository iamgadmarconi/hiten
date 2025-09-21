"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.types import (CorrectionResult,
                                              _CorrectionProblem)
from hiten.algorithms.utils.exceptions import (BackendError, ConvergenceError,
                                               EngineError)

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface."""

    def __init__(self, *, backend: _NewtonBackend, interface: _PeriodicOrbitCorrectorInterface | None = None) -> None:
        self._backend = backend
        self._interface = _PeriodicOrbitCorrectorInterface() if interface is None else interface

    def solve(self, orbit: "PeriodicOrbit", cfg: _OrbitCorrectionConfig) -> Tuple[CorrectionResult, float]:
        """Run correction and return corrected state, backend result, half-period.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.

        Returns
        -------
        Tuple[:class:`~hiten.algorithms.corrector.types.CorrectionResult`, float]
            backend result, half-period.
        """
        p0 = self._interface.initial_guess(orbit, cfg)
        fd_mode = bool(cfg.finite_difference)
        forward = cfg.forward
        residual, jacobian, to_full_state = self._interface.build_functions(
            orbit,
            cfg,
            forward,
            finite_difference=bool(fd_mode),
        )
        norm_fn = self._interface.norm_fn()

        problem = _CorrectionProblem(
            initial_guess=p0,
            residual_fn=residual,
            jacobian_fn=jacobian,
            norm_fn=norm_fn,
            tol=cfg.tol,
            max_attempts=cfg.max_attempts,
            max_delta=cfg.max_delta,
            fd_step=cfg.fd_step,
        )

        try:
            x_corr, info = self._backend.correct(
                x0=problem.initial_guess,
                residual_fn=problem.residual_fn,
                jacobian_fn=problem.jacobian_fn,
                norm_fn=problem.norm_fn,
                tol=problem.tol,
                max_attempts=problem.max_attempts,
                max_delta=problem.max_delta,
                fd_step=problem.fd_step,
            )
        except (ConvergenceError, BackendError) as exc:
            raise EngineError(
                f"Orbit correction failed for {getattr(orbit, 'family', type(orbit).__name__)} "
                f"(forward={getattr(cfg, 'forward', None)}): {exc}"
            ) from exc
        except Exception as exc:
            raise EngineError(
                f"Unexpected error during orbit correction for {getattr(orbit, 'family', type(orbit).__name__)}"
            ) from exc

        corrected_state = to_full_state(x_corr)
        half_period = self._interface.compute_half_period(orbit, corrected_state, cfg, forward)

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