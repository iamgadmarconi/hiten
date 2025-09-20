"""Define the engine for the corrector module.

This module provides the engine for the corrector module.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
from hiten.algorithms.corrector.engine.base import _CorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitInterface
from hiten.algorithms.corrector.types import (CorrectionResult,
                                              _CorrectionProblem)

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _OrbitCorrectionEngine(_CorrectionEngine):
    """Engine orchestrating periodic orbit correction via a backend and interface."""

    def __init__(self, *, backend: _NewtonBackend, interface: _PeriodicOrbitInterface | None = None) -> None:
        self._backend = backend
        self._interface = _PeriodicOrbitInterface() if interface is None else interface

    def solve(
        self,
        orbit: "PeriodicOrbit",
        cfg : _OrbitCorrectionConfig,
        *,
        forward: int,
        tol: float | None = None,
        max_attempts: int | None = None,
        max_delta: float | None = None,
        finite_difference: bool | None = None,
    ) -> Tuple[np.ndarray, CorrectionResult, float]:
        """Run correction and return corrected state, backend result, half-period.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.
        forward : int
            Forward integration direction.
        tol : float | None
            Convergence tolerance for the correction.
        max_attempts : int | None
            Maximum number of correction attempts.
        max_delta : float | None
            Maximum step size for corrections.
        finite_difference : bool | None
            Use finite-difference Jacobian instead of analytical.
        """
        p0 = self._interface.initial_guess(orbit, cfg)
        fd_mode = cfg.finite_difference if finite_difference is None else finite_difference
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
            tol=cfg.tol if tol is None else tol,
            max_attempts=cfg.max_attempts if max_attempts is None else max_attempts,
            max_delta=cfg.max_delta if max_delta is None else max_delta,
            fd_step=1e-8,
        )

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

        corrected_state = to_full_state(x_corr)
        half_period = self._interface.compute_half_period(orbit, corrected_state, cfg, forward)

        result = CorrectionResult(
            converged=True,
            x_corrected=corrected_state,
            residual_norm=float(info.get("residual_norm", np.nan)),
            iterations=int(info.get("iterations", 0)),
        )
        return corrected_state, result, half_period