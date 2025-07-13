"""
hiten.algorithms.continuation.predictors
===========================================

Concrete predictor classes that plug into
:pyclass:`hiten.algorithms.continuation.base._ContinuationEngine`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine
from hiten.algorithms.dynamics.utils.energy import crtbp_energy
from hiten.system.orbits.base import PeriodicOrbit, S


class _StateParameter(_ContinuationEngine):
    """Vary a single coordinate of the seed state by a constant increment.

    Examples
    --------
    >>> engine = _StateParameter(
    >>>     initial_orbit=halo0,
    >>>     state_index=S.Z,          # third component of state vector
    >>>     target=(halo0.initial_state[S.Z], 0.06),
    >>>     step=1e-4,
    >>>     corrector_kwargs=dict(tol=1e-12, max_attempts=250),
    >>> )
    >>> family = engine.run()
    """

    def __init__(
        self,
        *,
        initial_orbit: PeriodicOrbit,
        state: S | Sequence[S] | None = None,
        amplitude: bool | None = None,
        target: Sequence[float],
        step: float | Sequence[float] = 1e-4,
        corrector_kwargs: dict | None = None,
        max_orbits: int = 256,
    ) -> None:
        # Normalise *state* to a list
        if isinstance(state, S):
            state_list = [state]
        elif state is None:
            raise ValueError("state cannot be None after resolution")
        else:
            state_list = list(state)

        # Resolve amplitude flag
        if amplitude is None:
            try:
                amplitude = initial_orbit._continuation_config.amplitude
            except AttributeError:
                amplitude = False

        if amplitude and len(state_list) != 1:
            raise ValueError("Amplitude continuation supports exactly one state component.")

        if amplitude and state_list[0] not in (S.X, S.Y, S.Z):
            raise ValueError("Amplitude continuation is only supported for positional coordinates (X, Y, Z).")

        self._state_indices = np.array([s.value for s in state_list], dtype=int)

        # Parameter getter logic (returns np.ndarray)
        if amplitude:
            parameter_getter = lambda orb: np.asarray([float(getattr(orb, "amplitude"))])
        else:
            idxs = self._state_indices.copy()
            parameter_getter = lambda orb, idxs=idxs: np.asarray([float(orb.initial_state[i]) for i in idxs])

        super().__init__(
            initial_orbit=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_orbits=max_orbits,
        )

    def _predict(self, last_orbit: PeriodicOrbit, step: np.ndarray) -> np.ndarray:
        """Copy the state vector and increment the designated component(s)."""
        new_state = np.copy(last_orbit.initial_state)
        for idx, d in zip(self._state_indices, step):
            # Use base class helper to ensure reasonable step while preserving adaptive reduction
            d = self._clamp_step(d, reference_value=new_state[idx])
            new_state[idx] += d
        return new_state


class _FixedPeriod(_ContinuationEngine):
    def __init__(
        self,
        *,
        initial_orbit: PeriodicOrbit,
        target: "Sequence[float]",
        step: float = 1e-3,
        corrector_kwargs: dict | None = None,
        max_orbits: int = 256,
    ) -> None:
        # Continuation parameter (period)
        parameter_getter = lambda orb: np.asarray([float(orb.period)])

        super().__init__(
            initial_orbit=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_orbits=max_orbits,
        )

    def _predict(self, last_orbit: PeriodicOrbit, step: np.ndarray) -> np.ndarray:
        dT = float(step[0])
        T = float(last_orbit.period)
        if T == 0:
            raise ValueError("Last orbit has zero period - cannot continue in period.")

        scale = 1.0 - dT / T
        # Use base class helper to prevent pathological scaling while allowing adaptive reduction
        scale = self._clamp_scale(scale)

        new_state = np.copy(last_orbit.initial_state)
        new_state[3:6] *= scale  # scale vx, vy, vz
        return new_state


class _EnergyLevel(_ContinuationEngine):
    def __init__(
        self,
        *,
        initial_orbit: PeriodicOrbit,
        target: "Sequence[float]",
        step: float = 1e-4,
        use_jacobi: bool = False,
        corrector_kwargs: dict | None = None,
        max_orbits: int = 256,
    ) -> None:
        if use_jacobi:
            parameter_getter = lambda orb: np.asarray([float(orb.jacobi_constant)])
        else:
            parameter_getter = lambda orb: np.asarray([float(orb.energy)])

        self._use_jacobi = use_jacobi

        super().__init__(
            initial_orbit=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_orbits=max_orbits,
        )

    def _predict(self, last_orbit: PeriodicOrbit, step: np.ndarray) -> np.ndarray:
        dE = float(step[0])
        new_state = np.copy(last_orbit.initial_state)

        v = new_state[3:6]
        v_sq = float(np.dot(v, v))

        if v_sq < 1e-12:
            v_sq = 1e-12

        alpha = dE / v_sq

        max_alpha = 0.25
        if abs(alpha) > max_alpha:
            alpha = np.sign(alpha) * max_alpha

        scale = 1.0 + alpha
        scale = self._clamp_scale(scale, min_scale=0.8, max_scale=1.25)

        new_state[3:6] = v * scale

        current_E = last_orbit.energy
        scaled_E = crtbp_energy(new_state, last_orbit.mu)
        residual = dE - (scaled_E - current_E)

        if abs(residual) > 0.2 * abs(dE):
            grad = np.zeros(3)
            eps = 1e-5
            for i in range(3):
                up = np.copy(new_state)
                dn = np.copy(new_state)
                up[i] += eps
                dn[i] -= eps
                grad[i] = (crtbp_energy(up, last_orbit.mu) - crtbp_energy(dn, last_orbit.mu)) / (2 * eps)

            if np.linalg.norm(grad) > 0:
                direction = grad / np.linalg.norm(grad)
                new_state[:3] += 1e-4 * np.sign(residual) * direction

        return new_state
