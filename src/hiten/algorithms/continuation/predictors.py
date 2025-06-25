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
from hiten.system.orbits.base import PeriodicOrbit, S


class _NaturalParameter(_ContinuationEngine):
    """Vary a single coordinate of the seed state by a constant increment.

    Examples
    --------
    >>> engine = _NaturalParameter(
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
            new_state[idx] += d
        return new_state
