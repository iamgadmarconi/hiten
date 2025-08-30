from typing import Sequence

import numpy as np

from hiten.algorithms.continuation.interfaces import \
    _PeriodicOrbitContinuationInterface
from hiten.algorithms.continuation.strategies._algorithms import \
    _NaturalParameter
from hiten.algorithms.continuation.strategies._stepping import \
    _NaturalParameterStep
from hiten.algorithms.utils.types import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


class _StateParameter(_PeriodicOrbitContinuationInterface, _NaturalParameter):
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
        state: SynodicState | Sequence[SynodicState] | None = None,
        amplitude: bool | None = None,
        target: Sequence[float],
        step: float | Sequence[float] = 1e-4,
        corrector_kwargs: dict | None = None,
        max_orbits: int = 256,
    ) -> None:
        # Normalise *state* to a list
        if isinstance(state, SynodicState):
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

        if amplitude and state_list[0] not in (SynodicState.X, SynodicState.Y, SynodicState.Z):
            raise ValueError("Amplitude continuation is only supported for positional coordinates (X, Y, Z).")

        self._state_indices = np.array([s.value for s in state_list], dtype=int)

        # Parameter getter logic (returns np.ndarray)
        if amplitude:
            parameter_getter = lambda orb: np.asarray([float(getattr(orb, "amplitude"))])
        else:
            idxs = self._state_indices.copy()
            parameter_getter = lambda orb, idxs=idxs: np.asarray([float(orb.initial_state[i]) for i in idxs])

        # Predictor function that applies the step to selected state indices
        def _predict_state(orbit, step_vec):
            new_state = orbit.initial_state.copy()
            for idx, d in zip(self._state_indices, step_vec):
                new_state[idx] += d
            return new_state

        self._predict_state_fn = _predict_state

        super().__init__(
            initial_orbit=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_orbits=max_orbits,
        )

    # Override _make_stepper to supply the strategy
    def _make_stepper(self):
        return _NaturalParameterStep(self._predict_state_fn)


class _FixedPeriod(_PeriodicOrbitContinuationInterface, _NaturalParameter):
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Period continuation is not implemented yet.")


class _EnergyLevel(_PeriodicOrbitContinuationInterface, _NaturalParameter):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Energy continuation is not implemented yet.")