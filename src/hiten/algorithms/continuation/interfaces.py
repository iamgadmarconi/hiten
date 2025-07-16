from typing import Callable, NamedTuple, Sequence

import numpy as np

from hiten.system.orbits.base import PeriodicOrbit, S


class _OrbitContinuationConfig(NamedTuple):
    state: S | None
    amplitude: bool = False
    getter: Callable[["PeriodicOrbit"], float] | None = None
    extra_params: dict | None = None


class _PeriodicOrbitContinuationInterface:
    def __init__(self, *, initial_orbit: PeriodicOrbit, parameter_getter: Callable[[PeriodicOrbit], "np.ndarray | float"],
        target: Sequence[Sequence[float] | float], step: float | Sequence[float] = 1e-4, corrector_kwargs: dict | None = None,
        max_orbits: int = 256, **kwargs) -> None:

        self._orbit_class = type(initial_orbit)
        self._libration_point = initial_orbit.libration_point

        self._getter = parameter_getter

        super().__init__(
            initial_solution=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_iters=max_orbits,
            **kwargs,
        )

    def _instantiate(self, representation: np.ndarray):  
        """Instantiate a *PeriodicOrbit* from a 6-component state vector."""

        return self._orbit_class(
            libration_point=self._libration_point,
            initial_state=representation,
        )

    def _correct(self, obj: PeriodicOrbit, **kwargs):  
        """Apply orbit correction in-place, return the corrected orbit."""

        obj.correct(**(kwargs or {}))
        return obj

    def _parameter(self, obj: PeriodicOrbit) -> np.ndarray:  
        """Return parameter as *1-D numpy array* (engine expects array-like)."""

        return np.asarray(self._getter(obj), dtype=float)
    

class _InvariantToriContinuationInterface:
    pass