"""Orbit-specific continuation engine wiring backend and interface closures."""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.backends.base import _ContinuationBackend
from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.algorithms.continuation.engine.base import _ContinuationEngine
from hiten.algorithms.continuation.interfaces import \
    _PeriodicOrbitContinuationInterface
from hiten.algorithms.continuation.stepping import (make_natural_stepper,
                                                    make_secant_stepper)
from hiten.algorithms.continuation.types import (ContinuationResult,
                                                 _ContinuationProblem)
from hiten.algorithms.utils.exceptions import EngineError
from hiten.system.orbits.base import PeriodicOrbit


class _OrbitContinuationEngine(_ContinuationEngine):
    """
    Engine orchestrating periodic orbit continuation via a backend and interface.

    This class implements the predict-instantiate-correct-accept loop for periodic orbit continuation.
    It uses a backend to solve the continuation problem and an interface to build the necessary closures.

    Parameters
    ----------
    backend : :class:`~hiten.algorithms.continuation.backends.base._ContinuationBackend`
        The backend to use for solving the continuation problem.
    interface : :class:`~hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface`
        The interface to use for building the necessary closures.
    """
    def __init__(self, *, backend: _ContinuationBackend, interface: _PeriodicOrbitContinuationInterface | None = None) -> None:
        self._backend = backend
        self._interface = _PeriodicOrbitContinuationInterface() if interface is None else interface

    def solve(self, seed: PeriodicOrbit, cfg: _OrbitContinuationConfig) -> ContinuationResult:
        """
        Solve the periodic orbit continuation problem.

        This method solves the periodic orbit continuation problem using the backend and interface.
        It uses a backend to solve the continuation problem and an interface to build the necessary closures.

        Parameters
        ----------
        seed : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The seed orbit to use for the continuation problem.
        cfg : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration to use for the continuation problem.

        Returns
        -------
        :class:`~hiten.algorithms.continuation.types.ContinuationResult`
            The result of the continuation problem.

        Raises
        ------
        :class:`~hiten.algorithms.utils.exceptions.EngineError`
            If the continuation problem fails to solve.
        """
        try:
            # Build closures from stateless interface
            parameter_getter = self._interface.build_parameter_getter(seed, cfg)
            # Build instantiator and corrector closure here (engine responsibility)
            instantiator = self._interface.build_instantiator(seed)
            accepted_orbits: list[PeriodicOrbit] = []

            def _corrector(prediction):
                orbit = instantiator(prediction)
                x_corr, _halfT = orbit.correct(**(getattr(cfg, "extra_params", None) or {}))
                # Collect domain object with period set
                accepted_orbits.append(orbit)
                res = float(np.linalg.norm(np.asarray(x_corr, dtype=float) - np.asarray(prediction, dtype=float)))
                return np.asarray(x_corr, dtype=float), res, True

            corrector = _corrector

            # Choose stepper strategy based on config
            stepper_name = getattr(cfg, "stepper", "natural")
            if str(stepper_name).lower() == "secant":
                tangent_vec: np.ndarray | None = None

                def _tangent_provider():
                    return tangent_vec

                def _set_tangent(v: np.ndarray | None) -> None:
                    nonlocal tangent_vec
                    tangent_vec = None if v is None else np.asarray(v, dtype=float)

                stepper = make_secant_stepper(lambda v: np.asarray(v, dtype=float), _tangent_provider)
                set_tangent = _set_tangent
                # Pre-seed tangent using the natural predictor at the seed
                try:
                    predictor = self._interface.build_predictor(seed, cfg)
                    pred0 = np.asarray(predictor(self._interface.representation(seed), np.asarray(cfg.step, dtype=float)), dtype=float)
                    diff0 = (pred0 - self._interface.representation(seed)).ravel()
                    n0 = float(np.linalg.norm(diff0))
                    if n0 > 0.0:
                        _set_tangent(diff0 / n0)
                except Exception:
                    pass
            else:
                predictor = self._interface.build_predictor(seed, cfg)
                stepper = make_natural_stepper(predictor)
                set_tangent = None

            seed_repr = self._interface.representation(seed)

            # Construct a problem object (for clarity and traceability)
            _ContinuationProblem(
                initial_solution=seed,
                parameter_getter=lambda obj: parameter_getter(self._interface.representation(obj)),
                target=np.asarray(cfg.target, dtype=float),
                step=np.asarray(cfg.step, dtype=float),
                max_members=int(cfg.max_members),
                max_retries_per_step=int(cfg.max_retries_per_step),
                corrector_kwargs={},
            )

            # Normalize step direction toward target interval (natural-parameter parity)
            current_params = np.asarray(parameter_getter(seed_repr), dtype=float)
            target_arr = np.asarray(cfg.target, dtype=float)
            target_min = np.minimum(target_arr[0], target_arr[1])
            target_max = np.maximum(target_arr[0], target_arr[1])
            step_eff = np.asarray(cfg.step, dtype=float).copy()
            for i in range(step_eff.size):
                if (current_params[i] < target_min[i] and step_eff[i] < 0) or (
                    current_params[i] > target_max[i] and step_eff[i] > 0
                ):
                    step_eff[i] = -step_eff[i]

            # Early stop if already outside bounds (legacy parity)
            if np.any(current_params < target_min) or np.any(current_params > target_max):
                parameter_values = (current_params.copy(),)
                accepted_count = 1
                rejected_count = 0
                iterations = 0
                success_rate = 1.0
                instantiator = self._interface.build_instantiator(seed)
                return ContinuationResult(
                    accepted_count=accepted_count,
                    rejected_count=rejected_count,
                    success_rate=success_rate,
                    family=[seed],
                    parameter_values=parameter_values,
                    iterations=iterations,
                )

            family_repr, info = self._backend.solve(
                seed_repr=seed_repr,
                stepper=stepper,
                parameter_getter=parameter_getter,
                corrector=corrector,
                representation_of=lambda v: np.asarray(v, dtype=float),
                set_tangent=set_tangent,
                step=step_eff,
                target=np.asarray(cfg.target, dtype=float),
                max_members=int(cfg.max_members),
                max_retries_per_step=int(cfg.max_retries_per_step),
                shrink_policy=getattr(cfg, "shrink_policy", None),
                step_min=float(cfg.step_min),
                step_max=float(cfg.step_max),
            )
        except Exception as exc:
            raise EngineError("Orbit continuation failed") from exc

        # Package result
        parameter_values = info.get("parameter_values", tuple())
        accepted_count = int(info.get("accepted_count", len(family_repr)))
        rejected_count = int(info.get("rejected_count", 0))
        iterations = int(info.get("iterations", 0))
        denom = max(accepted_count + rejected_count, 1)
        success_rate = float(accepted_count) / float(denom)

        return ContinuationResult(
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            success_rate=success_rate,
            family=[seed] + accepted_orbits,
            parameter_values=tuple(parameter_values),
            iterations=iterations,
        )
