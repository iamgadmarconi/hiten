"""Provide interface classes for domain-specific continuation algorithms.

This module provides interface classes that adapt the generic continuation
engine to specific problem domains in dynamical systems. These interfaces
implement the abstract methods required by the continuation framework for
particular types of solutions (periodic orbits, invariant tori, etc.).

The interfaces serve as mix-ins that provide domain-specific implementations
of instantiation, correction, and parameter extraction methods, allowing
the generic continuation algorithm to work with different solution types.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.continuation.engine`
    Continuation engines that these interfaces work with.
:mod:`~hiten.system.orbits`
    Periodic orbit classes used by orbit continuation.
:mod:`~hiten.algorithms.corrector`
    Correction algorithms used by continuation interfaces.
"""

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.algorithms.continuation.stepping import (make_natural_stepper,
                                                    make_secant_stepper)
from hiten.algorithms.continuation.types import (ContinuationResult,
                                                 _ContinuationProblem)
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface
from hiten.algorithms.types.states import SynodicState

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitContinuationInterface(
    _HitenBaseInterface[
        _OrbitContinuationConfig,
        _ContinuationProblem,
        ContinuationResult,
        tuple[list[np.ndarray], dict[str, object]],
    ]
):
    """Adapter wiring periodic-orbit families to continuation backends."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(self, *, domain_obj: "PeriodicOrbit", config: _OrbitContinuationConfig) -> _ContinuationProblem:
        """Create a continuation problem.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object to continue.
        config : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration for the continuation problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The continuation problem.
        """
        parameter_getter = self._parameter_getter(config)
        state_indices = self._get_state_indices(config)
        return _ContinuationProblem(
            initial_solution=domain_obj,
            parameter_getter=parameter_getter,
            target=np.asarray(config.target, dtype=float),
            step=np.asarray(config.step, dtype=float),
            max_members=int(config.max_members),
            max_retries_per_step=int(config.max_retries_per_step),
            corrector_kwargs=config.extra_params or {},
            representation_of=lambda obj: np.asarray(obj, dtype=float),
            shrink_policy=config.shrink_policy,
            step_min=float(config.step_min),
            step_max=float(config.step_max),
            stepper=config.stepper,
            state_indices=state_indices,
        )

    def to_backend_inputs(self, problem: _ContinuationProblem) -> _BackendCall:
        """Convert the continuation problem to backend inputs.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The continuation problem.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend inputs.
        """
        domain_obj = problem.initial_solution

        stepper_type = problem.stepper
        
        predictor = self._predictor_from_problem(problem)
        domain_obj_repr = self._representation(domain_obj)
        
        step_eff = np.asarray(problem.step, dtype=float)
        
        # Create tangent function for secant stepper
        _current_tangent = None
        _seeded = False
        
        if str(stepper_type).lower() == "secant":
            # Calculate initial tangent
            try:
                pred0 = predictor(domain_obj_repr, step_eff)
                diff0 = (np.asarray(pred0, dtype=float) - domain_obj_repr).ravel()
                norm0 = float(np.linalg.norm(diff0))
                initial_tangent = None if norm0 == 0.0 else diff0 / norm0
            except Exception:
                initial_tangent = None
            
            _current_tangent = initial_tangent
            
            def tangent_fn() -> np.ndarray | None:
                """Get the tangent.
                
                Returns
                -------
                np.ndarray | None
                    The tangent.
                """
                nonlocal _current_tangent, _seeded
                # First try to get updated tangent from backend
                if hasattr(self, "_backend") and self._backend is not None and hasattr(self._backend, "get_tangent"):
                    backend_tangent = self._backend.get_tangent()
                    if backend_tangent is not None:
                        _current_tangent = backend_tangent
                        return _current_tangent
                
                # If we have an initial tangent and haven't seeded the backend yet, do it now
                if not _seeded and initial_tangent is not None and hasattr(self, "_backend") and self._backend is not None and hasattr(self._backend, "seed_tangent"):
                    try:
                        self._backend.seed_tangent(initial_tangent)
                        _seeded = True
                    except Exception:
                        pass
                
                return _current_tangent
        else:
            tangent_fn = None
        
        def update_tangent(new_tangent: np.ndarray) -> None:
            nonlocal _current_tangent
            _current_tangent = new_tangent
        
        stepper = self._make_stepper_from_problem(problem, predictor, tangent_fn)

        def corrector(prediction: np.ndarray) -> tuple[np.ndarray, float, bool]:
            """Correct the prediction.
            
            Parameters
            ----------
            prediction : np.ndarray
                The prediction to correct.
            
            Returns
            -------
            tuple[np.ndarray, float, bool]
                The corrected prediction, the residual, and whether the correction was successful.
            """
            orbit = self._instantiate(domain_obj, prediction)
            x_corr, _ = orbit.correct(**(problem.corrector_kwargs or {}))
            residual = float(np.linalg.norm(np.asarray(x_corr, dtype=float) - prediction))
            
            # Update tangent for secant stepper after successful correction
            if str(stepper_type).lower() == "secant" and tangent_fn is not None:
                # Get the last accepted solution from the backend if available
                if hasattr(self, "_backend") and self._backend is not None and hasattr(self._backend, "_last_accepted"):
                    last_accepted = self._backend._last_accepted
                    if last_accepted is not None:
                        # Compute secant between last accepted and current corrected solution
                        diff = (np.asarray(x_corr, dtype=float) - np.asarray(last_accepted, dtype=float)).ravel()
                        norm = float(np.linalg.norm(diff))
                        if norm > 0.0:
                            new_tangent = diff / norm
                            # Update the tangent in the closure
                            update_tangent(new_tangent)
                            # Also update the backend's tangent
                            if hasattr(self._backend, "seed_tangent"):
                                try:
                                    self._backend.seed_tangent(new_tangent)
                                except Exception:
                                    pass
                
                # Store the corrected solution for next iteration
                if hasattr(self, "_backend") and self._backend is not None:
                    self._backend._last_accepted = np.asarray(x_corr, dtype=float).copy()
            
            return np.asarray(x_corr, dtype=float), residual, True

        return _BackendCall(
            kwargs={
                "seed_repr": domain_obj_repr,
                "stepper": stepper,
                "parameter_getter": problem.parameter_getter,
                "corrector": corrector,
                "representation_of": problem.representation_of or (lambda v: np.asarray(v, dtype=float)),
                "step": step_eff,
                "target": np.asarray(problem.target, dtype=float),
                "max_members": int(problem.max_members),
                "max_retries_per_step": int(problem.max_retries_per_step),
                "shrink_policy": problem.shrink_policy,
                "step_min": float(problem.step_min),
                "step_max": float(problem.step_max),
            }
        )

    def to_domain(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem) -> dict[str, object]:
        """Convert the continuation outputs to a domain object.
        
        Parameters
        ----------
        outputs : tuple[list[np.ndarray], dict[str, object]]
            The continuation outputs.
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The continuation problem.
        
        Returns
        -------
        dict[str, object]
            The domain object.
        """
        family_repr, info = outputs
        info = dict(info)
        info.setdefault("accepted_count", len(family_repr))
        info.setdefault("parameter_values", tuple())
        return info

    def to_results(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem, domain_payload: Any = None) -> ContinuationResult:
        """Convert the continuation outputs to a results object.
        
        Parameters
        ----------
        outputs : tuple[list[np.ndarray], dict[str, object]]
            The continuation outputs.
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The continuation problem.
        domain_payload : Any, optional
            The domain payload.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.types.ContinuationResult`
            The results object.
        """
        family_repr, info = outputs
        info = dict(info)
        accepted_count = int(info.get("accepted_count", len(family_repr)))
        rejected_count = int(info.get("rejected_count", 0))
        iterations = int(info.get("iterations", 0))
        parameter_values = tuple(info.get("parameter_values", tuple()))
        denom = max(accepted_count + rejected_count, 1)
        success_rate = float(accepted_count) / float(denom)

        family = [problem.initial_solution]
        for repr_vec in family_repr[1:]:
            orbit = self._instantiate(problem.initial_solution, repr_vec)
            family.append(orbit)
        
        return ContinuationResult(
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            success_rate=success_rate,
            family=tuple(family),
            parameter_values=parameter_values,
            iterations=iterations,
        )

    def _representation(self, orbit) -> np.ndarray:
        """Convert the orbit to a representation.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        
        Returns
        -------
        np.ndarray
            The representation.
        """
        return np.asarray(orbit.initial_state, dtype=float).copy()

    def _instantiate(self, domain_obj: "PeriodicOrbit", representation: np.ndarray):
        """Instantiate an orbit from a representation.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.
        representation : np.ndarray
            The representation.
        
        Returns
        -------
        :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.
        """
        orbit_cls = type(domain_obj)
        lp = getattr(domain_obj, "libration_point", None)
        orbit = orbit_cls(libration_point=lp, initial_state=np.asarray(representation, dtype=float))
        # Copy the period from the domain_obj orbit if it has one
        if domain_obj.period is not None:
            orbit.period = domain_obj.period
        return orbit

    def _parameter_getter(self, cfg: _OrbitContinuationConfig) -> Callable[[np.ndarray], np.ndarray]:
        """Get the parameter from the representation.
        
        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            The parameter getter.
        """
        state = getattr(cfg, "state", None)
        if state is None:
            return lambda repr_vec: np.asarray(repr_vec, dtype=float)
        if isinstance(state, SynodicState):
            indices = [int(state.value)]
        elif isinstance(state, Sequence):
            indices = [int(s.value) if isinstance(s, SynodicState) else int(s) for s in state]
        else:
            indices = [int(state)]
        idx_arr = np.asarray(indices, dtype=int)
        return lambda repr_vec: np.asarray(repr_vec, dtype=float)[idx_arr]

    def _predictor(self, cfg: _OrbitContinuationConfig) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Get the predictor.
        
        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration.
        
        Returns
        -------
        Callable[[np.ndarray, np.ndarray], np.ndarray]
            The predictor.
        """
        state = getattr(cfg, "state", None)
        if state is None:
            return lambda last, step: np.asarray(last, dtype=float) + np.asarray(step, dtype=float)
        if isinstance(state, SynodicState):
            indices = [int(state.value)]
        elif isinstance(state, Sequence):
            indices = [int(s.value) if isinstance(s, SynodicState) else int(s) for s in state]
        else:
            indices = [int(state)]
        idx_arr = np.asarray(indices, dtype=int)

        def _predict(last: np.ndarray, step: np.ndarray) -> np.ndarray:
            """Predict the next state.
            
            Parameters
            ----------
            last : np.ndarray
                The last state.
            step : np.ndarray
                The step size.
            
            Returns
            -------
            np.ndarray
                The next state.
            """
            last = np.asarray(last, dtype=float).copy()
            step = np.asarray(step, dtype=float)
            for idx, d in zip(idx_arr, step):
                last[idx] += d
            return last

        return _predict

    def _get_state_indices(self, config: _OrbitContinuationConfig) -> np.ndarray:
        """Get the state indices.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration.
        
        Returns
        -------
        np.ndarray
            The state indices.
        """
        state = getattr(config, "state", None)
        if state is None:
            return None
        if isinstance(state, SynodicState):
            indices = [int(state.value)]
        elif isinstance(state, Sequence):
            indices = [int(s.value) if isinstance(s, SynodicState) else int(s) for s in state]
        else:
            indices = [int(state)]
        return np.asarray(indices, dtype=int)

    def _predictor_from_problem(self, problem: _ContinuationProblem) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Get the predictor from the problem.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The problem.
        
        Returns
        -------
        Callable[[np.ndarray, np.ndarray], np.ndarray]
            The predictor.
        """
        if problem.state_indices is None:
            return lambda last, step: np.asarray(last, dtype=float) + np.asarray(step, dtype=float)
        
        idx_arr = problem.state_indices

        def _predict(last: np.ndarray, step: np.ndarray) -> np.ndarray:
            """Predict the next state.
            
            Parameters
            ----------
            last : np.ndarray
                The last state.
            step : np.ndarray
                The step size.
            
            Returns
            -------
            np.ndarray
                The next state.
            """
            last = np.asarray(last, dtype=float).copy()
            step = np.asarray(step, dtype=float)
            for idx, d in zip(idx_arr, step):
                last[idx] += d
            return last

        return _predict

    def _make_stepper_from_problem(self, problem: _ContinuationProblem, predictor: Callable[[np.ndarray, np.ndarray], np.ndarray], tangent_fn: Callable[[], np.ndarray | None] | None = None):
        """Make a stepper from the problem.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.continuation.types._ContinuationProblem`
            The problem.
        predictor : Callable[[np.ndarray, np.ndarray], np.ndarray]
            The predictor.
        tangent_fn : Callable[[], np.ndarray | None] | None, optional
            The tangent function.

        Returns
        -------
        :class:`~hiten.algorithms.continuation.stepping.base._ContinuationStepBase`
            The stepper.
        """
        
        if str(problem.stepper).lower() == "secant":
            return make_secant_stepper(lambda v: np.asarray(v, dtype=float), tangent_fn)
        return make_natural_stepper(predictor)

    def _make_stepper(self, cfg: _OrbitContinuationConfig, predictor: Callable[[np.ndarray, np.ndarray], np.ndarray], tangent_fn: Callable[[], np.ndarray | None] | None = None):
        """Make a stepper from the configuration.
        
        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The configuration.
        predictor : Callable[[np.ndarray, np.ndarray], np.ndarray]
            The predictor.
        tangent_fn : Callable[[], np.ndarray | None] | None, optional
            The tangent function.

        Returns
        -------
        :class:`~hiten.algorithms.continuation.stepping.base._ContinuationStepBase`
            The stepper.
        """
        
        if str(cfg.stepper).lower() == "secant":
            return make_secant_stepper(lambda v: np.asarray(v, dtype=float), tangent_fn)
        return make_natural_stepper(predictor)
