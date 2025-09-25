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

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.algorithms.continuation.stepping import (make_natural_stepper,
                                                    make_secant_stepper)
from hiten.algorithms.continuation.types import (ContinuationResult,
                                                 _ContinuationProblem)
from hiten.algorithms.types.core import BackendCall, _HitenBaseInterface
from hiten.algorithms.types.states import SynodicState

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitContinuationInterface(
    _HitenBaseInterface[
        "PeriodicOrbit",
        _OrbitContinuationConfig,
        _ContinuationProblem,
        ContinuationResult,
        tuple[list[np.ndarray], dict[str, object]],
    ]
):
    """Adapter wiring periodic-orbit families to continuation backends."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(self, *, seed: "PeriodicOrbit", config: _OrbitContinuationConfig) -> _ContinuationProblem:
        parameter_getter = self._parameter_getter(config)
        return _ContinuationProblem(
            initial_solution=seed,
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
            cfg=config,
        )

    def to_backend_inputs(self, problem: _ContinuationProblem) -> BackendCall:
        cfg = problem.cfg
        assert cfg is not None
        seed = problem.initial_solution

        predictor = self._predictor(cfg)
        seed_repr = self._representation(seed)
        
        step_eff = np.asarray(cfg.step, dtype=float)
        
        # Create tangent function for secant stepper
        _current_tangent = None
        _seeded = False
        
        if str(cfg.stepper).lower() == "secant":
            # Calculate initial tangent
            try:
                pred0 = predictor(seed_repr, step_eff)
                diff0 = (np.asarray(pred0, dtype=float) - seed_repr).ravel()
                norm0 = float(np.linalg.norm(diff0))
                initial_tangent = None if norm0 == 0.0 else diff0 / norm0
            except Exception:
                initial_tangent = None
            
            _current_tangent = initial_tangent
            
            def tangent_fn() -> np.ndarray | None:
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
        
        stepper = self._make_stepper(cfg, predictor, tangent_fn)

        def corrector(prediction: np.ndarray) -> tuple[np.ndarray, float, bool]:
            orbit = self._instantiate(seed, prediction)
            x_corr, _ = orbit.correct(**(cfg.extra_params or {}))
            residual = float(np.linalg.norm(np.asarray(x_corr, dtype=float) - prediction))
            
            # Update tangent for secant stepper after successful correction
            if str(cfg.stepper).lower() == "secant" and tangent_fn is not None:
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

        return BackendCall(
            kwargs={
                "seed_repr": seed_repr,
                "stepper": stepper,
                "parameter_getter": problem.parameter_getter,
                "corrector": corrector,
                "representation_of": problem.representation_of or (lambda v: np.asarray(v, dtype=float)),
                "step": step_eff,
                "target": np.asarray(cfg.target, dtype=float),
                "max_members": int(cfg.max_members),
                "max_retries_per_step": int(cfg.max_retries_per_step),
                "shrink_policy": cfg.shrink_policy,
                "step_min": float(cfg.step_min),
                "step_max": float(cfg.step_max),
            }
        )

    def to_domain(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem) -> dict[str, object]:
        family_repr, info = outputs
        info = dict(info)
        info.setdefault("accepted_count", len(family_repr))
        info.setdefault("parameter_values", tuple())
        return info

    def to_results(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem) -> ContinuationResult:
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
        return np.asarray(orbit.initial_state, dtype=float).copy()

    def _instantiate(self, seed: "PeriodicOrbit", representation: np.ndarray):
        orbit_cls = type(seed)
        lp = getattr(seed, "libration_point", None)
        orbit = orbit_cls(libration_point=lp, initial_state=np.asarray(representation, dtype=float))
        # Copy the period from the seed orbit if it has one
        if seed.period is not None:
            orbit.period = seed.period
        return orbit

    def _parameter_getter(self, cfg: _OrbitContinuationConfig) -> Callable[[np.ndarray], np.ndarray]:
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
            last = np.asarray(last, dtype=float).copy()
            step = np.asarray(step, dtype=float)
            for idx, d in zip(idx_arr, step):
                last[idx] += d
            return last

        return _predict

    def _make_stepper(self, cfg: _OrbitContinuationConfig, predictor: Callable[[np.ndarray, np.ndarray], np.ndarray], tangent_fn: Callable[[], np.ndarray | None] | None = None):
        if str(cfg.stepper).lower() == "secant":
            return make_secant_stepper(lambda v: np.asarray(v, dtype=float), tangent_fn)
        return make_natural_stepper(predictor)
