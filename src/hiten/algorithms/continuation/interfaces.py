"""Provide interface classes for domain-specific continuation algorithms."""

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from hiten.algorithms.continuation.config import OrbitContinuationConfig
from hiten.algorithms.continuation.options import OrbitContinuationOptions
from hiten.algorithms.continuation.stepping import (
    _ContinuationStepperFactory,
    make_natural_stepper,
    make_secant_stepper,
)
from hiten.algorithms.continuation.types import (
    ContinuationResult,
    _ContinuationProblem,
)
from hiten.algorithms.corrector.options import OrbitCorrectionOptions
from hiten.algorithms.types.core import _BackendCall, _HitenBaseInterface
from hiten.algorithms.types.states import SynodicState

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _PeriodicOrbitContinuationInterface(
    _HitenBaseInterface[
        OrbitContinuationConfig,
        _ContinuationProblem,
        ContinuationResult,
        tuple[list[np.ndarray], dict[str, object]],
    ]
):
    """Adapter wiring periodic-orbit families to continuation backends."""

    def __init__(self) -> None:
        super().__init__()

    def create_problem(self, *, domain_obj: "PeriodicOrbit", config: OrbitContinuationConfig, options: OrbitContinuationOptions) -> _ContinuationProblem:
        target_state = getattr(config, "state", None)
        if target_state is None:
            def parameter_getter(vec: np.ndarray) -> np.ndarray:
                return np.asarray(vec, dtype=float)
            state_indices = None
        else:
            if isinstance(target_state, SynodicState):
                indices = [int(target_state.value)]
            elif isinstance(target_state, Sequence):
                indices = [int(s.value if isinstance(s, SynodicState) else s) for s in target_state]
            else:
                indices = [int(target_state)]
            idx_array = np.asarray(indices, dtype=int)

            def parameter_getter(vec: np.ndarray) -> np.ndarray:
                return np.asarray(vec, dtype=float)[idx_array]

            state_indices = idx_array

        corrector_tol = 1e-12
        corrector_max_attempts = 50
        corrector_max_delta = 1e-2
        corrector_order = 8
        corrector_steps = 500
        corrector_forward = 1
        corrector_fd_step = 1e-8

        return _ContinuationProblem(
            initial_solution=domain_obj,
            parameter_getter=parameter_getter,
            target=options.target,
            step=options.step,
            max_members=options.max_members,
            max_retries_per_step=options.max_retries_per_step,
            representation_of=lambda obj: np.asarray(obj.initial_state if hasattr(obj, 'initial_state') else obj, dtype=float),
            shrink_policy=options.shrink_policy,
            step_min=options.step_min,
            step_max=options.step_max,
            stepper=config.stepper,
            state_indices=state_indices,
            corrector_tol=corrector_tol,
            corrector_max_attempts=corrector_max_attempts,
            corrector_max_delta=corrector_max_delta,
            corrector_order=corrector_order,
            corrector_steps=corrector_steps,
            corrector_forward=corrector_forward,
            corrector_fd_step=corrector_fd_step,
        )

    def to_backend_inputs(self, problem: _ContinuationProblem) -> _BackendCall:
        corrector = self._build_corrector(problem)
        predictor = self._predictor_from_problem(problem)
        representation_fn = problem.representation_of or (lambda v: np.asarray(v, dtype=float))
        seed_repr = representation_fn(problem.initial_solution)
        return _BackendCall(
            kwargs={
                "seed_repr": seed_repr,
                "predictor_fn": predictor,
                "parameter_getter": problem.parameter_getter,
                "corrector": corrector,
                "representation_of": representation_fn,
                "step": np.asarray(problem.step, dtype=float),
                "target": np.asarray(problem.target, dtype=float),
                "max_members": int(problem.max_members),
                "max_retries_per_step": int(problem.max_retries_per_step),
                "shrink_policy": problem.shrink_policy,
                "step_min": float(problem.step_min),
                "step_max": float(problem.step_max),
            }
        )

    def _build_corrector(self, problem: _ContinuationProblem) -> Callable[[np.ndarray], tuple[np.ndarray, float, bool]]:
        domain_obj = problem.initial_solution

        def _correct(prediction: np.ndarray) -> tuple[np.ndarray, float, bool]:
            orbit = self._instantiate(domain_obj, prediction)
            from hiten.algorithms.types.options import (
                ConvergenceOptions,
                CorrectionOptions,
                IntegrationOptions,
                NumericalOptions,
            )
            corrector_options = OrbitCorrectionOptions(
                base=CorrectionOptions(
                    convergence=ConvergenceOptions(
                        tol=problem.corrector_tol,
                        max_attempts=problem.corrector_max_attempts,
                        max_delta=problem.corrector_max_delta,
                    ),
                    integration=IntegrationOptions(
                        order=problem.corrector_order,
                        steps=problem.corrector_steps,
                    ),
                    numerical=NumericalOptions(
                        fd_step=problem.corrector_fd_step,
                    ),
                ),
                forward=problem.corrector_forward,
            )
            x_corr, _ = orbit.correct(options=corrector_options)
            residual = float(np.linalg.norm(np.asarray(x_corr, dtype=float) - prediction))
            return np.asarray(x_corr, dtype=float), residual, True

        return _correct

    def _instantiate(self, domain_obj: "PeriodicOrbit", representation: np.ndarray) -> "PeriodicOrbit":
        orbit_cls = type(domain_obj)
        lp = getattr(domain_obj, "libration_point", None)
        orbit = orbit_cls(libration_point=lp, initial_state=np.asarray(representation, dtype=float))
        if domain_obj.period is not None:
            orbit.period = domain_obj.period
        return orbit

    def _predictor_from_problem(self, problem: _ContinuationProblem) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if problem.state_indices is None:
            return lambda last, step: np.asarray(last, dtype=float) + np.asarray(step, dtype=float)
        
        idx_arr = problem.state_indices

        def _predict(last: np.ndarray, step: np.ndarray) -> np.ndarray:
            last = np.asarray(last, dtype=float).copy()
            step = np.asarray(step, dtype=float)
            for idx, d in zip(idx_arr, step):
                last[idx] += d
            return last

        return _predict

    def to_domain(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem) -> dict[str, object]:
        family_repr, info = outputs
        info = dict(info)
        info.setdefault("accepted_count", len(family_repr))
        info.setdefault("parameter_values", tuple())
        return info

    def to_results(self, outputs: tuple[list[np.ndarray], dict[str, object]], *, problem: _ContinuationProblem, domain_payload: Any = None) -> ContinuationResult:
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
