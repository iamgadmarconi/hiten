"""Continuation stepping strategies.

This module provides factories that build continuation stepping strategies per
problem. The factories mirror the corrector stepping helpers: each returns a
callable that accepts backend-provided support objects and yields a concrete
stepper instance.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as _np

from hiten.algorithms.continuation.stepping.support import (
    _ContinuationStepSupport,
    _SecantSupport,
)

from hiten.algorithms.continuation.stepping.base import (
    _StepProposal,
    _ContinuationStepBase,
)

from hiten.algorithms.continuation.stepping.np.base import _NaturalParameterStep
from hiten.algorithms.continuation.stepping.plain import _ContinuationPlainStep
from hiten.algorithms.continuation.stepping.sc.base import _SecantStep

# Stepper factory: (stepper_fn, support, seed, step, predictor_for_tangent, step_params...) -> stepper
_ContinuationStepperFactory = Callable[[
    Callable,  # stepper_fn (predictor for natural, repr for secant)
    _ContinuationStepSupport | None,
    _np.ndarray,  # seed_repr
    _np.ndarray,  # initial_step  
    Callable,  # predictor_fn (only used for secant initial tangent)
    float,  # step_min
    float,  # step_max
    Optional[Callable[[_np.ndarray], _np.ndarray]]  # shrink_policy
], _ContinuationStepBase]


def make_natural_stepper() -> _ContinuationStepperFactory:
    """Return a natural-parameter stepper factory."""

    def _factory(
        stepper_fn: Callable[[object, _np.ndarray], _np.ndarray],
        support: _ContinuationStepSupport | None,
        seed_repr: _np.ndarray,
        initial_step: _np.ndarray,
        predictor_for_tangent: Callable,
        step_min: float,
        step_max: float,
        shrink_policy: Optional[Callable[[_np.ndarray], _np.ndarray]],
    ) -> _ContinuationStepBase:
        return _NaturalParameterStep(
            stepper_fn,
            step_min=step_min,
            step_max=step_max,
            shrink_policy=shrink_policy,
        )

    return _factory


def make_secant_stepper() -> _ContinuationStepperFactory:
    """Return a secant stepper factory."""

    def _factory(
        stepper_fn: Callable[[object], _np.ndarray],
        support: _ContinuationStepSupport | None,
        seed_repr: _np.ndarray,
        initial_step: _np.ndarray,
        predictor_fn: Callable[[object, _np.ndarray], _np.ndarray],
        step_min: float,
        step_max: float,
        shrink_policy: Optional[Callable[[_np.ndarray], _np.ndarray]],
    ) -> _ContinuationStepBase:
        if support is None or not isinstance(support, _SecantSupport):
            raise ValueError("Secant stepper requires _SecantSupport from backend")
        
        # Compute initial tangent using predictor_fn
        try:
            repr_seed = stepper_fn(seed_repr)
            pred0 = predictor_fn(repr_seed, initial_step)
            diff0 = (_np.asarray(pred0, dtype=float) - _np.asarray(repr_seed, dtype=float)).ravel()
            norm0 = float(_np.linalg.norm(diff0))
            initial_tangent = None if norm0 == 0.0 else diff0 / norm0
        except Exception:
            initial_tangent = None
        
        # Seed the support
        support.seed(initial_tangent)
        
        return _SecantStep(
            stepper_fn,
            support.get_tangent,
            step_min=step_min,
            step_max=step_max,
            shrink_policy=shrink_policy,
        )

    return _factory


__all__ = [
    "_ContinuationStepBase",
    "_ContinuationPlainStep",
    "_NaturalParameterStep",
    "_SecantStep",
    "_ContinuationStepperFactory",
    "make_natural_stepper",
    "make_secant_stepper",
    "StepProposal",
]