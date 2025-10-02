"""Continuation stepping strategies.

This module provides factories that build continuation stepping strategies per
problem. The factories mirror the corrector stepping helpers: each returns a
callable that accepts backend-provided support objects and yields a concrete
stepper instance.
"""

from __future__ import annotations

from typing import Callable

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

# Stepper factory: (predictor_fn, representation_fn, support) -> stepper
_ContinuationStepperFactory = Callable[[
    Callable[[object, _np.ndarray], _np.ndarray],
    Callable[[object], _np.ndarray],
    _ContinuationStepSupport | None
], _ContinuationStepBase]


def make_natural_stepper() -> _ContinuationStepperFactory:
    """Return a natural-parameter stepper factory."""

    def _factory(
        predictor_fn: Callable[[object, _np.ndarray], _np.ndarray],
        representation_fn: Callable[[object], _np.ndarray],
        support: _ContinuationStepSupport | None = None,
    ) -> _ContinuationStepBase:
        return _NaturalParameterStep(predictor_fn)

    return _factory


def make_secant_stepper() -> _ContinuationStepperFactory:
    """Return a secant stepper factory."""

    def _factory(
        predictor_fn: Callable[[object, _np.ndarray], _np.ndarray],
        representation_fn: Callable[[object], _np.ndarray],
        support: _ContinuationStepSupport | None = None,
    ) -> _ContinuationStepBase:
        if support is None or not isinstance(support, _SecantSupport):
            raise ValueError("Secant stepper requires _SecantSupport from backend")
        return _SecantStep(representation_fn, support.get_tangent)

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