"""Continuation stepping strategies.

This module provides the stepping module for the continuation package.
"""

from .base import _ContinuationStepBase
from .plain import _ContinuationPlainStep
from .np.base import _NaturalParameterStep
from .sc.base import _SecantStep

def make_natural_stepper(predict_fn):
    """Factory for a natural-parameter stepper.

    Returns a callable implementing (last, step) -> (prediction, step_hint).
    """
    return _NaturalParameterStep(predict_fn)


def make_secant_stepper(representation_fn, tangent_provider):
    """Factory for a secant stepper using an external tangent provider."""
    return _SecantStep(representation_fn, tangent_provider)

__all__ = [
    "_ContinuationStepBase",
    "_ContinuationPlainStep",
    "_NaturalParameterStep",
    "_SecantStep",
    "make_natural_stepper",
    "make_secant_stepper",
]