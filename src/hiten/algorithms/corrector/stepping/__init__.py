"""
Define the stepping module for the corrector package.

This module provides the stepping module for the corrector package.
"""

from typing import Callable

from hiten.algorithms.corrector.config import _LineSearchConfig
from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.types import NormFn, ResidualFn

from .armijo import _ArmijoLineSearch, _ArmijoStep
from .base import _CorrectorStepBase
from .plain import _CorrectorPlainStep


def make_plain_stepper() -> Callable[[ResidualFn, NormFn, float | None], CorrectorStepProtocol]:
    """Return a factory that builds a plain capped stepper per problem."""
    def _factory(residual_fn: ResidualFn, norm_fn: NormFn, max_delta: float | None) -> CorrectorStepProtocol:
        return _CorrectorPlainStep()._build_line_searcher(residual_fn, norm_fn, max_delta)
    return _factory


def make_armijo_stepper(config: "_LineSearchConfig") -> Callable[[ResidualFn, NormFn, float | None], CorrectorStepProtocol]:
    """Return a factory that builds an Armijo stepper per problem.

    Parameters
    ----------
    config : :class:`~hiten.algorithms.corrector.config._LineSearchConfig`
        Configuration for Armijo line search.
    """
    def _factory(residual_fn: ResidualFn, norm_fn: NormFn, max_delta: float | None) -> CorrectorStepProtocol:
        return _ArmijoStep(line_search_config=config)._build_line_searcher(residual_fn, norm_fn, max_delta)
    return _factory

__all__ = [
    "_ArmijoStep",
    "_ArmijoLineSearch",
    "_CorrectorStepBase",
    "_CorrectorPlainStep",
    "make_plain_stepper",
    "make_armijo_stepper",
]
