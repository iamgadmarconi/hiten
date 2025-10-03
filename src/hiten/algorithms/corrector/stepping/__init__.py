"""
Define the stepping module for the corrector package.

This module provides the stepping module for the corrector package.
"""

from typing import Callable

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.types import NormFn, ResidualFn

from .armijo import _ArmijoLineSearch, _ArmijoStep
from .base import _CorrectorStepBase
from .plain import _CorrectorPlainStep


def make_plain_stepper() -> Callable[[ResidualFn, NormFn, float | None], CorrectorStepProtocol]:
    """Return a factory that builds a plain capped stepper per problem."""
    def _factory(residual_fn: ResidualFn, norm_fn: NormFn, max_delta: float | None) -> CorrectorStepProtocol:
        """Return a factory that builds a plain capped stepper per problem.
        
        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            The residual function.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            The norm function.
        max_delta : float or None
            The maximum delta.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
            The plain capped stepper.
        """
        return _CorrectorPlainStep()._build_line_searcher(residual_fn, norm_fn, max_delta)
    return _factory


def make_armijo_stepper(
    *,
    alpha_reduction: float = 0.5,
    min_alpha: float = 1e-4,
    armijo_c: float = 0.1,
) -> Callable[[ResidualFn, NormFn, float | None], CorrectorStepProtocol]:
    """Return a factory that builds an Armijo stepper per problem.

    Parameters
    ----------
    alpha_reduction : float, default=0.5
        Factor to reduce step size in backtracking.
    min_alpha : float, default=1e-4
        Minimum step size before giving up.
    armijo_c : float, default=0.1
        Armijo parameter for sufficient decrease condition.
    """
    def _factory(residual_fn: ResidualFn, norm_fn: NormFn, max_delta: float | None) -> CorrectorStepProtocol:
        """Return a factory that builds an Armijo stepper per problem.
        
        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            The residual function.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            The norm function.
        max_delta : float or None
            The maximum delta.
        """
        return _ArmijoStep(
            alpha_reduction=alpha_reduction,
            min_alpha=min_alpha,
            armijo_c=armijo_c,
        )._build_line_searcher(residual_fn, norm_fn, max_delta)
    return _factory

__all__ = [
    "_ArmijoStep",
    "_ArmijoLineSearch",
    "_CorrectorStepBase",
    "_CorrectorPlainStep",
    "make_plain_stepper",
    "make_armijo_stepper",
]
