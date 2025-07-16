from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.line import _LineSearchConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]
NormFn = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class _BaseCorrectionConfig:
    max_attempts: int = 50
    tol: float = 1e-10

    max_delta: float = 1e-2

    line_search_config: _LineSearchConfig | bool | None = True
    """If *True* uses the default line-search parameters, *False* disables the
    line-search, and passing an explicit :class:`_LineSearchConfig` allows full
    customisation."""

    finite_difference: bool = False  # Force FD approximation even if analytic


class _Corrector(ABC):
    """Generic iterative corrector / root-finder.

    The class represents the *algorithmic* aspect of an iterative correction
    scheme such as Newton-Raphson or quasi-Newton.  It operates on an
    *abstract* vector of parameters ``x`` and a user-supplied *residual
    function* ``R(x)``.  Concrete scientific objects (periodic orbits,
    invariant curves, …) should supply thin wrappers that translate their
    domain-specific state into such a vector representation.
    """

    # NOTE: subclasses are expected to document additional keyword arguments
    # (max_iter, tolerance, …) relevant to their specific strategy.

    @abstractmethod
    def correct(
        self,
        x0: np.ndarray,
        residual_fn: ResidualFn,
        *,
        jacobian_fn: JacobianFn | None = None,
        norm_fn: NormFn | None = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """Return *x* s.t. ``||R(x)||`` is below tolerance.

        Parameters
        ----------
        x0 : ndarray
            Initial guess for the parameter vector.
        residual_fn : callable
            Function computing the residual vector ``R(x)``.
        jacobian_fn : callable, optional
            If provided, a function returning the Jacobian ``J(x)``.
        norm_fn : callable, optional
            Custom norm to assess convergence (defaults to 2-norm).
        **kwargs
            Additional algorithm-specific parameters (see subclass docs).

        Returns
        -------
        x_corr : ndarray
            Corrected parameter vector.
        info : Any
            Optional auxiliary information (e.g. number of iterations,
            final residual norm).  The exact content is implementation-defined.
        """
        raise NotImplementedError