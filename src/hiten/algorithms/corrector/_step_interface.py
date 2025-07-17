from abc import ABC, abstractmethod
from typing import Callable, Optional, Protocol

import numpy as np

from hiten.algorithms.corrector.line import (_ArmijoLineSearch,
                                             _LineSearchConfig)

ResidualFn = Callable[[np.ndarray], np.ndarray]
NormFn = Callable[[np.ndarray], float]


class _Stepper(Protocol):
    """Callable transforming a Newton step into an accepted update.

    Parameters
    ----------
    x : ndarray
        Current iterate.
    delta : ndarray
        Newton step direction.
    current_norm : float
        Norm of the residual at *x*.

    Returns
    -------
    x_new : ndarray
        Updated iterate.
    r_norm_new : float
        Norm of residual at *x_new*.
    alpha_used : float
        Step-size scaling actually employed (1.0 means full step).
    """

    def __call__(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        current_norm: float,
    ) -> tuple[np.ndarray, float, float]: ...


class _StepInterface(ABC):
    """Abstract interface for step-size/line-search strategies.
    """

    def __init__(self, **kwargs):
        # Allow clean cooperation in multiple-inheritance chains
        super().__init__(**kwargs)

    @abstractmethod
    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Return a :pydata:`_Stepper` object for the current problem."""


class _PlainStepInterface(_StepInterface):
    """Provide plain Newton updates with optional infinity-norm capping."""

    def _make_plain_stepper(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Return a plain stepper closure implementing the safeguard."""

        def _plain_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            # Optional safeguard
            if (max_delta is not None) and (not np.isinf(max_delta)):
                delta_norm = float(np.linalg.norm(delta, ord=np.inf))
                if delta_norm > max_delta:
                    delta = delta * (max_delta / delta_norm)

            x_new = x + delta
            r_norm_new = norm_fn(residual_fn(x_new))
            return x_new, r_norm_new, 1.0

        return _plain_step

    # Expose via abstract method name
    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        return self._make_plain_stepper(residual_fn, norm_fn, max_delta)


class _ArmijoStepInterface(_PlainStepInterface):

    _line_search_config: Optional[_LineSearchConfig]
    _use_line_search: bool

    def __init__(
        self,
        *,
        line_search_config: _LineSearchConfig | bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Process line-search configuration
        if line_search_config is None:
            self._line_search_config = None
            self._use_line_search = False
        elif isinstance(line_search_config, bool):
            if line_search_config:
                self._line_search_config = _LineSearchConfig()
                self._use_line_search = True
            else:
                self._line_search_config = None
                self._use_line_search = False
        else:
            self._line_search_config = line_search_config
            self._use_line_search = True

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        if not getattr(self, "_use_line_search", False):
            return self._make_plain_stepper(residual_fn, norm_fn, max_delta)

        cfg = self._line_search_config
        searcher = _ArmijoLineSearch(
            config=cfg._replace(residual_fn=residual_fn, norm_fn=norm_fn)
        )

        def _armijo_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            return searcher(x0=x, delta=delta, current_norm=current_norm)

        return _armijo_step 