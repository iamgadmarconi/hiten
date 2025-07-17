from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from hiten.algorithms.corrector.line import (_ArmijoLineSearch,
                                             _LineSearchConfig)

ResidualFn = Callable[[np.ndarray], np.ndarray]
NormFn = Callable[[np.ndarray], float]


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
    ) -> Optional[_ArmijoLineSearch]:
        """Return a stepper object for the current problem (may be *None*)."""


class _ArmijoStepInterface(_StepInterface):

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
    ) -> Optional[_ArmijoLineSearch]:
        if not getattr(self, "_use_line_search", False):
            return None

        cfg = self._line_search_config
        return _ArmijoLineSearch(config=cfg._replace(residual_fn=residual_fn, norm_fn=norm_fn)) 