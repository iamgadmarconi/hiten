from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine


class _NaturalParameter(_ContinuationEngine, ABC):
    """Abstract base class for natural-parameter continuation algorithms."""

    @abstractmethod
    def _predict(self, last_solution: object, step: np.ndarray) -> np.ndarray:
        """Return a predicted representation for the next solution."""
        pass