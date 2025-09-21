"""
Abstract base class for continuation engines.
"""

from abc import ABC, abstractmethod

from hiten.algorithms.continuation.types import ContinuationResult


class _ContinuationEngine(ABC):
    """Provide an abstract base class for continuation engines."""

    @abstractmethod
    def solve(self, *args, **kwargs) -> ContinuationResult:
        """Solve the continuation problem and return a ContinuationResult."""
        ...
