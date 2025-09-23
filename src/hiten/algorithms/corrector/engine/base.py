"""Define the base class for correction engines.

This module provides the base class for correction engines.
"""

from abc import ABC, abstractmethod

from hiten.algorithms.corrector.types import (CorrectionResult,
                                              _CorrectionProblem)


class _CorrectionEngine(ABC):
    """Provide an abstract base class for correction engines.

    This class provides the base class for correction engines.
    """

    @abstractmethod
    def solve(self, problem: _CorrectionProblem) -> CorrectionResult:
        """Solve a composed correction problem and return engine results."""
        ...