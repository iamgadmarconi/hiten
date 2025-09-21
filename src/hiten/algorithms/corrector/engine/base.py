"""Define the base class for correction engines.

This module provides the base class for correction engines.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from hiten.algorithms.corrector.types import CorrectionResult


class _CorrectionEngine(ABC):
    """Provide an abstract base class for correction engines.

    This class provides the base class for correction engines.
    """

    @abstractmethod
    def solve(self, orbit, cfg) -> Tuple[CorrectionResult, float]:
        """Solve the correction problem.

        This method solves the correction problem for a given orbit and configuration.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.

        Returns
        -------
        Tuple[:class:`~hiten.algorithms.corrector.types.CorrectionResult`, float]
            backend result, half-period.
        """
        ...