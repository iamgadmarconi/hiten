"""Define the base class for correction engines.

This module provides the base class for correction engines.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from hiten.algorithms.corrector.types import CorrectionResult


class _CorrectionEngine(ABC):
    """Provide an abstract base class for correction engines.

    This class provides the base class for correction engines.
    """

    @abstractmethod
    def solve(self, orbit, cfg, *, forward: int) -> Tuple[np.ndarray, CorrectionResult, float]:
        """Solve the correction problem.

        This method solves the correction problem for a given orbit and configuration.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Orbit to be corrected.
        cfg : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            Configuration for the correction.
        forward : int
            Forward integration direction.
        """
        ...